"""
AcousticTokenizer — A1 of the thesis adaptive-tokenization stack.

Operates on full-dimensional segment embeddings ``X ∈ R^{n, d}`` (e.g. the
192-D mel features from Assom's pipeline, the 2-D UMAP embedding, or BEATs
encoder outputs — the interface is agnostic as long as pairwise Euclidean
distance is meaningful).

Pipeline:

    1. **seed** — HDBSCAN on ``X`` (or a supplied seed labelling).
    2. **split** — for each cluster whose per-cluster silhouette is below
       ``split_silhouette_threshold``, attempt to split it by re-running
       HDBSCAN on its members.
    3. **merge** — greedy pairwise merge: merge the closest pair of clusters
       (centroid distance) while their joint silhouette either improves or
       stays within ``merge_silhouette_tolerance``.
    4. **add** — collect segments whose distance to the nearest centroid
       exceeds ``add_outlier_quantile`` of the global distribution. Run
       HDBSCAN on this outlier set; accept new clusters that meet size +
       silhouette gates.
    5. **prune** — drop clusters smaller than ``prune_min_size`` or with
       silhouette below ``prune_silhouette_threshold``; reassign their
       members to the nearest surviving cluster via KNN.

Each pass logs an :class:`Operation` in the state history. The caller can run
``fit_step`` repeatedly until convergence, or call ``fit`` for an all-in-one
driver with basic convergence control.

Design constraints:

- No context labels are ever used — tokenisation stays unsupervised.
- All thresholds are arguments of the constructor; no magic numbers.
- Random seeds are propagated to HDBSCAN fallbacks so runs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import hdbscan
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from tqdm.auto import tqdm

from .types import Operation, Token, TokenizerState


# ────────────────────────────────────────────────────────────────────────────
# Helpers — metric computation that handles degenerate cases gracefully.
# ────────────────────────────────────────────────────────────────────────────

def _locked_atomic_ids(state: TokenizerState) -> set[int]:
    """Atomic ids that are children of at least one composite token.

    These MUST NOT be deleted or renamed by acoustic ops (split/merge/prune),
    otherwise the composite's bookkeeping breaks. Composite tokens themselves
    are also skipped by acoustic ops (they live on sequences, not on labels).
    """
    locked: set[int] = set()
    for tok in state.tokens.values():
        if tok.is_composite:
            locked.update(tok.children)
    return locked


def _safe_silhouette(X: np.ndarray, labels: np.ndarray,
                      sample_size: Optional[int] = None,
                      random_state: int = 0) -> float:
    """Silhouette over non-noise points; returns NaN if undefined.

    ``sample_size`` — if set and smaller than the number of non-noise points,
    sklearn will subsample (stratification is not applied but with random
    sampling large n gives accurate estimate). Makes a huge difference on
    large N — silhouette is O(n²) in time and memory.
    """
    mask = labels >= 0
    uniq = np.unique(labels[mask])
    if mask.sum() < 2 or len(uniq) < 2:
        return float("nan")
    kwargs: dict = {"random_state": random_state}
    if sample_size is not None and sample_size < int(mask.sum()):
        kwargs["sample_size"] = int(sample_size)
    return float(silhouette_score(X[mask], labels[mask], **kwargs))


def _per_cluster_silhouette(X: np.ndarray, labels: np.ndarray,
                             sample_size: Optional[int] = None,
                             random_state: int = 0) -> dict[int, float]:
    """Mean silhouette per cluster (excluding noise). NaN if a cluster has <2 points.

    ``sample_size`` — if set, compute ``silhouette_samples`` on a random
    subsample of non-noise points, then aggregate per-cluster means. A
    subsample of 5000 is usually > 10× faster than full computation on
    50k-point datasets, while preserving per-cluster ranking in practice.
    """
    mask = labels >= 0
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return {int(c): float("nan") for c in np.unique(labels) if c >= 0}
    X_good = X[mask]
    y_good = labels[mask]
    if sample_size is not None and sample_size < len(X_good):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_good), size=int(sample_size), replace=False)
        X_good = X_good[idx]
        y_good = y_good[idx]
    if len(np.unique(y_good)) < 2:
        return {int(c): float("nan") for c in np.unique(labels) if c >= 0}
    sample_sil = silhouette_samples(X_good, y_good)
    out: dict[int, float] = {}
    for c in np.unique(y_good):
        sel = y_good == c
        out[int(c)] = float(sample_sil[sel].mean()) if sel.any() else float("nan")
    # Also fill any cluster not present in the subsample with NaN so the caller
    # can still find them
    for c in np.unique(labels):
        if c >= 0 and int(c) not in out:
            out[int(c)] = float("nan")
    return out


def _centroid(X: np.ndarray, member_ids: np.ndarray) -> np.ndarray:
    return X[member_ids].mean(axis=0) if len(member_ids) else np.zeros(X.shape[1])


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class AcousticTokenizerConfig:
    # Seed HDBSCAN — matches Assom's reference when left at defaults.
    seed_min_cluster_frac: float = 0.01
    seed_min_samples: int = 20
    seed_epsilon: float = 0.1
    seed_method: str = "leaf"

    # Split
    enable_split: bool = True
    split_silhouette_threshold: float = 0.35   # clusters below this are candidates
    split_min_size_factor: float = 0.5         # require subclusters of ≥ half of mcs
    split_max_per_pass: int = 5                # cap per iteration to keep stable

    # Merge
    enable_merge: bool = True
    merge_distance_quantile: float = 0.1       # consider pairs with centroid dist ≤ q-th %ile
    merge_silhouette_tolerance: float = 0.02   # accept merges that drop silh by at most this

    # Add
    enable_add: bool = True
    add_outlier_quantile: float = 0.95         # segments beyond this quantile are outliers
    add_min_size: int = 20
    add_min_silhouette: float = 0.3

    # Prune
    enable_prune: bool = True
    prune_min_size: int = 20
    prune_silhouette_threshold: float = 0.0

    # Run control
    max_iterations: int = 5
    convergence_patience: int = 1              # stop after N iterations without any op
    random_state: int = 0
    show_progress: bool = True                 # show tqdm bars for long steps
    # Silhouette subsampling: silhouette is O(n²) and dominates wall-time on
    # large datasets. ``sample_size=5000`` typically gives the same per-
    # cluster ranking as full computation at 10-100× speedup. Set to None
    # for exact computation.
    silhouette_sample_size: Optional[int] = 5000


# ────────────────────────────────────────────────────────────────────────────
# Main class
# ────────────────────────────────────────────────────────────────────────────

class AcousticTokenizer:
    """Adaptive tokenizer on acoustic embeddings.

    Usage:

        tok = AcousticTokenizer(AcousticTokenizerConfig())
        state = tok.fit(X)                                # full driver
        state = tok.fit(X, seed_labels=hdbscan_labels)    # skip own seed
        # inspect
        state.summary()
        [op.kind for op in state.history]

    The fitted ``TokenizerState`` can be passed to :class:`BPEMerger` or to
    any evaluation routine.
    """

    def __init__(self, config: Optional[AcousticTokenizerConfig] = None):
        self.cfg = config or AcousticTokenizerConfig()

    # ── public driver ────────────────────────────────────────────────────────

    def fit(self,
            X: np.ndarray,
            seed_labels: Optional[np.ndarray] = None,
            sequences_per_file: Optional[list[list[int]]] = None,
            ) -> TokenizerState:
        """Run the full adaptive pipeline until convergence or max iterations.

        Parameters
        ----------
        X : (n, d) segment embeddings.
        seed_labels : optional pre-computed cluster labels (same length as X,
            -1 for noise). If None, runs HDBSCAN internally.
        sequences_per_file : optional list of per-file segment-index sequences,
            so the tokenizer can keep ``state.sequences`` up-to-date after
            every operation. If None, sequences are inferred as one-segment-
            per-sequence.
        """
        state = self._seed(X, seed_labels)
        state.sequences = self._initial_sequences(state.labels, sequences_per_file)
        state.validate()

        no_op_streak = 0
        pbar = tqdm(range(self.cfg.max_iterations),
                     desc='A1 acoustic iter',
                     disable=not self.cfg.show_progress,
                     leave=False)
        for i in pbar:
            state.iteration = i + 1
            any_op = self.fit_step(X, state)
            sil = _safe_silhouette(X, state.labels, sample_size=self.cfg.silhouette_sample_size, random_state=self.cfg.random_state)
            pbar.set_postfix(vocab=state.vocab_size,
                              silh=f'{sil:.3f}' if not np.isnan(sil) else 'nan',
                              any_op=any_op)
            if not any_op:
                no_op_streak += 1
                if no_op_streak >= self.cfg.convergence_patience:
                    break
            else:
                no_op_streak = 0
        pbar.close()
        state.validate()
        return state

    def fit_step(self, X: np.ndarray, state: TokenizerState) -> bool:
        """Apply one pass of split/merge/add/prune. Returns True iff any
        operation actually changed the vocabulary."""
        touched = False
        if self.cfg.enable_split:
            touched |= self._split_pass(X, state)
        if self.cfg.enable_merge:
            touched |= self._merge_pass(X, state)
        if self.cfg.enable_add:
            touched |= self._add_pass(X, state)
        if self.cfg.enable_prune:
            touched |= self._prune_pass(X, state)
        # Re-derive sequences after any change (cheap; keeps invariant)
        state.sequences = self._rederive_sequences(state)
        return touched

    # ── seeding ──────────────────────────────────────────────────────────────

    def _seed(self, X: np.ndarray, seed_labels: Optional[np.ndarray]) -> TokenizerState:
        if seed_labels is None:
            n = X.shape[0]
            mcs = max(int(n * self.cfg.seed_min_cluster_frac), 10)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=self.cfg.seed_min_samples,
                cluster_selection_epsilon=self.cfg.seed_epsilon,
                cluster_selection_method=self.cfg.seed_method,
            )
            labels = clusterer.fit_predict(X)
        else:
            labels = np.asarray(seed_labels, dtype=int).copy()

        tokens: dict[int, Token] = {}
        for c in np.unique(labels):
            if c < 0:
                continue
            member_ids = np.where(labels == c)[0]
            tokens[int(c)] = Token(id=int(c),
                                   centroid=_centroid(X, member_ids),
                                   member_ids=member_ids)

        state = TokenizerState(tokens=tokens, labels=labels, sequences=[])
        state.log(Operation(
            kind="seed", iteration=0,
            affected_ids=tuple(sorted(tokens)),
            metrics_after={"vocab_size": float(len(tokens)),
                           "silhouette": _safe_silhouette(X, labels, sample_size=self.cfg.silhouette_sample_size, random_state=self.cfg.random_state),
                           "noise_frac": float((labels == -1).mean())},
        ))
        return state

    # ── split ────────────────────────────────────────────────────────────────

    def _split_pass(self, X: np.ndarray, state: TokenizerState) -> bool:
        locked = _locked_atomic_ids(state)
        sil_per = _per_cluster_silhouette(X, state.labels, sample_size=self.cfg.silhouette_sample_size, random_state=self.cfg.random_state)
        candidates = [cid for cid, s in sil_per.items()
                      if not np.isnan(s) and s < self.cfg.split_silhouette_threshold
                      and cid not in locked]
        candidates.sort(key=lambda c: sil_per[c])   # worst first
        candidates = candidates[: self.cfg.split_max_per_pass]
        did_any = False
        pbar = tqdm(candidates, desc=f'A1 iter {state.iteration} split',
                     disable=not self.cfg.show_progress or not candidates,
                     leave=False)
        for cid in pbar:
            pbar.set_postfix(candidate=cid, silh=f'{sil_per[cid]:.3f}')
            if self._split_one(X, state, cid):
                did_any = True
        pbar.close()
        return did_any

    def _split_one(self, X: np.ndarray, state: TokenizerState, cid: int) -> bool:
        tok = state.tokens[cid]
        if tok.size < 2 * self.cfg.seed_min_samples:
            return False
        sub_mcs = max(
            int(tok.size * self.cfg.seed_min_cluster_frac * self.cfg.split_min_size_factor),
            self.cfg.seed_min_samples,
        )
        sub_mcs = max(sub_mcs, 2)
        if sub_mcs * 2 > tok.size:
            return False
        try:
            sub = hdbscan.HDBSCAN(
                min_cluster_size=sub_mcs,
                min_samples=max(self.cfg.seed_min_samples // 2, 2),
                cluster_selection_epsilon=self.cfg.seed_epsilon,
                cluster_selection_method=self.cfg.seed_method,
            ).fit_predict(X[tok.member_ids])
        except Exception:
            return False

        n_sub = len(set(sub)) - (1 if -1 in sub else 0)
        if n_sub < 2:
            return False

        # Check the split actually improves structure
        labels_probe = state.labels.copy()
        new_ids: list[int] = []
        sub_clean = sub.copy()
        # noise in the sub-cluster keeps the original parent id
        mapping: dict[int, int] = {}
        next_id = state.next_token_id()
        for s in sorted(set(sub_clean)):
            if s < 0:
                continue
            mapping[int(s)] = next_id
            new_ids.append(next_id)
            next_id += 1
        for idx_local, s in enumerate(sub_clean):
            global_idx = tok.member_ids[idx_local]
            if s < 0:
                labels_probe[global_idx] = cid          # stay with parent
            else:
                labels_probe[global_idx] = mapping[int(s)]

        sil_before = _safe_silhouette(X, state.labels, sample_size=self.cfg.silhouette_sample_size, random_state=self.cfg.random_state)
        sil_after  = _safe_silhouette(X, labels_probe, sample_size=self.cfg.silhouette_sample_size, random_state=self.cfg.random_state)
        if np.isnan(sil_before) or np.isnan(sil_after):
            return False
        if sil_after < sil_before - 0.01:
            return False

        # Commit
        # Drop old, create new tokens
        del state.tokens[cid]
        state.labels = labels_probe
        for new_id in new_ids:
            mids = np.where(state.labels == new_id)[0]
            state.tokens[new_id] = Token(id=new_id,
                                         centroid=_centroid(X, mids),
                                         member_ids=mids)
        # Residual noise-kept piece: some points may still carry `cid`
        residual = np.where(state.labels == cid)[0]
        if len(residual) >= self.cfg.prune_min_size:
            state.tokens[cid] = Token(id=cid,
                                      centroid=_centroid(X, residual),
                                      member_ids=residual)
        else:
            # absorb residual into the nearest new cluster
            self._reassign(X, state, residual)
        state.log(Operation(
            kind="split", iteration=state.iteration,
            affected_ids=(cid, *new_ids),
            metrics_before={"silhouette": sil_before},
            metrics_after={"silhouette": sil_after,
                           "vocab_size": float(state.vocab_size)},
            note=f"split {cid} → {new_ids}",
        ))
        return True

    # ── merge ────────────────────────────────────────────────────────────────

    def _merge_pass(self, X: np.ndarray, state: TokenizerState) -> bool:
        locked = _locked_atomic_ids(state)
        ids = [tid for tid, t in state.tokens.items()
               if not t.is_composite and t.size > 0 and tid not in locked]
        if len(ids) < 2:
            return False
        centroids = np.stack([state.tokens[i].centroid for i in ids])
        dists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
        iu = np.triu_indices(len(ids), k=1)
        pair_dists = dists[iu]
        if len(pair_dists) == 0:
            return False
        thresh = float(np.quantile(pair_dists, self.cfg.merge_distance_quantile))

        did_any = False
        # greedy: pick closest eligible pair repeatedly
        pbar = tqdm(desc=f'A1 iter {state.iteration} merge',
                     disable=not self.cfg.show_progress,
                     leave=False)
        while True:
            locked = _locked_atomic_ids(state)
            ids = [tid for tid, t in state.tokens.items()
                   if not t.is_composite and t.size > 0 and tid not in locked]
            if len(ids) < 2:
                break
            centroids = np.stack([state.tokens[i].centroid for i in ids])
            dists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
            iu = np.triu_indices(len(ids), k=1)
            pair_dists = dists[iu]
            if len(pair_dists) == 0:
                break
            k = int(np.argmin(pair_dists))
            i, j = int(iu[0][k]), int(iu[1][k])
            if pair_dists[k] > thresh:
                break
            id_a, id_b = ids[i], ids[j]
            if not self._merge_two(X, state, id_a, id_b):
                break
            did_any = True
            pbar.update(1)
            pbar.set_postfix(vocab=state.vocab_size)
        pbar.close()
        return did_any

    def _merge_two(self, X: np.ndarray, state: TokenizerState, id_a: int, id_b: int) -> bool:
        tok_a, tok_b = state.tokens[id_a], state.tokens[id_b]
        labels_probe = state.labels.copy()
        labels_probe[labels_probe == id_b] = id_a
        sil_before = _safe_silhouette(X, state.labels, sample_size=self.cfg.silhouette_sample_size, random_state=self.cfg.random_state)
        sil_after  = _safe_silhouette(X, labels_probe, sample_size=self.cfg.silhouette_sample_size, random_state=self.cfg.random_state)
        if np.isnan(sil_before) or np.isnan(sil_after):
            return False
        if sil_after < sil_before - self.cfg.merge_silhouette_tolerance:
            return False
        # commit
        state.labels = labels_probe
        # Remap sequences: any occurrence of id_b -> id_a. This preserves any
        # composites built on top (composite children still reference id_b in
        # their `children` tuple; graceful _flatten handles the missing lookup).
        for seq in state.sequences:
            for i, tid in enumerate(seq):
                if tid == id_b:
                    seq[i] = id_a
        merged_members = np.concatenate([tok_a.member_ids, tok_b.member_ids])
        state.tokens[id_a] = Token(id=id_a,
                                   centroid=_centroid(X, merged_members),
                                   member_ids=merged_members)
        del state.tokens[id_b]
        state.log(Operation(
            kind="merge", iteration=state.iteration,
            affected_ids=(id_a, id_b),
            metrics_before={"silhouette": sil_before},
            metrics_after={"silhouette": sil_after,
                           "vocab_size": float(state.vocab_size)},
            note=f"merged {id_b} into {id_a}",
        ))
        return True

    # ── add ──────────────────────────────────────────────────────────────────

    def _add_pass(self, X: np.ndarray, state: TokenizerState) -> bool:
        atomic_ids = [tid for tid, t in state.tokens.items() if not t.is_composite]
        if not atomic_ids:
            return False
        centroids = np.stack([state.tokens[i].centroid for i in atomic_ids])
        nn = NearestNeighbors(n_neighbors=1).fit(centroids)
        dists, _ = nn.kneighbors(X)
        dists = dists.ravel()
        thresh = float(np.quantile(dists, self.cfg.add_outlier_quantile))
        outlier_idx = np.where(dists > thresh)[0]
        if len(outlier_idx) < self.cfg.add_min_size * 2:
            return False
        # cluster outliers
        try:
            sub_mcs = max(self.cfg.add_min_size, 2)
            sub_labels = hdbscan.HDBSCAN(
                min_cluster_size=sub_mcs,
                min_samples=max(self.cfg.seed_min_samples // 2, 2),
                cluster_selection_epsilon=self.cfg.seed_epsilon,
                cluster_selection_method=self.cfg.seed_method,
            ).fit_predict(X[outlier_idx])
        except Exception:
            return False
        n_new = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
        if n_new == 0:
            return False

        added_ids: list[int] = []
        next_id = state.next_token_id()
        # propose each sub-cluster as a new token
        for s in sorted(set(sub_labels)):
            if s < 0:
                continue
            sel_local = np.where(sub_labels == s)[0]
            sel_global = outlier_idx[sel_local]
            if len(sel_global) < self.cfg.add_min_size:
                continue
            labels_probe = state.labels.copy()
            labels_probe[sel_global] = next_id
            sil_after = _safe_silhouette(X, labels_probe, sample_size=self.cfg.silhouette_sample_size, random_state=self.cfg.random_state)
            if np.isnan(sil_after) or sil_after < self.cfg.add_min_silhouette:
                continue
            # commit
            state.labels = labels_probe
            state.tokens[next_id] = Token(id=next_id,
                                          centroid=_centroid(X, sel_global),
                                          member_ids=sel_global)
            added_ids.append(next_id)
            next_id += 1
        if not added_ids:
            return False
        state.log(Operation(
            kind="add", iteration=state.iteration,
            affected_ids=tuple(added_ids),
            metrics_after={"silhouette": _safe_silhouette(X, state.labels, sample_size=self.cfg.silhouette_sample_size, random_state=self.cfg.random_state),
                           "vocab_size": float(state.vocab_size)},
            note=f"added {len(added_ids)} tokens from outliers",
        ))
        return True

    # ── prune ────────────────────────────────────────────────────────────────

    def _prune_pass(self, X: np.ndarray, state: TokenizerState) -> bool:
        # collect doomed ids — but never drop atomics that are children of
        # any composite (doing so would orphan the composite).
        locked = _locked_atomic_ids(state)
        sil_per = _per_cluster_silhouette(X, state.labels, sample_size=self.cfg.silhouette_sample_size, random_state=self.cfg.random_state)
        to_drop: list[int] = []
        for tid, tok in state.tokens.items():
            if tok.is_composite or tid in locked:
                continue
            s = sil_per.get(tid, float("nan"))
            too_small = tok.size < self.cfg.prune_min_size
            low_sil = not np.isnan(s) and s < self.cfg.prune_silhouette_threshold
            if too_small or low_sil:
                to_drop.append(tid)
        if not to_drop:
            return False
        # gather orphan members
        orphan_idx = np.concatenate(
            [state.tokens[tid].member_ids for tid in to_drop]
        ) if to_drop else np.array([], dtype=int)
        for tid in to_drop:
            del state.tokens[tid]
            state.labels[state.labels == tid] = -1
        self._reassign(X, state, orphan_idx)
        state.log(Operation(
            kind="prune", iteration=state.iteration,
            affected_ids=tuple(to_drop),
            metrics_after={"vocab_size": float(state.vocab_size)},
            note=f"pruned {len(to_drop)} tokens, reassigned {len(orphan_idx)} members",
        ))
        return True

    def _reassign(self, X: np.ndarray, state: TokenizerState, idx: np.ndarray) -> None:
        if len(idx) == 0:
            return
        atomic_ids = [tid for tid, t in state.tokens.items() if not t.is_composite and t.size > 0]
        if not atomic_ids:
            # no surviving token — leave as noise
            state.labels[idx] = -1
            return
        centroids = np.stack([state.tokens[i].centroid for i in atomic_ids])
        nn = NearestNeighbors(n_neighbors=1).fit(centroids)
        _, ix = nn.kneighbors(X[idx])
        for local, global_i in enumerate(idx):
            assigned = atomic_ids[int(ix[local, 0])]
            state.labels[global_i] = assigned
            tok = state.tokens[assigned]
            new_members = np.concatenate([tok.member_ids, [int(global_i)]])
            state.tokens[assigned] = Token(
                id=assigned,
                centroid=_centroid(X, new_members),
                member_ids=new_members,
            )

    # ── sequence bookkeeping ─────────────────────────────────────────────────

    def _initial_sequences(
        self,
        labels: np.ndarray,
        sequences_per_file: Optional[list[list[int]]],
    ) -> list[list[int]]:
        """Turn a list of per-file segment-index sequences into token sequences.

        Empty sequences (all-noise files) are KEPT as ``[]`` so that
        ``len(state.sequences)`` stays aligned with any per-file auxiliary
        arrays (contexts, emitters). Downstream evaluation filters them.
        """
        if sequences_per_file is None:
            return [[int(l)] for l in labels if l >= 0]
        out: list[list[int]] = []
        for seg_ids in sequences_per_file:
            tok_seq = [int(labels[i]) for i in seg_ids if labels[i] >= 0]
            out.append(tok_seq)       # keep even if empty for alignment
        return out

    def _rederive_sequences(self, state: TokenizerState) -> list[list[int]]:
        """Keep sequences as per-segment labels in their original order.

        Empty sequences are kept (see ``_initial_sequences``). BPE / PMI
        mergers operate on the same structure and preserve the per-file
        alignment.
        """
        out: list[list[int]] = []
        for seq in state.sequences:
            new_seq: list[int] = [tid for tid in seq if tid in state.tokens]
            out.append(new_seq)       # keep even if empty for alignment
        return out
