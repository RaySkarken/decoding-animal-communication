"""Adaptive tokenizer with dynamic vocabulary for bat vocalizations.

Core contribution of the thesis. Implements:
  - Seed vocabulary from HDBSCAN on full-dimensional embeddings
  - Split / Merge / Add / Prune vocabulary operations
  - Iterative refinement loop with convergence tracking
  - Sequence-aware BPE merge pass
"""

from __future__ import annotations

import collections
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VocalType:
    """A single entry in the adaptive vocabulary."""
    type_id: int
    centroid: np.ndarray
    member_indices: List[int]
    intra_variance: float
    creation_reason: str
    children: Optional[Tuple[int, int]] = None  # for BPE composites: (left_id, right_id)

    @property
    def size(self) -> int:
        return len(self.member_indices)


@dataclass
class VocabSnapshot:
    """Lightweight snapshot for tracking history."""
    iteration: int
    n_types: int
    n_splits: int
    n_merges: int
    n_adds: int
    n_prunes: int
    type_ids: List[int]
    type_sizes: List[int]


# ---------------------------------------------------------------------------
# AdaptiveTokenizer
# ---------------------------------------------------------------------------

class AdaptiveTokenizer:
    """Adaptive tokenizer with dynamic vocabulary.

    Parameters
    ----------
    embeddings : ndarray of shape (n_segments, dim)
        Segment embeddings (full-dimensional, NOT 2-D UMAP).
    min_cluster_size : int
        HDBSCAN ``min_cluster_size`` for seed clustering.
    min_samples : int
        HDBSCAN ``min_samples``.
    pca_dim : int or None
        If set, reduce embeddings to this dimensionality via PCA before
        HDBSCAN (recommended for high-D spaces like 672-D mel).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 50,
        min_samples: int = 10,
        pca_dim: Optional[int] = None,
        random_state: int = 42,
    ):
        self.raw_embeddings = embeddings.astype(np.float32)
        self.random_state = random_state
        self._next_id = 0

        if pca_dim is not None and pca_dim < embeddings.shape[1]:
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=pca_dim, random_state=random_state)
            self.embeddings = self._pca.fit_transform(self.raw_embeddings)
        else:
            self._pca = None
            self.embeddings = self.raw_embeddings

        self.labels = np.full(len(embeddings), -1, dtype=int)
        self.vocab: Dict[int, VocalType] = {}
        self._history: List[VocabSnapshot] = []
        self._sequences: Optional[List[List[int]]] = None
        self._bpe_merges: List[Tuple[int, int, int]] = []

        self._seed(min_cluster_size, min_samples)

    # ------------------------------------------------------------------
    # Seed vocabulary
    # ------------------------------------------------------------------

    def _seed(self, min_cluster_size: int, min_samples: int) -> None:
        """Run HDBSCAN on embeddings to create the initial vocabulary."""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method="leaf",
            prediction_data=True,
        )
        clusterer.fit(self.embeddings)
        raw_labels = clusterer.labels_

        for cid in sorted(set(raw_labels)):
            if cid == -1:
                continue
            members = np.where(raw_labels == cid)[0].tolist()
            tid = self._make_type(members, reason="seed")
            self.labels[members] = tid

        logger.info(
            "Seed: %d types, %d noise points",
            len(self.vocab),
            int((self.labels == -1).sum()),
        )
        self._history.append(self._snapshot(0, 0, 0, 0, 0))

    def _make_type(self, members: List[int], reason: str) -> int:
        """Create a new VocalType and register it."""
        tid = self._next_id
        self._next_id += 1
        emb = self.embeddings[members]
        centroid = emb.mean(axis=0)
        var = float(np.mean(np.sum((emb - centroid) ** 2, axis=1)))
        self.vocab[tid] = VocalType(
            type_id=tid,
            centroid=centroid,
            member_indices=list(members),
            intra_variance=var,
            creation_reason=reason,
        )
        return tid

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def _try_splits(
        self,
        sil_threshold: float = 0.0,
        min_size_for_split: int = 30,
    ) -> int:
        """Split clusters whose per-sample silhouette is below threshold.

        Returns the number of splits performed.
        """
        n_splits = 0
        types_to_split: List[int] = []

        if len(self.vocab) < 2:
            return 0

        active_mask = self.labels >= 0
        if active_mask.sum() < 4:
            return 0

        sil_per_sample = silhouette_samples(
            self.embeddings[active_mask], self.labels[active_mask],
        )
        idx_active = np.where(active_mask)[0]

        mean_sil_by_type: Dict[int, float] = {}
        for global_idx, s in zip(idx_active, sil_per_sample):
            tid = self.labels[global_idx]
            mean_sil_by_type.setdefault(tid, [])
            mean_sil_by_type[tid].append(s)
        mean_sil_by_type = {k: float(np.mean(v)) for k, v in mean_sil_by_type.items()}

        for tid, avg_sil in mean_sil_by_type.items():
            vt = self.vocab[tid]
            if avg_sil < sil_threshold and vt.size >= min_size_for_split:
                types_to_split.append(tid)

        for tid in types_to_split:
            vt = self.vocab[tid]
            emb_sub = self.embeddings[vt.member_indices]
            km = KMeans(n_clusters=2, random_state=self.random_state, n_init=5)
            sub_labels = km.fit_predict(emb_sub)

            members_a = [vt.member_indices[i] for i in range(len(sub_labels)) if sub_labels[i] == 0]
            members_b = [vt.member_indices[i] for i in range(len(sub_labels)) if sub_labels[i] == 1]

            if len(members_a) < 5 or len(members_b) < 5:
                continue

            del self.vocab[tid]

            tid_a = self._make_type(members_a, reason=f"split_from_{tid}")
            tid_b = self._make_type(members_b, reason=f"split_from_{tid}")
            self.labels[members_a] = tid_a
            self.labels[members_b] = tid_b
            n_splits += 1

        return n_splits

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _try_merges(
        self,
        distance_threshold: float | None = None,
        pmi_threshold: float = 0.0,
    ) -> int:
        """Merge pairs of types that are close in embedding space
        and (optionally) have high bigram PMI.

        If *distance_threshold* is None, use the 20th-percentile of
        all inter-centroid distances as an adaptive threshold.
        """
        n_merges = 0
        tids = sorted(self.vocab.keys())
        if len(tids) < 2:
            return 0

        centroids = np.array([self.vocab[t].centroid for t in tids])
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(centroids))

        if distance_threshold is None:
            avg_radius = float(np.mean([
                np.sqrt(vt.intra_variance) for vt in self.vocab.values()
                if vt.intra_variance > 0
            ])) if self.vocab else 1.0
            distance_threshold = avg_radius * 1.5

        pmi_matrix = self._bigram_pmi() if self._sequences is not None else {}

        merged_set: set = set()
        pairs = []
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                if dists[i, j] <= distance_threshold:
                    pmi_val = pmi_matrix.get((tids[i], tids[j]), 0.0)
                    pairs.append((dists[i, j], pmi_val, tids[i], tids[j]))

        pairs.sort(key=lambda x: x[0])

        for dist_val, pmi_val, tid_a, tid_b in pairs:
            if tid_a in merged_set or tid_b in merged_set:
                continue
            if self._sequences is not None and pmi_val < pmi_threshold:
                continue

            vt_a = self.vocab[tid_a]
            vt_b = self.vocab[tid_b]
            combined = vt_a.member_indices + vt_b.member_indices

            del self.vocab[tid_a]
            del self.vocab[tid_b]

            new_tid = self._make_type(combined, reason=f"merge_{tid_a}_{tid_b}")
            self.labels[combined] = new_tid

            merged_set.update({tid_a, tid_b})
            n_merges += 1

        return n_merges

    # ------------------------------------------------------------------
    # Add (rescue noise)
    # ------------------------------------------------------------------

    def _try_adds(
        self,
        noise_min_cluster_size: int = 15,
    ) -> int:
        """Try to form new types from noise points via mini-HDBSCAN."""
        noise_idx = np.where(self.labels == -1)[0]
        if len(noise_idx) < noise_min_cluster_size * 2:
            return 0

        emb_noise = self.embeddings[noise_idx]
        mini = hdbscan.HDBSCAN(
            min_cluster_size=noise_min_cluster_size,
            min_samples=max(3, noise_min_cluster_size // 3),
            cluster_selection_method="leaf",
        )
        mini.fit(emb_noise)

        n_adds = 0
        for cid in sorted(set(mini.labels_)):
            if cid == -1:
                continue
            local_members = np.where(mini.labels_ == cid)[0]
            global_members = noise_idx[local_members].tolist()
            if len(global_members) < 5:
                continue
            tid = self._make_type(global_members, reason="add_from_noise")
            self.labels[global_members] = tid
            n_adds += 1

        return n_adds

    # ------------------------------------------------------------------
    # Prune
    # ------------------------------------------------------------------

    def _try_prunes(self, min_support_frac: float = 0.005) -> int:
        """Remove types with too few members, reassigning them via KNN."""
        n = (self.labels >= 0).sum()
        min_support = max(3, int(n * min_support_frac))
        n_prunes = 0

        to_prune = [
            tid for tid, vt in self.vocab.items()
            if vt.size < min_support and vt.children is None
        ]
        if not to_prune or len(self.vocab) - len(to_prune) < 2:
            return 0

        good_tids = [t for t in self.vocab if t not in set(to_prune)]
        good_members = []
        good_labels = []
        for tid in good_tids:
            for m in self.vocab[tid].member_indices:
                good_members.append(m)
                good_labels.append(tid)

        knn = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)
        knn.fit(self.embeddings[good_members], good_labels)

        for tid in to_prune:
            vt = self.vocab[tid]
            if not vt.member_indices:
                del self.vocab[tid]
                n_prunes += 1
                continue
            preds = knn.predict(self.embeddings[vt.member_indices])
            for member_idx, new_tid in zip(vt.member_indices, preds):
                self.labels[member_idx] = new_tid
                self.vocab[new_tid].member_indices.append(member_idx)
            del self.vocab[tid]
            n_prunes += 1

        self._recompute_stats()
        return n_prunes

    # ------------------------------------------------------------------
    # Refinement loop
    # ------------------------------------------------------------------

    def refine(
        self,
        max_iter: int = 10,
        convergence_threshold: float = 0.01,
        sil_threshold: float = 0.0,
        distance_threshold: float | None = None,
        pmi_threshold: float = 0.0,
        min_support_frac: float = 0.005,
        noise_min_cluster_size: int = 15,
        min_size_for_split: int = 30,
    ) -> List[VocabSnapshot]:
        """Iteratively refine the vocabulary.

        Returns the history of snapshots (also stored in self._history).
        """
        for it in range(1, max_iter + 1):
            n_splits = self._try_splits(sil_threshold, min_size_for_split)
            n_merges = self._try_merges(distance_threshold, pmi_threshold)
            n_adds = self._try_adds(noise_min_cluster_size)
            n_prunes = self._try_prunes(min_support_frac)

            total_changes = n_splits + n_merges + n_adds + n_prunes
            snap = self._snapshot(it, n_splits, n_merges, n_adds, n_prunes)
            self._history.append(snap)

            logger.info(
                "Iter %d: splits=%d merges=%d adds=%d prunes=%d  |V|=%d",
                it, n_splits, n_merges, n_adds, n_prunes, len(self.vocab),
            )

            if len(self.vocab) == 0:
                break
            if total_changes / max(len(self.vocab), 1) < convergence_threshold:
                logger.info("Converged at iteration %d", it)
                break

        return self._history

    # ------------------------------------------------------------------
    # BPE sequence-aware merge
    # ------------------------------------------------------------------

    def set_sequences(self, sequences: List[List[int]]) -> None:
        """Provide syllable sequences for sequence-aware operations."""
        self._sequences = sequences

    def bpe_merge(
        self,
        max_merges: int = 20,
        min_bigram_count: int = 10,
    ) -> List[Tuple[int, int, int]]:
        """BPE-like merging of frequent bigrams into composite tokens.

        Each merge creates a new composite VocalType with ``children=(left, right)``.
        All sequences in ``self._sequences`` are updated in-place.

        Returns list of (left_id, right_id, new_id) merge records.
        """
        if self._sequences is None:
            raise ValueError("Call set_sequences() before bpe_merge().")

        merges: List[Tuple[int, int, int]] = []

        for _ in range(max_merges):
            bigram_counts = collections.Counter()
            for seq in self._sequences:
                for a, b in zip(seq[:-1], seq[1:]):
                    bigram_counts[(a, b)] += 1

            if not bigram_counts:
                break

            (best_a, best_b), best_count = bigram_counts.most_common(1)[0]
            if best_count < min_bigram_count:
                break

            new_tid = self._next_id
            self._next_id += 1
            self.vocab[new_tid] = VocalType(
                type_id=new_tid,
                centroid=np.zeros_like(next(iter(self.vocab.values())).centroid),
                member_indices=[],
                intra_variance=0.0,
                creation_reason=f"bpe_{best_a}+{best_b}",
                children=(best_a, best_b),
            )

            for seq in self._sequences:
                i = 0
                new_seq: list[int] = []
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == best_a and seq[i + 1] == best_b:
                        new_seq.append(new_tid)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                seq.clear()
                seq.extend(new_seq)

            merges.append((best_a, best_b, new_tid))
            self._bpe_merges.append((best_a, best_b, new_tid))
            logger.info(
                "BPE merge: (%d, %d) -> %d  (count=%d)", best_a, best_b, new_tid, best_count,
            )

        return merges

    def encode_sequence(self, seq: List[int]) -> List[int]:
        """Apply stored BPE merges to a new sequence."""
        result = list(seq)
        for left, right, new_id in self._bpe_merges:
            i = 0
            merged: list[int] = []
            while i < len(result):
                if i < len(result) - 1 and result[i] == left and result[i + 1] == right:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(result[i])
                    i += 1
            result = merged
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bigram_pmi(self) -> Dict[Tuple[int, int], float]:
        """Compute pointwise mutual information for all observed bigrams."""
        if self._sequences is None:
            return {}

        bigram_counts: Dict[Tuple[int, int], int] = collections.defaultdict(int)
        unigram_counts: Dict[int, int] = collections.defaultdict(int)
        total_bigrams = 0

        for seq in self._sequences:
            for tok in seq:
                unigram_counts[tok] += 1
            for a, b in zip(seq[:-1], seq[1:]):
                bigram_counts[(a, b)] += 1
                total_bigrams += 1

        if total_bigrams == 0:
            return {}

        total_unigrams = sum(unigram_counts.values())
        pmi: Dict[Tuple[int, int], float] = {}
        for (a, b), count in bigram_counts.items():
            p_ab = count / total_bigrams
            p_a = unigram_counts[a] / total_unigrams
            p_b = unigram_counts[b] / total_unigrams
            pmi[(a, b)] = float(np.log2(p_ab / (p_a * p_b + 1e-12) + 1e-12))

        return pmi

    def _recompute_stats(self) -> None:
        """Recompute centroids and variances from current member lists."""
        for tid, vt in list(self.vocab.items()):
            if not vt.member_indices:
                continue
            if vt.children is not None:
                continue
            emb = self.embeddings[vt.member_indices]
            vt.centroid = emb.mean(axis=0)
            vt.intra_variance = float(np.mean(np.sum((emb - vt.centroid) ** 2, axis=1)))

    def _snapshot(self, iteration, n_splits, n_merges, n_adds, n_prunes) -> VocabSnapshot:
        return VocabSnapshot(
            iteration=iteration,
            n_types=len(self.vocab),
            n_splits=n_splits,
            n_merges=n_merges,
            n_adds=n_adds,
            n_prunes=n_prunes,
            type_ids=sorted(self.vocab.keys()),
            type_sizes=[self.vocab[t].size for t in sorted(self.vocab.keys())],
        )

    @property
    def history(self) -> List[VocabSnapshot]:
        return list(self._history)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def noise_count(self) -> int:
        return int((self.labels == -1).sum())

    def get_labels(self) -> np.ndarray:
        """Return a copy of the current label array."""
        return self.labels.copy()

    def summary(self) -> str:
        lines = [
            f"AdaptiveTokenizer: {self.vocab_size} types, "
            f"{self.noise_count} noise, {len(self.embeddings)} segments",
        ]
        if self._bpe_merges:
            lines.append(f"  BPE merges: {len(self._bpe_merges)}")
        for snap in self._history[-3:]:
            lines.append(
                f"  iter {snap.iteration}: |V|={snap.n_types} "
                f"(+{snap.n_splits}s +{snap.n_adds}a -{snap.n_merges}m -{snap.n_prunes}p)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BEATs-inspired iterative refinement (Phase 5 / stretch goal)
# ---------------------------------------------------------------------------

def iterative_tokenize(
    raw_embeddings: np.ndarray,
    n_iterations: int = 3,
    hidden_dim: int = 64,
    epochs_per_iter: int = 20,
    batch_size: int = 256,
    tokenizer_kwargs: dict | None = None,
    refine_kwargs: dict | None = None,
    random_state: int = 42,
) -> Tuple[AdaptiveTokenizer, List[dict]]:
    """BEATs-style self-distillation loop.

    1. Tokenize with AdaptiveTokenizer on *raw_embeddings*.
    2. Train a small MLP to predict token ID from raw embedding.
    3. Use penultimate-layer activations as new embeddings.
    4. Re-tokenize on the new embeddings.
    5. Repeat for *n_iterations*.

    Returns the final tokenizer and a list of per-iteration metric dicts.
    """
    import tensorflow as tf
    from tensorflow import keras

    tk_kw = tokenizer_kwargs or {}
    ref_kw = refine_kwargs or {}
    emb = raw_embeddings.copy()
    iter_metrics: List[dict] = []

    for it in range(n_iterations):
        logger.info("=== Iterative tokenization: round %d/%d ===", it + 1, n_iterations)

        tok = AdaptiveTokenizer(emb, random_state=random_state, **tk_kw)
        tok.refine(**ref_kw)
        labels = tok.get_labels()

        non_noise = labels >= 0
        n_classes = len(set(labels[non_noise]))

        iter_metrics.append({
            "iteration": it + 1,
            "vocab_size": tok.vocab_size,
            "noise_count": tok.noise_count,
        })

        if it == n_iterations - 1:
            return tok, iter_metrics

        X_train = emb[non_noise]
        y_train = labels[non_noise]
        label_map = {old: new for new, old in enumerate(sorted(set(y_train)))}
        y_mapped = np.array([label_map[y] for y in y_train])

        model = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation="gelu",
                               input_shape=(X_train.shape[1],)),
            keras.layers.Dense(hidden_dim, activation="gelu", name="penultimate"),
            keras.layers.Dense(n_classes, activation="softmax"),
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            X_train, y_mapped,
            epochs=epochs_per_iter,
            batch_size=batch_size,
            verbose=0,
        )

        feat_model = keras.Model(
            inputs=model.input,
            outputs=model.get_layer("penultimate").output,
        )
        emb = feat_model.predict(raw_embeddings, batch_size=batch_size, verbose=0)
        logger.info("New embedding shape: %s", emb.shape)

    return tok, iter_metrics
