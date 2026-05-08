"""
Unified evaluation for any tokenizer producing a :class:`TokenizerState`.

Metrics computed (mirrors thesis §3.4):

- **silhouette** — structural quality on the embedding used for seeding
- **noise_fraction** — share of segments not assigned to any atomic cluster
- **ari_context / nmi_context** — agreement between token labels and
  behavioural context (diagnostic, not a paper-reported metric)
- **ari_proxy / nmi_proxy** — agreement with DTW-MFCC qt_ward proxy labels,
  averaged per emitter; this is the paper's main metric
- **types_per_emitter** — the "repertoire size" summary
- **hp1_f1_original / hp1_f1_permuted / hp1_f1_delta** — context-classification
  F1 under original and within-sequence-permuted tokens, plus Δ
- **hp1_feature_bundle** — optional: BoS, Inv, Full breakdown (from our
  HP1-ablation methodology)
- **vocab_size** and **composite_vocab_size** — bookkeeping
- **stability** (multi-seed) — optional: variance of silhouette / ARI across
  N independent runs with different random seeds

Everything returns a plain dict so results can be tabulated directly.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    adjusted_rand_score, f1_score, normalized_mutual_info_score, silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .types import TokenizerState


# ────────────────────────────────────────────────────────────────────────────
# Silhouette / ARI / NMI against fixed references
# ────────────────────────────────────────────────────────────────────────────

def silhouette_of(state: TokenizerState, X: np.ndarray,
                    sample_size: Optional[int] = 5000,
                    random_state: int = 0) -> float:
    """Silhouette on the subset of segments with atomic labels.

    ``sample_size`` — default 5000 subsample for O(n²) speedup; set to None
    for exact. Ranking across methods is preserved to within ~±0.01 for
    subsamples ≥ 3000 on datasets of 50k points.
    """
    labels = state.labels
    mask = labels >= 0
    if mask.sum() < 2:
        return float("nan")
    if len(np.unique(labels[mask])) < 2:
        return float("nan")
    kwargs: dict = {"random_state": random_state}
    if sample_size is not None and sample_size < int(mask.sum()):
        kwargs["sample_size"] = int(sample_size)
    return float(silhouette_score(X[mask], labels[mask], **kwargs))


def noise_fraction_of(state: TokenizerState) -> float:
    return float((state.labels == -1).mean())


def ari_nmi_against(labels_a: np.ndarray, labels_b: np.ndarray,
                     restrict_to_valid: bool = True) -> tuple[float, float]:
    """Compute ARI and NMI between two label arrays of equal length.

    If ``restrict_to_valid`` is True, positions where either array is negative
    are dropped.
    """
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if restrict_to_valid:
        mask = (a >= 0) & (b >= 0)
        a = a[mask]
        b = b[mask]
    if len(a) < 2:
        return float("nan"), float("nan")
    return (float(adjusted_rand_score(a, b)),
            float(normalized_mutual_info_score(a, b)))


def per_emitter_proxy_agreement(
    state: TokenizerState,
    proxy_labels: np.ndarray,
    emitters: np.ndarray,
    show_progress: bool = True,
) -> dict[str, float]:
    """Compute Mean ARI/NMI of tokenizer labels vs qt_ward proxy labels,
    averaged per emitter (the paper's protocol).

    ``proxy_labels`` is expected to be -1 where the proxy wasn't computed
    (e.g. because of per-emitter subsampling).
    """
    ids = np.unique(emitters)
    aris: list[float] = []
    nmis: list[float] = []
    n_types: list[int] = []
    for em in tqdm(ids, desc='per-emitter ARI/NMI',
                    disable=not show_progress, leave=False):
        em_mask = (emitters == em) & (proxy_labels >= 0) & (state.labels >= 0)
        if em_mask.sum() < 5:
            continue
        a, n = ari_nmi_against(state.labels[em_mask], proxy_labels[em_mask],
                                restrict_to_valid=False)
        if not math.isnan(a):
            aris.append(a)
            nmis.append(n)
            n_types.append(len(np.unique(proxy_labels[em_mask])))
    if not aris:
        return dict(ari_proxy=float("nan"), nmi_proxy=float("nan"),
                    ari_proxy_std=float("nan"), nmi_proxy_std=float("nan"),
                    types_per_emitter=float("nan"), types_per_emitter_std=float("nan"),
                    n_emitters_evaluated=0)
    return dict(
        ari_proxy=float(np.mean(aris)),
        nmi_proxy=float(np.mean(nmis)),
        ari_proxy_std=float(np.std(aris)),
        nmi_proxy_std=float(np.std(nmis)),
        types_per_emitter=float(np.mean(n_types)),
        types_per_emitter_std=float(np.std(n_types)),
        n_emitters_evaluated=len(aris),
    )


# ────────────────────────────────────────────────────────────────────────────
# HP1 — context classification + permutation test
# ────────────────────────────────────────────────────────────────────────────

def _bos_features(sequences: Sequence[Sequence[int]], vocab: Sequence[int]) -> np.ndarray:
    idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(sequences), len(vocab)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for s in seq:
            if s in idx:
                X[i, idx[s]] += 1
    return X


def _invariant_features(sequences: Sequence[Sequence[int]], contexts: Sequence[int]) -> np.ndarray:
    """Assom's order-invariant subset {a, b, f, g, i} — for HP1 ablation.

    Rebuilt locally, matches ``notebooks/hp1_feature_ablation.ipynb``.
    """
    # per-context syllable probability
    ctx_probs: dict[int, dict[int, float]] = {}
    for c in set(contexts):
        c_seqs = [s for s, cc in zip(sequences, contexts) if cc == c]
        freq: Counter[int] = Counter()
        for s in c_seqs:
            freq.update(s)
        tot = sum(freq.values())
        ctx_probs[c] = {k: v / tot for k, v in freq.items()} if tot else {}
    rows: list[list[float]] = []
    for seq, ctx in zip(sequences, contexts):
        a = float(len(set(seq)))
        b = float(len(seq))
        i = a / max(b, 1.0)
        counts = list(Counter(seq).values())
        total = sum(counts)
        if total == 0:
            f_ = 0.0
        else:
            ps = np.asarray([c / total for c in counts if c > 0])
            f_ = float(-np.sum(ps * np.log2(ps))) if len(ps) > 0 else 0.0
        pm = ctx_probs[ctx]
        g = math.prod([pm.get(s, 1e-4) for s in seq]) if seq else 0.0
        rows.append([a, b, i, f_, g])
    return np.asarray(rows, dtype=np.float32)


def _rf_cv_f1(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
              random_state: int = 0) -> tuple[float, float]:
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    X = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=random_state, n_jobs=-1)
    scores = cross_val_score(rf, X, y, cv=cv, scoring="f1_weighted", n_jobs=1)
    return float(scores.mean()), float(scores.std())


def hp1_evaluate(
    state: TokenizerState,
    contexts_per_sequence: Sequence[int],
    feature_bundle: str = "bos",
    random_state: int = 0,
    n_splits: int = 5,
    permute: bool = True,
    min_seq_length: int = 2,
) -> dict[str, float]:
    """HP1 F1 on a given tokenizer state.

    ``feature_bundle``:
      - ``"bos"`` — bag-of-syllables counts (the simplest, fastest);
      - ``"inv"`` — Assom order-invariant {a, b, f, g, i};
      - ``"combo"`` — concatenation of bos + inv.

    ``permute``: if True, also computes F1 on within-sequence-shuffled
    sequences and reports ``f1_delta = f1_orig − f1_perm``.

    The function accepts ``state.sequences`` that may contain empty or
    too-short sequences (< ``min_seq_length``). Such sequences are filtered
    along with the matching ``contexts_per_sequence`` entries — as long as
    both arrays are the same length on entry.
    """
    sequences = [list(s) for s in state.sequences]
    contexts = list(contexts_per_sequence)
    if len(sequences) != len(contexts):
        raise ValueError(f"seq/context length mismatch: {len(sequences)} vs {len(contexts)}")

    # Filter empty / degenerate sequences, keeping alignment with contexts.
    keep = [i for i, s in enumerate(sequences) if len(s) >= min_seq_length]
    if len(keep) < len(sequences):
        sequences = [sequences[i] for i in keep]
        contexts = [contexts[i] for i in keep]

    vocab = sorted({t for seq in sequences for t in seq})
    y = np.asarray(contexts)

    def build(seqs: list[list[int]]) -> np.ndarray:
        if feature_bundle == "bos":
            return _bos_features(seqs, vocab)
        if feature_bundle == "inv":
            return _invariant_features(seqs, contexts)
        if feature_bundle == "combo":
            return np.concatenate([
                _bos_features(seqs, vocab),
                _invariant_features(seqs, contexts),
            ], axis=1)
        raise ValueError(f"unknown feature_bundle: {feature_bundle}")

    X_orig = build(sequences)
    f1_orig, s_orig = _rf_cv_f1(X_orig, y, n_splits=n_splits, random_state=random_state)
    out = {
        "hp1_f1_original": f1_orig,
        "hp1_f1_original_std": s_orig,
        "hp1_feature_bundle": feature_bundle,
        "hp1_vocab_size": len(vocab),
    }
    if permute:
        rng = np.random.default_rng(random_state)
        perm_seqs = [list(rng.permutation(s)) for s in sequences]
        X_perm = build(perm_seqs)
        f1_perm, s_perm = _rf_cv_f1(X_perm, y, n_splits=n_splits, random_state=random_state)
        out.update({
            "hp1_f1_permuted": f1_perm,
            "hp1_f1_permuted_std": s_perm,
            "hp1_f1_delta": f1_orig - f1_perm,
        })
    return out


# ────────────────────────────────────────────────────────────────────────────
# Top-level "one call, all metrics"
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    vocab_breakdown: dict[str, int]
    per_emitter_detail: Optional[pd.DataFrame] = None

    def as_row(self, method_name: str) -> pd.Series:
        row = {"method": method_name}
        row.update(self.vocab_breakdown)
        row.update(self.metrics)
        return pd.Series(row)


def full_evaluation(
    state: TokenizerState,
    *,
    embedding: np.ndarray,
    contexts_per_segment: Optional[np.ndarray] = None,
    contexts_per_sequence: Optional[Sequence[int]] = None,
    proxy_labels: Optional[np.ndarray] = None,
    emitters: Optional[np.ndarray] = None,
    run_hp1: bool = True,
    hp1_feature_bundles: Sequence[str] = ("bos", "inv"),
    random_state: int = 0,
    show_progress: bool = True,
    method_label: str = "",
) -> EvaluationResult:
    """Compute the full metric suite.

    Parameters
    ----------
    state : fitted TokenizerState.
    embedding : segment embeddings used for silhouette (same shape as during
        fit).
    contexts_per_segment : per-segment context labels (diagnostic ARI/NMI vs
        context at segment level). If None, that metric is skipped.
    contexts_per_sequence : per-sequence majority context (needed for HP1).
    proxy_labels : per-segment qt_ward proxy labels (-1 where not assigned).
    emitters : per-segment emitter ids, required for per-emitter proxy
        agreement.
    run_hp1 : if True, HP1 F1 for each bundle.
    """
    # rough stage list for the outer progress bar
    stages: list[str] = ["silhouette+noise"]
    if contexts_per_segment is not None:
        stages.append("ARI/NMI ctx")
    if proxy_labels is not None and emitters is not None:
        stages.append("ARI/NMI proxy (per-emitter)")
    if run_hp1 and contexts_per_sequence is not None:
        stages.extend([f"HP1 [{b}] orig+perm" for b in hp1_feature_bundles])

    prefix = f'eval[{method_label}] ' if method_label else 'eval '
    outer = tqdm(total=len(stages), desc=prefix.strip(),
                  disable=not show_progress, leave=False)

    outer.set_postfix_str("silhouette")
    metrics: dict[str, float] = {
        "silhouette": silhouette_of(state, embedding),
        "noise_fraction": noise_fraction_of(state),
    }
    vocab_breakdown = {
        "vocab_size": state.vocab_size,
        "atomic_vocab_size": state.atomic_vocab_size,
        "composite_vocab_size": state.composite_vocab_size,
    }
    outer.update(1)

    if contexts_per_segment is not None and len(contexts_per_segment) == len(state.labels):
        outer.set_postfix_str("ARI/NMI vs context")
        ari_ctx, nmi_ctx = ari_nmi_against(state.labels, np.asarray(contexts_per_segment))
        metrics["ari_context"] = ari_ctx
        metrics["nmi_context"] = nmi_ctx
        outer.update(1)

    if proxy_labels is not None and emitters is not None:
        outer.set_postfix_str("per-emitter proxy")
        metrics.update(per_emitter_proxy_agreement(state, np.asarray(proxy_labels),
                                                     np.asarray(emitters),
                                                     show_progress=show_progress))
        outer.update(1)

    if run_hp1 and contexts_per_sequence is not None:
        for bundle in hp1_feature_bundles:
            outer.set_postfix_str(f"HP1 {bundle}")
            res = hp1_evaluate(state, contexts_per_sequence,
                                feature_bundle=bundle, random_state=random_state)
            for k, v in res.items():
                metrics[f"{k}_{bundle}" if not k.startswith("hp1_") else k + f"_{bundle}"] = v
            outer.update(1)

    outer.close()
    return EvaluationResult(metrics=metrics, vocab_breakdown=vocab_breakdown)


# ────────────────────────────────────────────────────────────────────────────
# Stability across random seeds (multi-run variance)
# ────────────────────────────────────────────────────────────────────────────

def stability(
    fit_fn: Callable[[int], TokenizerState],
    seeds: Iterable[int],
    *,
    embedding: np.ndarray,
    **eval_kwargs,
) -> dict[str, float]:
    """Run ``fit_fn(seed)`` for each seed, collect metrics, report
    mean ± std of silhouette, ARI, NMI, vocab size.
    """
    rows = []
    for s in seeds:
        state = fit_fn(s)
        res = full_evaluation(state, embedding=embedding, **eval_kwargs)
        flat = {**res.metrics, **res.vocab_breakdown}
        flat["seed"] = s
        rows.append(flat)
    df = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for col in df.columns:
        if col == "seed":
            continue
        try:
            summary[f"{col}_mean"] = float(df[col].mean())
            summary[f"{col}_std"]  = float(df[col].std())
        except Exception:
            pass
    summary["n_seeds"] = len(df)
    return summary
