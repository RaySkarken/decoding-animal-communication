"""Evaluation metrics: silhouette, ARI, NMI, F1, MR stats, network, stability."""

from __future__ import annotations

import collections
from itertools import combinations
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def cluster_quality(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Silhouette, noise fraction, cluster count."""
    unique = set(labels)
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = int((labels == -1).sum())
    noise_frac = n_noise / len(labels) if len(labels) > 0 else 0.0

    sil = float("nan")
    if n_clusters >= 2:
        non_noise = labels >= 0
        if non_noise.sum() > n_clusters:
            sil = float(silhouette_score(embeddings[non_noise], labels[non_noise]))

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_frac": noise_frac,
        "silhouette": sil,
    }


def context_alignment(
    context_labels: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict:
    """ARI/NMI between **behavioral context** and clusters (diagnostic only).

    Paper RQ1 compares HDBSCAN (or similar) to the **acoustic proxy**
    (DTW–MFCC + Ward + cophenetic quantile), not to context — use
    :func:`repertoire_agreement_acoustic_proxy` for that.
    """
    return {
        "ari": float(adjusted_rand_score(context_labels, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(context_labels, cluster_labels)),
    }


def repertoire_agreement_acoustic_proxy(
    cluster_labels: np.ndarray,
    proxy_labels: np.ndarray,
    seg_df: pd.DataFrame,
    *,
    proxy_mask: np.ndarray | None = None,
    exclude_hdbscan_noise: bool = True,
) -> dict:
    """RQ1-style agreement: cluster labels vs Assom acoustic proxy, **per emitter**.

    For each emitter, ARI and NMI are computed on rows where the proxy is defined
    (and optionally where ``cluster_labels >= 0``). Returns means/stds over emitters
    with at least two distinct proxy and two distinct cluster labels.

    ``proxy_labels`` / ``cluster_labels`` must align row-for-row with ``seg_df``.
    """
    if len(cluster_labels) != len(seg_df) or len(proxy_labels) != len(seg_df):
        raise ValueError("cluster_labels and proxy_labels must match seg_df length")

    if proxy_mask is None:
        valid_base = proxy_labels >= 0
    else:
        valid_base = np.asarray(proxy_mask, dtype=bool)

    aris: List[float] = []
    nmis: List[float] = []
    for em in np.unique(seg_df["emitter"].values):
        m = (seg_df["emitter"].values == em) & valid_base
        if exclude_hdbscan_noise:
            m &= cluster_labels >= 0
        if m.sum() < 2:
            continue
        y_p = proxy_labels[m]
        y_c = cluster_labels[m]
        if len(np.unique(y_p)) < 2 or len(np.unique(y_c)) < 2:
            continue
        aris.append(float(adjusted_rand_score(y_p, y_c)))
        nmis.append(float(normalized_mutual_info_score(y_p, y_c)))

    if not aris:
        nan = float("nan")
        return {
            "ari_mean": nan,
            "ari_std": nan,
            "nmi_mean": nan,
            "nmi_std": nan,
            "n_emitters": 0,
        }

    return {
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
        "nmi_mean": float(np.mean(nmis)),
        "nmi_std": float(np.std(nmis)),
        "n_emitters": len(aris),
    }


def vocabulary_diagnostics(
    labels: np.ndarray,
    seg_df: pd.DataFrame,
) -> dict:
    """Cross-emitter token reuse, per-type frequency stats."""
    types = sorted(set(labels))
    n_emitters = seg_df["emitter"].nunique()

    emitter_per_type: Dict[int, set] = {t: set() for t in types}
    for lbl, emitter in zip(labels, seg_df["emitter"]):
        emitter_per_type[lbl].add(emitter)

    reuse_fracs = [len(v) / n_emitters for v in emitter_per_type.values()]

    type_counts = collections.Counter(labels)
    freqs = np.array([type_counts[t] for t in types])

    return {
        "vocab_size": len(types),
        "mean_cross_emitter_reuse": float(np.mean(reuse_fracs)),
        "min_cross_emitter_reuse": float(np.min(reuse_fracs)) if reuse_fracs else 0.0,
        "type_freq_mean": float(freqs.mean()),
        "type_freq_std": float(freqs.std()),
        "type_freq_min": int(freqs.min()),
        "type_freq_max": int(freqs.max()),
    }


def stability_across_seeds(
    embeddings: np.ndarray,
    cluster_fn,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict:
    """Run clustering with different random seeds, measure ARI pairwise."""
    all_labels = []
    for seed in seeds:
        labels = cluster_fn(embeddings, random_state=seed)
        all_labels.append(labels)

    ari_pairs = []
    for i, j in combinations(range(len(all_labels)), 2):
        ari_pairs.append(adjusted_rand_score(all_labels[i], all_labels[j]))

    return {
        "mean_pairwise_ari": float(np.mean(ari_pairs)),
        "std_pairwise_ari": float(np.std(ari_pairs)),
        "n_seeds": len(seeds),
    }


def maximal_repeats_stats(sequences: List[List[int]]) -> dict:
    """Extract maximal repeats and compute summary statistics."""
    all_mr_lens: List[int] = []
    for seq in sequences:
        mr_lens = _extract_maximal_repeats(seq)
        all_mr_lens.extend(mr_lens)

    arr = np.array(all_mr_lens) if all_mr_lens else np.array([0])
    return {
        "total_mr": len(all_mr_lens),
        "mean_mr_len": float(arr.mean()),
        "max_mr_len": int(arr.max()),
        "median_mr_len": float(np.median(arr)),
        "mr_lengths": arr,
    }


def _extract_maximal_repeats(seq: List[int]) -> List[int]:
    """Extract maximal repeat lengths from a single sequence."""
    if len(seq) < 2:
        return []
    tokens = [str(s) for s in seq]
    str_seq = " ".join(tokens)
    n = len(tokens)
    mrs: List[int] = []

    for length in range(2, n + 1):
        for start in range(n - length + 1):
            subseq = " ".join(tokens[start : start + length])
            count = str_seq.count(subseq)
            if count < 2:
                continue
            longer = False
            if start > 0:
                ext_left = " ".join(tokens[start - 1 : start + length])
                if str_seq.count(ext_left) >= 2:
                    longer = True
            if start + length < n and not longer:
                ext_right = " ".join(tokens[start : start + length + 1])
                if str_seq.count(ext_right) >= 2:
                    longer = True
            if not longer:
                mrs.append(length)
    return mrs if mrs else [1]


def network_metrics(sequences: List[List[int]], min_edge_weight: int = 2) -> dict:
    """Build transition network and compute graph-level metrics."""
    G = nx.DiGraph()
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    edges_to_remove = [
        (u, v) for u, v, d in G.edges(data=True) if d["weight"] < min_edge_weight
    ]
    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))

    if len(G) < 3:
        return {"nodes": len(G), "edges": G.number_of_edges(),
                "density": 0, "avg_clustering": 0, "sigma": None, "omega": None}

    density = nx.density(G)
    Gu = G.to_undirected()
    avg_clustering = nx.average_clustering(Gu)

    sigma, omega = None, None
    try:
        sigma = nx.sigma(Gu, niter=10, seed=42)
    except Exception:
        pass
    try:
        omega = nx.omega(Gu, niter=5, seed=42)
    except Exception:
        pass

    return {
        "nodes": len(G),
        "edges": G.number_of_edges(),
        "density": density,
        "avg_clustering": avg_clustering,
        "sigma": sigma,
        "omega": omega,
    }


def sequence_compressibility(sequences: List[List[int]], order: int = 3) -> float:
    """Cross-entropy of a simple n-gram LM on the token sequences (bits/token)."""
    ngram_counts: Dict[tuple, int] = collections.defaultdict(int)
    context_counts: Dict[tuple, int] = collections.defaultdict(int)

    for seq in sequences:
        for i in range(len(seq)):
            start = max(0, i - order + 1)
            ngram = tuple(seq[start : i + 1])
            ngram_counts[ngram] += 1
            if len(ngram) > 1:
                context_counts[ngram[:-1]] += 1

    total_log_prob = 0.0
    n_tokens = 0
    for seq in sequences:
        for i in range(len(seq)):
            start = max(0, i - order + 1)
            ngram = tuple(seq[start : i + 1])
            ctx = ngram[:-1]
            if ctx and context_counts[ctx] > 0:
                prob = ngram_counts[ngram] / context_counts[ctx]
            else:
                prob = ngram_counts[(seq[i],)] / sum(
                    v for k, v in ngram_counts.items() if len(k) == 1
                )
            total_log_prob += np.log2(prob + 1e-12)
            n_tokens += 1

    return -total_log_prob / n_tokens if n_tokens > 0 else float("inf")


def hp2_wilcoxon(seq_df: pd.DataFrame, all_types: List[int]) -> pd.DataFrame:
    """HP2: Wilcoxon rank-sum tests for context-dependent syllable usage."""
    contexts = sorted(seq_df["context_name"].unique())
    freq_by_ctx: Dict[str, np.ndarray] = {}

    for ctx in contexts:
        ctx_seqs = seq_df[seq_df["context_name"] == ctx]["seq"]
        counts = np.zeros(len(all_types))
        for seq in ctx_seqs:
            for tok in seq:
                idx = all_types.index(tok) if tok in all_types else -1
                if idx >= 0:
                    counts[idx] += 1
        total = counts.sum()
        freq_by_ctx[ctx] = counts / total if total > 0 else counts

    pval_matrix = pd.DataFrame(1.0, index=contexts, columns=contexts)
    for c1, c2 in combinations(contexts, 2):
        _, pval = ranksums(freq_by_ctx[c1], freq_by_ctx[c2])
        pval_matrix.loc[c1, c2] = pval
        pval_matrix.loc[c2, c1] = pval

    return pval_matrix


def full_evaluation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seg_df: pd.DataFrame,
    seq_df: pd.DataFrame,
    sequences: List[List[int]],
    context_enc: np.ndarray | None = None,
) -> dict:
    """Run all evaluation metrics and return a summary dict."""
    results: dict = {}

    results["cluster"] = cluster_quality(embeddings, labels)

    if context_enc is not None:
        results["context_alignment"] = context_alignment(context_enc, labels)

    results["vocab"] = vocabulary_diagnostics(labels, seg_df)
    results["compressibility"] = sequence_compressibility(sequences)

    return results
