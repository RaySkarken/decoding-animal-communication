"""Per-file sequence builder, (a)-(q) feature extractor, and RF classifier."""

from __future__ import annotations

import collections
from typing import List, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score


def build_sequences(
    seg_df: pd.DataFrame,
    min_seq_len: int = 2,
    min_ctx_samples: int = 30,
) -> pd.DataFrame:
    """Group segments by file_name -> ordered syllable sequences."""
    seq_data = []
    for fname, group in seg_df.groupby("file_name"):
        syl_seq = group["syllable_id"].tolist()
        ctx = collections.Counter(group["context"].tolist()).most_common(1)[0][0]
        ctx_name = collections.Counter(group["context_name"].tolist()).most_common(1)[0][0]
        emitter = collections.Counter(group["emitter"].tolist()).most_common(1)[0][0]
        seq_data.append({
            "file_name": fname,
            "seq": syl_seq,
            "seq_len": len(syl_seq),
            "context": ctx,
            "context_name": ctx_name,
            "emitter": emitter,
        })

    seq_df = pd.DataFrame(seq_data)
    seq_df = seq_df[seq_df["seq_len"] >= min_seq_len].reset_index(drop=True)

    ctx_counts = seq_df["context_name"].value_counts()
    keep = ctx_counts[ctx_counts >= min_ctx_samples].index.tolist()
    seq_df = seq_df[seq_df["context_name"].isin(keep)].reset_index(drop=True)
    return seq_df


def compute_sequence_features(seq: List[int], all_types: List[int]) -> dict:
    """Compute the 18 features (a-r) from the paper's Table 1."""
    n = len(seq)
    types_present = set(seq)
    V = len(all_types)
    v = len(types_present)

    transitions = list(zip(seq[:-1], seq[1:]))
    trans_counts = collections.Counter(transitions)
    unique_transitions = len(trans_counts)

    type_counts = collections.Counter(seq)
    freqs = np.array([type_counts.get(t, 0) for t in all_types], dtype=float)
    probs = freqs / freqs.sum() if freqs.sum() > 0 else freqs

    p = probs[probs > 0]
    entropy = float(-np.sum(p * np.log2(p))) if len(p) > 1 else 0.0

    linearity = unique_transitions / len(transitions) if transitions else 0.0
    versatility = v / V if V > 0 else 0.0

    G = nx.DiGraph()
    G.add_nodes_from(all_types)
    for (a, b), c in trans_counts.items():
        G.add_edge(a, b, weight=c)

    out_probs: list[float] = []
    for node in G.nodes():
        total_out = sum(d["weight"] for _, _, d in G.out_edges(node, data=True))
        if total_out > 0:
            for _, _tgt, d in G.out_edges(node, data=True):
                out_probs.append(d["weight"] / total_out)

    mean_tp = float(np.mean(out_probs)) if out_probs else 0.0
    std_tp = float(np.std(out_probs)) if out_probs else 0.0
    max_tp = float(np.max(out_probs)) if out_probs else 0.0
    min_tp = float(np.min(out_probs)) if out_probs else 0.0

    self_loops = sum(1 for a, b in transitions if a == b)
    self_loop_prob = self_loops / len(transitions) if transitions else 0.0

    unique_trigrams = 0
    if n >= 3:
        trigrams = [(seq[i], seq[i + 1], seq[i + 2]) for i in range(n - 2)]
        unique_trigrams = len(set(trigrams))

    return {
        "a_seq_length": n,
        "b_richness": v,
        "c_versatility": versatility,
        "d_entropy": entropy,
        "e_linearity": linearity,
        "f_n_transitions": unique_transitions,
        "g_mean_trans_prob": mean_tp,
        "h_std_trans_prob": std_tp,
        "i_max_trans_prob": max_tp,
        "j_min_trans_prob": min_tp,
        "k_self_loop_prob": self_loop_prob,
        "l_unique_trigrams": unique_trigrams,
        "m_max_type_freq": type_counts.most_common(1)[0][1] / n if n else 0,
        "n_min_type_freq": min(type_counts.values()) / n if n else 0,
        "o_std_type_freq": float(np.std(list(type_counts.values()))) / n if n else 0,
        "p_mean_type_freq": float(np.mean(list(type_counts.values()))) / n if n else 0,
        "q_graph_density": nx.density(G),
        "r_n_types_total": V,
    }


def build_feature_matrix(
    seq_df: pd.DataFrame,
    all_types: List[int] | None = None,
) -> pd.DataFrame:
    """Compute feature matrix for all sequences."""
    if all_types is None:
        all_types = sorted(
            set(t for seq in seq_df["seq"] for t in seq)
        )
    rows = []
    for _, row in seq_df.iterrows():
        rows.append(compute_sequence_features(row["seq"], all_types))
    return pd.DataFrame(rows)


def classify_context(
    seq_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    n_folds: int = 5,
    classifier: str = "svc",
    random_state: int = 42,
) -> dict:
    """Run HP1 context classification + permutation test.

    Returns dict with f1_orig, acc_orig, f1_perm, acc_perm.
    """
    X = feat_df.fillna(0).values
    y = LabelEncoder().fit_transform(seq_df["context_name"].values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_folds = min(n_folds, pd.Series(y).value_counts().min())
    if n_folds < 2:
        raise ValueError("Not enough samples per class for CV (need >= 2).")

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    if classifier == "svc":
        clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=random_state)
    else:
        clf = RandomForestClassifier(
            n_estimators=100, criterion="entropy",
            random_state=random_state, n_jobs=-1,
        )

    y_pred_orig = cross_val_predict(clf, X_scaled, y, cv=cv)
    f1_orig = f1_score(y, y_pred_orig, average="weighted")
    acc_orig = accuracy_score(y, y_pred_orig)

    all_types = sorted(set(t for seq in seq_df["seq"] for t in seq))
    rng = np.random.default_rng(random_state)
    perm_rows = []
    for _, row in seq_df.iterrows():
        perm_seq = list(row["seq"])
        rng.shuffle(perm_seq)
        perm_rows.append(compute_sequence_features(perm_seq, all_types))

    X_perm = pd.DataFrame(perm_rows).fillna(0).values
    X_perm_scaled = scaler.fit_transform(X_perm)
    y_pred_perm = cross_val_predict(clf, X_perm_scaled, y, cv=cv)
    f1_perm = f1_score(y, y_pred_perm, average="weighted")
    acc_perm = accuracy_score(y, y_pred_perm)

    return {
        "f1_orig": f1_orig,
        "acc_orig": acc_orig,
        "f1_perm": f1_perm,
        "acc_perm": acc_perm,
        "y_true": y,
        "y_pred_orig": y_pred_orig,
        "y_pred_perm": y_pred_perm,
    }
