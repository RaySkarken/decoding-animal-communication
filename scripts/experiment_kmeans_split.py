"""
Final rescue attempt: replace HDBSCAN-based split with forced k-means split.

Hypothesis: HDBSCAN fails to split catch-all clusters because they're
continuous (no density peaks). k-means FORCES a k-way partition
regardless of density structure. If the split into k=2 or k=3 sub-clusters
produces:
  - higher silhouette OR
  - higher context purity OR
  - higher HP1 F1
then it's a meaningful split.

This is our last engineering attempt. If it too fails, we have a clean
negative result: "density-based AND centroid-based adaptive operations
both fail to improve over static HDBSCAN on graded bat vocalisations".
"""
from __future__ import annotations

import sys, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples

from src.adaptive_tokenizer import (
    TokenizerState, Token, full_evaluation,
)

CHECKPOINT_DIR = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CHECKPOINT_DIR / 'ablation_state.joblib')
mel = st['tf_specs'].reshape(-1, 192).astype(np.float32)
hdbnca = st['hdb_nca_labels']
seg_df = st['seg_df']
embedding = st['embedding']
RANDOM_STATE = st['RANDOM_STATE']

sequences_per_file, context_per_sequence = [], []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if len(seg_ids) < 2: continue
    sequences_per_file.append(seg_ids)
    context_per_sequence.append(int(np.bincount(g['context'].to_numpy()).argmax()))
emitters_per_segment = seg_df['emitter'].to_numpy()
contexts_per_segment = seg_df['context'].to_numpy()
proxy_labels = seg_df['proxy_label'].to_numpy() if 'proxy_label' in seg_df.columns else None


def state_from_labels(labels, X, sequences_per_file):
    tokens = {}
    for c in sorted(set(labels)):
        if c < 0: continue
        mids = np.where(labels == c)[0]
        tokens[int(c)] = Token(id=int(c), centroid=X[mids].mean(axis=0), member_ids=mids)
    sequences = [[int(labels[i]) for i in seg_ids if labels[i] >= 0]
                  for seg_ids in sequences_per_file]
    return TokenizerState(tokens=tokens, labels=np.asarray(labels, dtype=int), sequences=sequences)


def ctx_metrics(labels, contexts):
    Hs = []; pure_seg = 0; total = 0
    for tid in sorted(set(labels)):
        if tid < 0: continue
        members = np.where(labels == tid)[0]
        if len(members) == 0: continue
        ctx_cnt = Counter(contexts[members])
        probs = np.array(list(ctx_cnt.values())); probs = probs / probs.sum(); probs = probs[probs > 0]
        Hs.append(-np.sum(probs * np.log2(probs)))
        if max(ctx_cnt.values()) / len(members) >= 0.5:
            pure_seg += len(members)
        total += len(members)
    return dict(mean_H=float(np.mean(Hs)), context_purity=pure_seg/total if total else 0)


def eval_quick(labels, label_name, fit_space='mel'):
    state = state_from_labels(labels, mel if fit_space == 'mel' else embedding, sequences_per_file)
    mask = state.labels >= 0
    n = min(8000, int(mask.sum()))
    sil_u = float(silhouette_score(embedding[mask][:n], state.labels[mask][:n], random_state=0))
    sil_m = float(silhouette_score(mel[mask][:n], state.labels[mask][:n], random_state=0))
    native = mel if fit_space == 'mel' else embedding
    res = full_evaluation(
        state, embedding=native,
        contexts_per_segment=contexts_per_segment,
        contexts_per_sequence=context_per_sequence,
        proxy_labels=proxy_labels, emitters=emitters_per_segment,
        run_hp1=True, hp1_feature_bundles=('bos', 'inv'),
        random_state=RANDOM_STATE, show_progress=False,
    )
    cm = ctx_metrics(state.labels, contexts_per_segment)
    return {
        'method': label_name,
        'vocab': len(set(state.labels)) - (1 if -1 in state.labels else 0),
        'silh_Mel': round(sil_m, 3),
        'silh_UMAP': round(sil_u, 3),
        'ari_proxy': round(res.metrics.get('ari_proxy', np.nan), 3),
        'nmi_proxy': round(res.metrics.get('nmi_proxy', np.nan), 3),
        'ctx_H': round(cm['mean_H'], 2),
        'ctx_purity': round(cm['context_purity'], 3),
        'F1_bos': round(res.metrics.get('hp1_f1_original_bos', np.nan), 3),
        'F1_inv': round(res.metrics.get('hp1_f1_original_inv', np.nan), 3),
    }


rows = [eval_quick(hdbnca, 'assom (baseline)')]

# --- Strategy 1: k-means split on cluster 5 (catch-all) only ---
def force_split(labels, target_cluster, k=2, random_state=0, method='kmeans'):
    labels = labels.copy()
    members = np.where(labels == target_cluster)[0]
    X_sub = mel[members]
    if method == 'kmeans':
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        sub_labs = km.fit_predict(X_sub)
    elif method == 'gmm':
        gm = GaussianMixture(n_components=k, random_state=random_state)
        sub_labs = gm.fit_predict(X_sub)
    next_id = max(labels) + 1
    # First sub-cluster keeps the original id, rest get new ids
    for i, s in enumerate(sub_labs):
        if s > 0:
            labels[members[i]] = next_id + s - 1
    return labels

for k in [2, 3, 4, 5]:
    lbl_k = force_split(hdbnca, target_cluster=5, k=k, method='kmeans')
    rows.append(eval_quick(lbl_k, f'kmeans-split-C5-k{k}'))
    print(f'kmeans split C5 k={k} done')

# --- Strategy 2: split on ALL shared clusters (0, 1, 5) with k=2 each ---
lbl = hdbnca.copy()
for c in [0, 1, 5]:
    lbl = force_split(lbl, target_cluster=c, k=2, method='kmeans')
rows.append(eval_quick(lbl, 'kmeans-split-C015'))

# --- Strategy 3: GMM split on cluster 5 k=3 ---
lbl_gmm = force_split(hdbnca, target_cluster=5, k=3, method='gmm')
rows.append(eval_quick(lbl_gmm, 'gmm-split-C5-k3'))

df = pd.DataFrame(rows)
print('\n=== FORCED-SPLIT EXPERIMENTS ===')
print(df.to_string(index=False))
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/kmeans_split_sweep.csv', index=False)
print('\nSaved: docs/thesis/figures/kmeans_split_sweep.csv')
