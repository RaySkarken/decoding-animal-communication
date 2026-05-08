"""
Fair comparison: Assom baseline (mel) vs BEATs variants on the SAME
subset of segments (top-5 bats, ~19k).

Previous BEATs experiment only evaluated BEATs methods; need mel-baseline
numbers on identical data to say anything honest.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

CHECKPOINT_DIR = Path('/Volumes/T7/cache/assom_paper_repro')
st_full = joblib.load(CHECKPOINT_DIR / 'ablation_state.joblib')
st_beats = joblib.load(CHECKPOINT_DIR / 'beats_subset_experiment.joblib')

# Subset of seg_df_all that matches BEATs subset by (file_name, pos_segment order).
# We can't exactly match — BEATs re-ran dynamic segmentation from scratch. But
# we can approximate: take segments from top-5 emitters in ablation_state
# and compute metrics on THAT.

seg_df = st_full['seg_df']
hdbnca = st_full['hdb_nca_labels']
mel_flat = st_full['tf_specs'].reshape(-1, 192).astype(np.float32)
embedding = st_full['embedding']

# Match BEATs' top-5 bats
top_emitters = st_beats['seg_beats_meta']['emitter'].unique().tolist()
print(f'Top-5 emitters: {sorted(top_emitters)}')

mask = seg_df['emitter'].isin(top_emitters)
print(f'ablation_state segments in subset: {mask.sum()} (BEATs subset: {len(st_beats["seg_beats_meta"])})')

labels_sub = hdbnca[mask]
mel_sub = mel_flat[mask]
embedding_sub = embedding[mask]
ctx_sub = seg_df.loc[mask, 'context'].to_numpy()


def ctx_metrics(labels, contexts):
    Hs = []; pure_seg = 0; total = 0
    for tid in sorted(set(labels)):
        if tid < 0: continue
        members = np.where(labels == tid)[0]
        if len(members) == 0: continue
        cnt = Counter(contexts[members])
        probs = np.array(list(cnt.values())) / sum(cnt.values()); probs = probs[probs > 0]
        Hs.append(-np.sum(probs * np.log2(probs)))
        if max(cnt.values()) / len(members) >= 0.5:
            pure_seg += len(members)
        total += len(members)
    return dict(mean_H=float(np.mean(Hs)), context_purity=pure_seg/total if total else 0)


rows = []

# Mel baseline on subset
sil_u = silhouette_score(embedding_sub[labels_sub >= 0][:8000],
                           labels_sub[labels_sub >= 0][:8000], random_state=0)
sil_m = silhouette_score(mel_sub[labels_sub >= 0][:8000],
                           labels_sub[labels_sub >= 0][:8000], random_state=0)
cm = ctx_metrics(labels_sub, ctx_sub)
rows.append({
    'method': 'Mel baseline (HDBSCAN+NCA)',
    'vocab': len(set(labels_sub)) - (1 if -1 in labels_sub else 0),
    'silh_native': round(sil_m, 3),
    'silh_UMAP': round(sil_u, 3),
    'ctx_H': round(cm['mean_H'], 2),
    'ctx_purity': round(cm['context_purity'], 3),
})

# BEATs methods — already have numbers
meta_beats = st_beats['seg_beats_meta']
ctx_b = meta_beats['context'].to_numpy()
from sklearn.decomposition import PCA
# Re-derive sil_native as PCA50-BEATs (already done earlier but recompute)
from sklearn.metrics import silhouette_score as silhouette
import numpy as np
X_beats = st_beats['X_beats']
umap_b = st_beats['umap_beats']
X_pca = PCA(n_components=50, random_state=0).fit_transform(X_beats)

for name, lbls in [('BEATs HDBSCAN+NCA', st_beats['hdb_labels_nca']),
                    ('BEATs DP-GMM UMAP', st_beats['dp_labels_umap']),
                    ('BEATs DP-GMM PCA50', st_beats['dp_labels_hd'])]:
    mask = lbls >= 0
    n = min(8000, mask.sum())
    sil_u = silhouette(umap_b[mask][:n], lbls[mask][:n], random_state=0) if n > 100 else np.nan
    sil_n = silhouette(X_pca[mask][:n], lbls[mask][:n], random_state=0) if n > 100 else np.nan
    cm = ctx_metrics(lbls, ctx_b)
    rows.append({
        'method': name,
        'vocab': len(set(lbls)) - (1 if -1 in lbls else 0),
        'silh_native': round(sil_n, 3),
        'silh_UMAP': round(sil_u, 3),
        'ctx_H': round(cm['mean_H'], 2),
        'ctx_purity': round(cm['context_purity'], 3),
    })

df = pd.DataFrame(rows)
print('\n=== FAIR COMPARISON on top-5 bats subset ===')
print(df.to_string(index=False))
df.to_csv('docs/thesis/figures/beats_vs_mel_fair.csv', index=False)
