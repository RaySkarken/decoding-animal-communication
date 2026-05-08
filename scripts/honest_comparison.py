"""
Consolidated, academically honest comparison of ALL methods tested.

Key decisions for fairness:
  - Report silh_UMAP as the primary "silhouette" metric because that's
    what Assom [2025] reports (sil > 0.5 on UMAP 2D embedding)
  - Report silh_native as the metric in the feature space the method
    was fit on — useful context, NOT a substitute for silh_UMAP
  - ctx_purity as proxy for behavioural-context alignment
  - Flag each method's fundamental weakness, not just its strength
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.metrics import silhouette_score
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
tf_specs = st['tf_specs']
mel = tf_specs.reshape(len(tf_specs), -1).astype(np.float32)
emb = st['embedding']           # Mel-UMAP-2D
ctx = st['seg_df']['context'].to_numpy()
hdbnca = st['hdb_nca_labels']

# Load BEATs results (different segment set — can't compare silh_UMAP to mel directly)
st_b = joblib.load(CKPT / 'beats_full_experiment.joblib')
meta_b = st_b['seg_meta']
umap_b = st_b['umap_beats']
X_beats = st_b['X_beats']
hdb_b = st_b['hdb_nca']
dp_b = st_b['dp_umap']
dp_b_pca = st_b['dp_pca']
ctx_b = meta_b['context'].to_numpy()
from sklearn.decomposition import PCA
X_beats_pca = PCA(n_components=50, random_state=0).fit_transform(X_beats)

# Load A1 / BPE / other experiment states
def try_load(name):
    try: return joblib.load(CKPT / name)
    except FileNotFoundError: return None

st_a1 = try_load('adaptive_tokenizer_results.joblib')
st_mel_exp = try_load('mel_space_experiment.joblib')
st_nomerge = try_load('nomerge_experiment.joblib')
st_additive = try_load('additive_experiment.joblib')


def ctx_purity(labels, ctx):
    p, t = 0, 0
    for tid in sorted(set(labels)):
        if tid < 0: continue
        m = np.where(labels == tid)[0]
        if not len(m): continue
        cnt = Counter(ctx[m])
        if max(cnt.values()) / len(m) >= 0.5: p += len(m)
        t += len(m)
    return p / t if t else 0


def silh(X, labels, n=8000):
    mask = labels >= 0
    n = min(n, int(mask.sum()))
    if n < 100: return np.nan
    lbls = labels[mask][:n]
    if len(set(lbls)) < 2: return np.nan
    return float(silhouette_score(X[mask][:n], lbls, random_state=0))


def eval_method(labels, X_native, X_umap, ctx_arr, method, notes=''):
    return {
        'method': method,
        'vocab': len(set(labels)) - (1 if -1 in labels else 0),
        'silh_UMAP (paper-comparable)': round(silh(X_umap, labels), 3),
        'silh_native (fit-space)': round(silh(X_native, labels), 3),
        'ctx_purity': round(ctx_purity(labels, ctx_arr), 3),
        'notes': notes,
    }


rows = []

# ─── Baseline ─────────────────────────────────────────────────────────
rows.append(eval_method(hdbnca, mel, emb, ctx,
    'Assom baseline (HDBSCAN+NCA on UMAP)',
    'paper method'))

# ─── Our AdaptiveTokenizer variants (from adaptive_tokenizer_results.joblib) ──
if st_a1 is not None:
    states = st_a1.get('states', {})
    for name, state in states.items():
        if not hasattr(state, 'labels'): continue
        lbl = np.asarray(state.labels)
        rows.append(eval_method(lbl, mel, emb, ctx,
            f'AdaptiveTokenizer: {name}',
            'from adaptive_tokenizer_results.joblib'))

# ─── A1 variants from mel_space_experiment ─────────────────────────────
if st_mel_exp is not None:
    for name, state in st_mel_exp.get('states', {}).items():
        if not hasattr(state, 'labels'): continue
        lbl = np.asarray(state.labels)
        rows.append(eval_method(lbl, mel, emb, ctx,
            f'MEL-space experiment: {name}',
            'with/without merge'))

# ─── no-merge variants ─────────────────────────────────────────────────
if st_nomerge is not None:
    for name, state in st_nomerge.get('states', {}).items():
        if not hasattr(state, 'labels') or name == 'assom': continue
        lbl = np.asarray(state.labels)
        rows.append(eval_method(lbl, mel, emb, ctx,
            f'A1 no-merge: {name}', 'merge disabled'))

# ─── additive variants ─────────────────────────────────────────────────
if st_additive is not None:
    for name, state in st_additive.get('states', {}).items():
        if not hasattr(state, 'labels') or name == 'assom': continue
        lbl = np.asarray(state.labels)
        rows.append(eval_method(lbl, mel, emb, ctx,
            f'A1 additive: {name}', 'split+add only'))

# ─── DP-GMM on mel (UMAP k=40) — re-fit for determinism ───────────────
from pathlib import Path as _P
_cache = CKPT / 'dp_mel_umap_k40.npy'
if _cache.exists():
    dp_mel_umap = np.load(_cache)
else:
    bgm = BayesianGaussianMixture(n_components=40,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1, covariance_type='full',
        max_iter=100, random_state=0)
    dp_mel_umap = bgm.fit_predict(emb)
    cnt = Counter(int(x) for x in dp_mel_umap)
    active = {k for k, v in cnt.items() if v >= 20}
    if len(active) < len(cnt):
        am = np.isin(dp_mel_umap, list(active))
        if (~am).any():
            knn = NearestNeighbors(n_neighbors=1).fit(emb[am])
            _, idx = knn.kneighbors(emb[~am])
            dp_mel_umap[~am] = dp_mel_umap[am][idx.ravel()]
    np.save(_cache, dp_mel_umap)
rows.append(eval_method(dp_mel_umap, mel, emb, ctx,
    'Mel DP-GMM on UMAP (k_max=40)',
    'non-parametric Bayesian'))

_cache = CKPT / 'dp_mel_native_k20.npy'
if _cache.exists():
    dp_mel_native = np.load(_cache)
    rows.append(eval_method(dp_mel_native, mel, emb, ctx,
        'Mel DP-GMM on native 192D (k_max=20)',
        'fits native feature space'))

# ─── BEATs methods (different segment set — NOTE THIS) ─────────────────
# silh_UMAP for BEATs is silh on BEATs-UMAP, not Mel-UMAP — different space.
# We MUST report this caveat.
def eval_beats(labels, X_native, X_umap, ctx_arr, method):
    return {
        'method': method,
        'vocab': len(set(labels)) - (1 if -1 in labels else 0),
        'silh_UMAP (paper-comparable)': round(silh(X_umap, labels), 3),
        'silh_native (fit-space)': round(silh(X_native, labels), 3),
        'ctx_purity': round(ctx_purity(labels, ctx_arr), 3),
        'notes': 'DIFFERENT segment set + DIFFERENT UMAP space than mel',
    }

rows.append(eval_beats(hdb_b, X_beats_pca, umap_b, ctx_b, 'BEATs HDBSCAN+NCA'))
rows.append(eval_beats(dp_b,  X_beats_pca, umap_b, ctx_b, 'BEATs DP-GMM on BEATs-UMAP'))
rows.append(eval_beats(dp_b_pca, X_beats_pca, umap_b, ctx_b, 'BEATs DP-GMM on BEATs-PCA50'))

df = pd.DataFrame(rows).sort_values('silh_UMAP (paper-comparable)', ascending=False)
print(df.to_string(index=False))
df.to_csv('docs/thesis/figures/honest_consolidated.csv', index=False)
print('\nSaved to docs/thesis/figures/honest_consolidated.csv')
