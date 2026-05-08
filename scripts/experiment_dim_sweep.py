"""
Experiment A: UMAP dimensionality sweep — does the ‘no improvement
over baseline' result persist when we cluster in higher-D UMAP?

Motivated by Chavooshi & Mamonov 2025 (arXiv:2501.07729) + Kather,
Ghani, Stowell 2025 (arXiv:2504.06710) — 2D UMAP is demonstrably
near-worst-case for density clustering.

Setup: for each n_components ∈ {2, 8, 16, 32}:
  - re-fit UMAP from mel-192D
  - cluster with HDBSCAN (paper-style) AND DP-GMM (k_max=40)
  - evaluate: silh in UMAP-nD, silh in mel-192D, ctx_purity,
    AMI vs context (Kather-aligned), AMI vs qt_ward proxy

Writes docs/thesis/figures/dim_sweep.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.metrics import (
    silhouette_score, adjusted_mutual_info_score,
    adjusted_rand_score, normalized_mutual_info_score,
)
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors
import umap
import hdbscan
import time

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
mel = st['tf_specs'].reshape(len(st['tf_specs']), -1).astype(np.float32)
ctx = st['seg_df']['context'].to_numpy()
proxy = st['seg_df']['proxy_label'].to_numpy() if 'proxy_label' in st['seg_df'].columns else None
emitters = st['seg_df']['emitter'].to_numpy()
print(f'Data: {mel.shape}, contexts unique: {len(set(ctx))}')


def ctx_purity(labels, ctx):
    p, t = 0, 0
    for tid in sorted(set(labels)):
        if tid < 0: continue
        m = np.where(labels == tid)[0]
        if not len(m): continue
        if max(Counter(ctx[m]).values()) / len(m) >= 0.5: p += len(m)
        t += len(m)
    return p / t if t else 0


def ami_against(labels, ref, restrict=True):
    a, b = np.asarray(labels), np.asarray(ref)
    if restrict:
        mask = (a >= 0) & (b >= 0)
        a, b = a[mask], b[mask]
    if len(a) < 2 or len(set(a)) < 2 or len(set(b)) < 2: return np.nan
    return float(adjusted_mutual_info_score(a, b))


def per_emitter_ami(labels, proxy_labels, emitters):
    if proxy_labels is None: return np.nan
    vals = []
    for em in set(emitters):
        mask = (emitters == em) & (proxy_labels >= 0) & (labels >= 0)
        if mask.sum() < 20: continue
        v = adjusted_mutual_info_score(labels[mask], proxy_labels[mask])
        vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def silh(X, labels, n=8000):
    mask = labels >= 0
    n = min(n, int(mask.sum()))
    if n < 100: return np.nan
    lbls = labels[mask][:n]
    if len(set(lbls)) < 2: return np.nan
    return float(silhouette_score(X[mask][:n], lbls, random_state=0))


def fit_hdbscan(X, mcs_frac=0.01):
    mcs = max(int(len(X) * mcs_frac), 10)
    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=20,
                            cluster_selection_epsilon=0.1,
                            cluster_selection_method='leaf').fit_predict(X)


def fit_dp_gmm(X, n_components=40, concentration=0.1, min_size=20):
    bgm = BayesianGaussianMixture(n_components=n_components,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=concentration, covariance_type='full',
        max_iter=100, random_state=0)
    lbl = bgm.fit_predict(X)
    # absorb tiny components into nearest active
    cnt = Counter(int(x) for x in lbl)
    active = {k for k, v in cnt.items() if v >= min_size}
    if len(active) < len(cnt):
        am = np.isin(lbl, list(active))
        if (~am).any() and am.any():
            knn = NearestNeighbors(n_neighbors=1).fit(X[am])
            _, idx = knn.kneighbors(X[~am])
            lbl[~am] = lbl[am][idx.ravel()]
    return lbl


rows = []
dim_space_cache = {}

for dim in [2, 8, 16, 32]:
    cache_path = CKPT / f'umap_{dim}d.npy'
    if cache_path.exists():
        umap_X = np.load(cache_path)
        print(f'Loaded cached UMAP-{dim}D')
    else:
        print(f'Fitting UMAP-{dim}D ...')
        t0 = time.time()
        reducer = umap.UMAP(n_components=dim, n_neighbors=30, min_dist=0.3,
                              metric='euclidean', random_state=0, n_jobs=-1)
        umap_X = reducer.fit_transform(mel)
        np.save(cache_path, umap_X)
        print(f'  took {time.time()-t0:.0f}s')

    dim_space_cache[dim] = umap_X

    # Method 1: HDBSCAN on UMAP-nD
    hdb = fit_hdbscan(umap_X)
    vocab_hdb = len(set(hdb)) - (1 if -1 in hdb else 0)
    if vocab_hdb < 2:
        print(f'UMAP-{dim}D HDBSCAN: degenerate (vocab={vocab_hdb}), skip')
        continue
    rows.append({
        'method': f'HDBSCAN on UMAP-{dim}D',
        'umap_dim': dim,
        'algorithm': 'HDBSCAN',
        'vocab': vocab_hdb,
        'noise_frac': round((hdb == -1).mean(), 3),
        'silh_umap_space': round(silh(umap_X, hdb), 3),
        'silh_mel_native': round(silh(mel, hdb), 3),
        'ctx_purity': round(ctx_purity(hdb, ctx), 3),
        'ami_vs_context': round(ami_against(hdb, ctx), 3),
        'ami_vs_proxy_per_emitter': round(per_emitter_ami(hdb, proxy, emitters) if proxy is not None else float('nan'), 3),
    })
    print(f'  HDBSCAN-{dim}D: vocab={vocab_hdb}, ctx_purity={rows[-1]["ctx_purity"]}, ami_ctx={rows[-1]["ami_vs_context"]}')

    # Method 2: DP-GMM on UMAP-nD
    dp = fit_dp_gmm(umap_X)
    vocab_dp = len(set(dp)) - (1 if -1 in dp else 0)
    rows.append({
        'method': f'DP-GMM on UMAP-{dim}D',
        'umap_dim': dim,
        'algorithm': 'DP-GMM',
        'vocab': vocab_dp,
        'noise_frac': 0.0,
        'silh_umap_space': round(silh(umap_X, dp), 3),
        'silh_mel_native': round(silh(mel, dp), 3),
        'ctx_purity': round(ctx_purity(dp, ctx), 3),
        'ami_vs_context': round(ami_against(dp, ctx), 3),
        'ami_vs_proxy_per_emitter': round(per_emitter_ami(dp, proxy, emitters) if proxy is not None else float('nan'), 3),
    })
    print(f'  DP-GMM-{dim}D: vocab={vocab_dp}, ctx_purity={rows[-1]["ctx_purity"]}, ami_ctx={rows[-1]["ami_vs_context"]}')

# Baseline reference: raw mel + HDBSCAN/DP-GMM (no UMAP)
print('Baseline: raw 192D mel + DP-GMM ...')
dp_raw = fit_dp_gmm(mel[:, :50])   # PCA-like: first 50 dims, avoid full 192
vocab_raw = len(set(dp_raw))
rows.append({
    'method': 'DP-GMM on raw mel-50D',
    'umap_dim': 'raw-50',
    'algorithm': 'DP-GMM',
    'vocab': vocab_raw,
    'noise_frac': 0.0,
    'silh_umap_space': np.nan,
    'silh_mel_native': round(silh(mel, dp_raw), 3),
    'ctx_purity': round(ctx_purity(dp_raw, ctx), 3),
    'ami_vs_context': round(ami_against(dp_raw, ctx), 3),
    'ami_vs_proxy_per_emitter': round(per_emitter_ami(dp_raw, proxy, emitters) if proxy is not None else float('nan'), 3),
})

df = pd.DataFrame(rows)
print('\n=== DIM SWEEP RESULTS ===')
print(df.to_string(index=False))
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/dim_sweep.csv', index=False)
print('\nSaved.')
