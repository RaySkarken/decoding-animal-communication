"""Cluster homogeneity by tokenization — direct test of advisor's question.

For each tokenization strategy, compute homogeneity / completeness / V-measure
of context labels w.r.t. that strategy's clusters. If a strategy gives more
context-homogeneous clusters → it better captures context structure.

Tokenization strategies (on paper-faithful 153k):
  G       — global HDBSCAN-NCA (V≈11)
  PC-DP   — per-context DP-GMM (V≈109, oracle context for fitting)
  PC-KM   — per-context k-means
  PC-HDB  — per-context HDBSCAN
  AGG-13  — agglomerative (Ward) on UMAP-8D, 13 clusters (matching mean K_95)
  AGG-30  — agglomerative on UMAP-8D, 30 clusters
  AGG-100 — agglomerative on UMAP-8D, 100 clusters (matching per-context vocab)
  KM-30   — global k-means on UMAP-8D, K=30
  KM-100  — global k-means on UMAP-8D, K=100
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib
from pathlib import Path
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN as sk_HDBSCAN
from sklearn.metrics import (homogeneity_score, completeness_score, v_measure_score,
                              mutual_info_score, normalized_mutual_info_score,
                              adjusted_rand_score, adjusted_mutual_info_score,
                              silhouette_score)
from sklearn.neighbors import KNeighborsClassifier
import hdbscan as hdb_lib

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
print('Loading state...')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
hdb_nca = np.load(CACHE / 'hdb_nca_labels_152k_21x32.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
HP1_CTX = [2,3,4,5,6,7,9,10]

# Filter to identified emitters + HP1 contexts (so context is a meaningful target)
mask_eval = (em_arr != 0) & np.isin(ctx, HP1_CTX)
ctx_eval = ctx[mask_eval]
emb_eval = emb[mask_eval]
print(f'Evaluation set: {mask_eval.sum()} segments across {len(set(ctx_eval))} contexts')


def per_context_dpgmm(emb, ctx_arr, K=15):
    """Fit DP-GMM per context (oracle), then assign each point its tokenizer's prediction."""
    labels = np.full(len(emb), -1, dtype=np.int32)
    offset = 0
    for c in HP1_CTX:
        mc = ctx_arr == c
        if mc.sum() < 30: continue
        bgm = BayesianGaussianMixture(n_components=K,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1, covariance_type='full',
            max_iter=200, random_state=0).fit(emb[mc])
        labels[mc] = bgm.predict(emb[mc]) + offset
        offset += bgm.n_components
    return labels


def per_context_kmeans(emb, ctx_arr, K=15):
    labels = np.full(len(emb), -1, dtype=np.int32)
    offset = 0
    for c in HP1_CTX:
        mc = ctx_arr == c
        if mc.sum() < 30: continue
        km = KMeans(n_clusters=min(K, mc.sum()//5), n_init=10, random_state=0).fit(emb[mc])
        labels[mc] = km.labels_ + offset
        offset += K
    return labels


def per_context_hdbscan(emb, ctx_arr):
    labels = np.full(len(emb), -1, dtype=np.int32)
    offset = 0
    for c in HP1_CTX:
        mc = ctx_arr == c
        if mc.sum() < 30: continue
        mcs = max(20, int(0.02 * mc.sum()))
        h = sk_HDBSCAN(min_cluster_size=mcs, min_samples=10,
                        cluster_selection_epsilon=0.05).fit(emb[mc])
        L = h.labels_
        # reassign noise via knn
        nn = L >= 0
        if nn.sum() < 10:
            labels[mc] = offset; offset += 1
            continue
        L_full = L.copy()
        if (~nn).sum() > 0:
            knn = KNeighborsClassifier(n_neighbors=min(10, nn.sum()), n_jobs=-1).fit(emb[mc][nn], L[nn])
            L_full[~nn] = knn.predict(emb[mc][~nn])
        labels[mc] = L_full + offset
        offset += int(L_full.max()) + 1
    return labels


def agglomerative(emb, K, sample_n=30000):
    """Agglomerative Ward — fit on subsample, predict via nearest centroid for full."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(emb), min(sample_n, len(emb)), replace=False)
    ag = AgglomerativeClustering(n_clusters=K, linkage='ward').fit(emb[idx])
    L_sub = ag.labels_
    # build centroids
    cents = np.stack([emb[idx][L_sub == k].mean(0) for k in range(K)])
    # assign all by nearest centroid
    dists = np.linalg.norm(emb[:, None, :] - cents[None, :, :], axis=-1)
    return dists.argmin(axis=1)


def kmeans_global(emb, K):
    return KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(emb)


# === Compute all tokenizations ===
print('\nComputing tokenizations...')
tokens = {}
print('  G (HDBSCAN+NCA, V=11)')
tokens['G_hdbscan_nca'] = hdb_nca
print('  PC-DP (per-context DP-GMM, V≈109)')
tokens['PC_dpgmm'] = per_context_dpgmm(emb, ctx, K=15)
print('  PC-KM (per-context k-means)')
tokens['PC_kmeans'] = per_context_kmeans(emb, ctx, K=15)
print('  PC-HDB (per-context HDBSCAN)')
tokens['PC_hdbscan'] = per_context_hdbscan(emb, ctx)
print('  AGG-13 / AGG-30 / AGG-100 (Ward agglomerative on UMAP-8D)')
for K in [13, 30, 100]:
    tokens[f'AGG_{K}'] = agglomerative(emb, K)
print('  KM-13 / KM-30 / KM-100 (global k-means)')
for K in [13, 30, 100]:
    tokens[f'KM_{K}'] = kmeans_global(emb, K)


# === Compute homogeneity metrics ===
print('\nComputing homogeneity metrics on identified-bat HP1 evaluation subset...')
results = []
for name, lbl in tokens.items():
    L = lbl[mask_eval]
    valid = L >= 0
    if valid.sum() < 100: continue
    L_v = L[valid]; ctx_v = ctx_eval[valid]
    V = len(set(L_v.tolist()))
    H_ck = homogeneity_score(ctx_v, L_v)
    C_ck = completeness_score(ctx_v, L_v)
    Vm = v_measure_score(ctx_v, L_v)
    NMI = normalized_mutual_info_score(ctx_v, L_v)
    AMI = adjusted_mutual_info_score(ctx_v, L_v)
    ARI = adjusted_rand_score(ctx_v, L_v)
    MI = mutual_info_score(ctx_v, L_v) / np.log(2)   # bits
    # Per-cluster purity
    purity_list = []
    for k in set(L_v.tolist()):
        sub = ctx_v[L_v == k]
        if len(sub) < 5: continue
        c = Counter(sub.tolist())
        purity_list.append(c.most_common(1)[0][1] / len(sub))
    purity = np.mean(purity_list) if purity_list else 0
    # Silhouette (subsample)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(emb_eval), min(20000, len(emb_eval)), replace=False)
    L_sample = L[sample_idx]; emb_sample = emb_eval[sample_idx]
    nn = L_sample >= 0
    if nn.sum() > 1000 and len(set(L_sample[nn].tolist())) > 1:
        sil = silhouette_score(emb_sample[nn], L_sample[nn])
    else: sil = np.nan
    results.append({
        'tokenization': name,
        'V': V,
        'homogeneity': H_ck,
        'completeness': C_ck,
        'V_measure': Vm,
        'NMI(ctx,clust)': NMI,
        'AMI': AMI,
        'ARI': ARI,
        'I(ctx;clust) [bits]': MI,
        'mean_cluster_purity': purity,
        'silhouette_8D': sil,
    })

df = pd.DataFrame(results)
df = df.sort_values('homogeneity', ascending=False).reset_index(drop=True)

# Print readable table
print(f'\n{"tokenization":<22s} {"V":>4s} {"homog":>7s} {"compl":>7s} {"V-m":>7s} {"NMI":>7s} {"AMI":>7s} {"ARI":>7s} {"I-bits":>7s} {"purity":>7s} {"sil":>7s}')
print('-'*110)
for _, r in df.iterrows():
    print(f'{r.tokenization:<22s} {int(r.V):>4d} {r.homogeneity:>7.3f} {r.completeness:>7.3f} {r.V_measure:>7.3f} '
          f'{r["NMI(ctx,clust)"]:>7.3f} {r.AMI:>7.3f} {r.ARI:>7.3f} {r["I(ctx;clust) [bits]"]:>7.3f} {r.mean_cluster_purity:>7.3f} {r.silhouette_8D if not np.isnan(r.silhouette_8D) else 0:>7.3f}')

df.to_csv('docs/thesis/figures/cluster_homogeneity_sweep.csv', index=False)
print('\nSaved: docs/thesis/figures/cluster_homogeneity_sweep.csv')
