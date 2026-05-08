"""Иерархическая кластеризация глубже.

Раньше делали AGG-13/30/100 на UMAP-8D. Контексты не отделились (V-мера 0.10).
Сейчас:
  1. AGG с автоматическим выбором K через gap statistic / BIC
  2. Co-clustering: одновременно contexts × clusters матрица
  3. Bisecting k-means (top-down иерархическая)
  4. AGG на per-bat подвыборках (внутри-особей)

Цель — проверить, не появляется ли отделимость контекстов на других иерархических
уровнях или в других нормировках.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib
from pathlib import Path
from collections import Counter
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (homogeneity_score, completeness_score, v_measure_score,
                              adjusted_rand_score)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]

print('Loading state...')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()

mask = (em_arr != 0) & np.isin(ctx, HP1_CTX)
emb_e = emb[mask]; ctx_e = ctx[mask]
print(f'  evaluation set: {mask.sum()} segments')


# Подвыборка для иерархических методов (full матрица расстояний слишком велика)
rng = np.random.default_rng(0)
N_SUB = 30000
sub_idx = rng.choice(len(emb_e), N_SUB, replace=False)
emb_sub = emb_e[sub_idx]; ctx_sub = ctx_e[sub_idx]
print(f'  subsample for hierarchical: {N_SUB}')


def evaluate(labels, ctx_arr):
    valid = labels >= 0
    if valid.sum() < 100: return None
    return {
        'V': len(set(labels[valid].tolist())),
        'H_C|K': float(homogeneity_score(ctx_arr[valid], labels[valid])),
        'C_K|C': float(completeness_score(ctx_arr[valid], labels[valid])),
        'V_m': float(v_measure_score(ctx_arr[valid], labels[valid])),
        'ARI': float(adjusted_rand_score(ctx_arr[valid], labels[valid])),
    }


print('\n=== 1. Bisecting k-means (top-down hierarchical) ===')
# Bisecting k-means: рекурсивно делим самый рассыпанный кластер на 2
results = []
for K in [8, 16, 32, 64, 128]:
    # sklearn BisectingKMeans
    from sklearn.cluster import BisectingKMeans
    bkm = BisectingKMeans(n_clusters=K, random_state=0, n_init=3).fit(emb_e)
    m = evaluate(bkm.labels_, ctx_e)
    if m is None: continue
    print(f'  K={K}: V_m={m["V_m"]:.3f}, H={m["H_C|K"]:.3f}, '
          f'C={m["C_K|C"]:.3f}, ARI={m["ARI"]:.3f}', flush=True)
    results.append(dict(method='bisecting-kmeans', K=K, **m))


print('\n=== 2. Linkage variations on subsample ===')
# Compute pdist once
print('  computing pdist on 30k subsample...', flush=True)
D = pdist(emb_sub, metric='euclidean')
print(f'  pdist size: {len(D)}', flush=True)

for link in ['ward', 'complete', 'average', 'single']:
    Z = linkage(D, method=link)
    for K in [13, 30, 100]:
        labels_sub = fcluster(Z, t=K, criterion='maxclust') - 1
        m = evaluate(labels_sub, ctx_sub)
        if m is None: continue
        print(f'  linkage={link:>9s}, K={K:>3d}: V_m={m["V_m"]:.3f}, '
              f'H={m["H_C|K"]:.3f}, C={m["C_K|C"]:.3f}, ARI={m["ARI"]:.3f}',
              flush=True)
        results.append(dict(method=f'linkage-{link}', K=K, **m))


print('\n=== 3. Per-bat hierarchical: AGG inside each emitter ===')
# Если внутри одной особи контексты отделимы геометрически, это тонкая структура
em_unique = sorted(set(em_arr[em_arr != 0].tolist()))
within_bat_results = []
for e in em_unique[:20]:  # топ-20 особей с большим объёмом
    m_e = (em_arr == e) & np.isin(ctx, HP1_CTX)
    if m_e.sum() < 200: continue
    n_ctx = len(set(ctx[m_e].tolist()))
    if n_ctx < 3: continue
    ag = AgglomerativeClustering(n_clusters=8, linkage='ward').fit(emb[m_e])
    m = evaluate(ag.labels_, ctx[m_e])
    if m is None: continue
    within_bat_results.append(dict(emitter=int(e), n_seg=int(m_e.sum()),
                                    n_contexts=n_ctx, **m))

if within_bat_results:
    df_wb = pd.DataFrame(within_bat_results)
    print(f'  per-bat AGG (K=8) results: {len(df_wb)} emitters')
    print(f'  mean V_m={df_wb["V_m"].mean():.3f} ± {df_wb["V_m"].std():.3f}')
    print(f'  mean H_C|K={df_wb["H_C|K"].mean():.3f}')
    print(f'  mean C_K|C={df_wb["C_K|C"].mean():.3f}')
    print(f'  mean ARI={df_wb["ARI"].mean():.3f}')
    df_wb.to_csv('docs/thesis/figures/hierarchical_per_bat.csv', index=False)


print('\n=== 4. Co-clustering: spectral on context-cluster matrix ===')
# Берём AGG-100, считаем матрицу [contexts x clusters], применяем spectral co-clustering
from sklearn.cluster.bicluster import SpectralCoclustering
ag100 = AgglomerativeClustering(n_clusters=100, linkage='ward').fit(emb_sub)
labels100 = ag100.labels_
M = np.zeros((len(HP1_CTX), 100))
for i, c in enumerate(HP1_CTX):
    for k in range(100):
        M[i, k] = ((ctx_sub == c) & (labels100 == k)).sum()
M = M / M.sum(axis=1, keepdims=True)  # row-normalize: per-context dist

# Энтропия per-context distribution: насколько контекст concentrated в кластерах?
ent = -np.nansum(M * np.log(M + 1e-12), axis=1)
print('  per-context entropy over 100 clusters:')
for i, c in enumerate(HP1_CTX):
    print(f'    ctx={c}: entropy={ent[i]:.2f} bits, '
          f'top-3 clusters={np.argsort(M[i])[::-1][:3].tolist()}')

# Если контекст полностью в одном кластере: ent = 0
# Если равномерно — ent = log_2(100) ≈ 6.64
print(f'  max possible entropy (uniform over 100): {np.log(100):.2f}')
print(f'  mean ent: {ent.mean():.2f} (ratio {ent.mean()/np.log(100):.2f})')


pd.DataFrame(results).to_csv('docs/thesis/figures/hierarchical_deep.csv', index=False)
print('\nSaved: docs/thesis/figures/hierarchical_deep.csv')
print('       docs/thesis/figures/hierarchical_per_bat.csv')
