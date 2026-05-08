"""Anomaly detection per-context: поиск странных сегментов внутри контекста.

Идея:
  1. Для каждого контекста c обучаем per-context DP-GMM на UMAP-8D.
  2. Считаем log-likelihood каждого сегмента под "своей" моделью.
  3. Сегменты с min log-likelihood — кандидаты в аномалии.
  4. Сравниваем: где эти аномалии лежат в глобальной HDBSCAN-разметке?
     Если в типичных кластерах своего контекста — это просто хвосты.
     Если в кластерах, доминирующих в ДРУГИХ контекстах — это интересно
     (контекстная аномалия: акустически чужой паттерн).

Это качественный диагностический эксперимент — не для F1.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib
from pathlib import Path
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from scipy.special import logsumexp

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAMES = {2: 'Mating', 3: 'Feeding', 4: 'Fighting', 5: 'Isolation',
             6: 'Biting', 7: 'Threat', 9: 'Grooming', 10: 'Kissing'}

print('Loading state...')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
hdb = np.load(CACHE / 'hdb_nca_labels_152k_21x32.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()


print('\nFitting per-context DP-GMM and computing per-segment log-likelihoods...')
log_p = np.full(len(emb), -np.inf, dtype=np.float64)
models = {}
for c in HP1_CTX:
    mc = (ctx == c) & (em_arr != 0)
    if mc.sum() < 50: continue
    bgm = BayesianGaussianMixture(n_components=15,
                                    weight_concentration_prior_type='dirichlet_process',
                                    weight_concentration_prior=0.1, covariance_type='full',
                                    max_iter=200, random_state=0).fit(emb[mc])
    models[c] = bgm
    log_p[mc] = bgm.score_samples(emb[mc])
    print(f'  ctx={c} ({CTX_NAMES[c]}): n={mc.sum()}, K_eff={int((bgm.weights_>0.01).sum())}, '
          f'log_p mean={log_p[mc].mean():.2f}, min={log_p[mc].min():.2f}', flush=True)


# --- Поиск аномалий ---
print('\nFinding anomalies (bottom 1% log-likelihood per context)...')
anomaly_results = []
for c in HP1_CTX:
    mc = (ctx == c) & (em_arr != 0)
    if mc.sum() < 100: continue
    lp = log_p[mc]
    thresh = np.percentile(lp, 1)  # bottom 1%
    is_anom = lp <= thresh
    anom_indices = np.where(mc)[0][is_anom]
    n_anom = len(anom_indices)

    # Какие глобальные кластеры HDBSCAN у аномалий vs типичных?
    typ_indices = np.where(mc)[0][~is_anom]
    anom_clusters = Counter(hdb[anom_indices].tolist())
    typ_clusters = Counter(hdb[typ_indices].tolist())
    typ_total = sum(typ_clusters.values())
    # Top-3 кластера для аномалий и для типичных
    anom_top = anom_clusters.most_common(3)
    typ_top = typ_clusters.most_common(3)

    # Доля аномалий, попадающих в кластеры, которые ДОМИНИРУЮТ в ДРУГИХ контекстах
    cluster_top_ctx = {}
    for k in set(hdb[hdb >= 0].tolist()):
        m_k = (hdb == k) & np.isin(ctx, HP1_CTX) & (em_arr != 0)
        if m_k.sum() == 0: continue
        ctx_count = Counter(ctx[m_k].tolist())
        cluster_top_ctx[k] = ctx_count.most_common(1)[0][0]
    n_other_ctx = sum(1 for i in anom_indices
                       if hdb[i] in cluster_top_ctx and cluster_top_ctx[hdb[i]] != c)

    print(f'\n  ctx={c} ({CTX_NAMES[c]}):', flush=True)
    print(f'    anomalies: {n_anom} segments (bottom 1%, threshold log_p <= {thresh:.2f})')
    print(f'    log_p anomalies: mean={lp[is_anom].mean():.2f}, range=[{lp.min():.2f}, {thresh:.2f}]')
    print(f'    typical clusters (HDBSCAN top-3): {typ_top}')
    print(f'    anomalous clusters (HDBSCAN top-3): {anom_top}')
    print(f'    anomalies in clusters dominated by other contexts: '
          f'{n_other_ctx}/{n_anom} ({100*n_other_ctx/n_anom:.0f}%)')
    anomaly_results.append({
        'context': CTX_NAMES[c],
        'n_segments': mc.sum(),
        'n_anomalies': n_anom,
        'log_p_threshold_bot1': thresh,
        'log_p_min': float(lp.min()),
        'log_p_anom_mean': float(lp[is_anom].mean()),
        'anom_in_other_ctx_clusters': n_other_ctx,
        'pct_in_other': 100 * n_other_ctx / max(n_anom, 1),
        'top_typ_cluster': typ_top[0][0],
        'top_anom_cluster': anom_top[0][0],
        'typ_top_pct': typ_top[0][1] / max(typ_total, 1) * 100,
    })


# --- Sequence-level anomaly: вокализации с min average log_p ---
print('\nFinding anomalous vocalizations (sequence-level)...')
file_arr = seg_df['file_name'].to_numpy()
voc_anom = []
for fname in pd.unique(file_arr):
    mask_v = file_arr == fname
    if mask_v.sum() < 2: continue
    c = int(np.bincount(ctx[mask_v]).argmax())
    if c not in HP1_CTX: continue
    if (em_arr[mask_v] == 0).all(): continue
    mean_logp = log_p[mask_v].mean()
    voc_anom.append({'file': fname, 'context': CTX_NAMES[c], 'mean_log_p': mean_logp,
                     'n_segments': int(mask_v.sum())})

vdf = pd.DataFrame(voc_anom).sort_values('mean_log_p')
print(f'\n  Top-10 most anomalous vocalizations (smallest mean log_p):')
for _, r in vdf.head(10).iterrows():
    print(f"    {r['file']}: ctx={r['context']:>10s}, n={r['n_segments']:>3d}, "
          f"mean_log_p={r['mean_log_p']:.2f}")

vdf.to_csv('docs/thesis/figures/anomaly_vocalizations.csv', index=False)
pd.DataFrame(anomaly_results).to_csv('docs/thesis/figures/anomaly_per_context.csv', index=False)
print(f"\nSaved: docs/thesis/figures/anomaly_*.csv")
print(f"  Total anomalies (1% per context): {sum(r['n_anomalies'] for r in anomaly_results)}")
mean_other = np.mean([r['pct_in_other'] for r in anomaly_results])
print(f"  Mean % anomalies in clusters dominated by OTHER contexts: {mean_other:.1f}%")
