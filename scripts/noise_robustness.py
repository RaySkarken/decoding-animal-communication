"""Устойчивость к шумам — рекомендуемая конфигурация vs опорный пайплайн.

Возмущения на UMAP-эмбеддингах:
  - noise: x' = x + N(0, sigma * std(x))  sigma in {0.05, 0.1, 0.2, 0.3, 0.5}
  - drop:  случайно выбрасываем долю сегментов из вокализации
  - trunc: оставляем только первые N сегментов

Метрика: macro F1, 5 разбиений по особям.
Конфигурация: рекомендуемая (per-context DP-GMM full, UMAP-8D, равн. приор).
Сравнение с опорным пайплайном (RF на 18 признаках после HDBSCAN).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib, time
from sklearn.mixture import BayesianGaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
SEEDS = list(range(5))

print('Loading state and embeddings...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy').astype(np.float64)
ctx_arr = seg_df['context'].to_numpy()
emb_std = emb.std(0).mean()
print(f'  emb shape: {emb.shape}, mean std per dim: {emb_std:.3f}', flush=True)

# Build vocs
vocs = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids: continue
    dom_em_signed = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
    if dom_em_signed == 0: continue
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    if dom_ctx not in HP1_CTX: continue
    vocs.append({'fname': fname, 'seg_ids': seg_ids, 'ctx': dom_ctx,
                 'em': abs(dom_em_signed)})

all_bats = sorted(set(v['em'] for v in vocs))
log_prior = {c: -np.log(len(HP1_CTX)) for c in HP1_CTX}
print(f'  vocs: {len(vocs)}, bats: {len(all_bats)}', flush=True)


def fit_pc_dpgmm(train_mask):
    toks = {}
    for c in HP1_CTX:
        m = train_mask & (ctx_arr == c)
        if m.sum() < 30: continue
        toks[c] = BayesianGaussianMixture(
            n_components=15,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1, covariance_type='full',
            max_iter=150, random_state=0
        ).fit(emb[m])
    return toks


def perturb_segments(seg_ids, kind, param, rng):
    """Возвращает (perturbed_emb_subset, kept_ids)."""
    n = len(seg_ids)
    if kind == 'clean':
        return emb[seg_ids], seg_ids
    if kind == 'noise':
        x = emb[seg_ids].copy()
        x += rng.normal(0, param * emb_std, size=x.shape)
        return x, seg_ids
    if kind == 'drop':
        keep = rng.random(n) > param
        if keep.sum() == 0:
            keep[0] = True
        kept_ids = np.array(seg_ids)[keep].tolist()
        return emb[kept_ids], kept_ids
    if kind == 'trunc':
        k = min(int(param), n)
        kept = seg_ids[:k]
        return emb[kept], kept
    raise ValueError(kind)


def score_pc(toks, perturbed_X):
    if len(perturbed_X) == 0:
        return None
    lls = {c: t.score_samples(perturbed_X).sum() + log_prior[c] for c, t in toks.items()}
    return max(lls, key=lls.get)


PERTURBATIONS = [
    ('clean', 0.0),
    ('noise', 0.05),
    ('noise', 0.1),
    ('noise', 0.2),
    ('noise', 0.3),
    ('noise', 0.5),
    ('drop', 0.1),
    ('drop', 0.3),
    ('drop', 0.5),
    ('trunc', 1),
    ('trunc', 3),
    ('trunc', 5),
]

rows = []
for seed in SEEDS:
    t0 = time.time()
    rng_split = np.random.default_rng(seed)
    ba = np.array(all_bats); rng_split.shuffle(ba)
    test_b = set(ba[:11].tolist())
    train_v = [v for v in vocs if v['em'] not in test_b]
    test_v = [v for v in vocs if v['em'] in test_b]
    train_mask = np.zeros(len(emb), dtype=bool)
    for v in train_v: train_mask[v['seg_ids']] = True
    toks = fit_pc_dpgmm(train_mask)
    fit_t = time.time() - t0
    print(f'\nseed {seed}: train={len(train_v)}, test={len(test_v)}, fit={fit_t:.0f}s', flush=True)

    for kind, param in PERTURBATIONS:
        rng_perturb = np.random.default_rng(seed * 1000 + hash((kind, param)) % 997)
        yt, yp = [], []
        for v in test_v:
            X_p, _ = perturb_segments(v['seg_ids'], kind, param, rng_perturb)
            if len(X_p) == 0: continue
            cp = score_pc(toks, X_p)
            if cp is None: continue
            yt.append(v['ctx']); yp.append(cp)
        f1m = f1_score(yt, yp, average='macro', labels=HP1_CTX, zero_division=0)
        f1w = f1_score(yt, yp, average='weighted', labels=HP1_CTX, zero_division=0)
        rows.append({'seed': seed, 'kind': kind, 'param': param,
                     'method': 'pc_dpgmm_reco', 'f1_m': f1m, 'f1_w': f1w,
                     'n_test': len(yt)})
        print(f'  {kind:6s} {param:>5}: f1m={f1m:.3f} f1w={f1w:.3f}', flush=True)

df = pd.DataFrame(rows)
out = Path('docs/thesis/figures/noise_robustness_reco.csv')
df.to_csv(out, index=False)
print(f'\nSaved: {out}', flush=True)

print('\n=== Summary by perturbation (avg over 5 seeds) ===')
agg = df.groupby(['kind', 'param']).agg({'f1_m': ['mean', 'std'], 'f1_w': ['mean', 'std']})
print(agg)
