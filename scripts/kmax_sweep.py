"""Ablation: K_max sweep для per-context DP-GMM.

Цель: ответить честно на «почему именно K_max = 15».
Проверяем: K_active, K_95, macro F1 при K_max ∈ {10, 15, 20, 25, 30}.

Если K_95 насыщается ниже K_max → 15 достаточно.
Если K_95 растёт с K_max → 15 урезает словарь, ablation нужен.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib, time
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import f1_score

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
KMAX_VALUES = [10, 15, 20, 25, 30]
SEEDS = list(range(5))

print('Loading state and embeddings...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
ctx_arr = seg_df['context'].to_numpy()

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
print(f'  vocs: {len(vocs)}, bats: {len(all_bats)}', flush=True)
log_prior = {c: -np.log(len(HP1_CTX)) for c in HP1_CTX}


def fit_and_score(kmax, seed):
    rng = np.random.default_rng(seed)
    ba = np.array(all_bats); rng.shuffle(ba)
    test_b = set(ba[:11].tolist())
    train_v = [v for v in vocs if v['em'] not in test_b]
    test_v = [v for v in vocs if v['em'] in test_b]
    train_mask = np.zeros(len(emb), dtype=bool)
    for v in train_v: train_mask[v['seg_ids']] = True

    toks = {}
    k95_per_ctx = {}
    k_active_per_ctx = {}
    for c in HP1_CTX:
        m = train_mask & (ctx_arr == c)
        if m.sum() < 30: continue
        bgm = BayesianGaussianMixture(
            n_components=kmax,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1, covariance_type='full',
            max_iter=150, random_state=seed
        ).fit(emb[m])
        toks[c] = bgm
        # K_95: число компонент, покрывающих 95% массы (по убыванию веса)
        w_sorted = np.sort(bgm.weights_)[::-1]
        cum = np.cumsum(w_sorted)
        k95 = int(np.searchsorted(cum, 0.95) + 1)
        k_active = int((bgm.weights_ > 1e-3).sum())
        k95_per_ctx[c] = k95
        k_active_per_ctx[c] = k_active

    yt, yp = [], []
    for v in test_v:
        X = emb[v['seg_ids']]
        if len(X) == 0: continue
        lls = {c: t.score_samples(X).sum() + log_prior[c] for c, t in toks.items()}
        if not lls: continue
        cp = max(lls, key=lls.get)
        yt.append(v['ctx']); yp.append(cp)
    f1m = f1_score(yt, yp, average='macro', labels=HP1_CTX, zero_division=0)
    return f1m, k95_per_ctx, k_active_per_ctx


rows = []
for kmax in KMAX_VALUES:
    for seed in SEEDS:
        t0 = time.time()
        f1m, k95_per_ctx, k_active_per_ctx = fit_and_score(kmax, seed)
        elapsed = time.time() - t0
        k95_med = int(np.median(list(k95_per_ctx.values())))
        k95_max = max(k95_per_ctx.values())
        k_act_med = int(np.median(list(k_active_per_ctx.values())))
        k_act_max = max(k_active_per_ctx.values())
        rows.append({
            'K_max': kmax, 'seed': seed, 'macro_f1': f1m,
            'K95_median': k95_med, 'K95_max': k95_max,
            'K_active_median': k_act_med, 'K_active_max': k_act_max,
            'elapsed_s': round(elapsed, 1),
        })
        print(f'  K_max={kmax} seed={seed}: '
              f'F1={f1m:.3f}, K95_med={k95_med}, K95_max={k95_max}, '
              f'K_active_max={k_act_max}, {elapsed:.0f}s', flush=True)

df = pd.DataFrame(rows)
out = Path('docs/thesis/figures/kmax_sweep.csv')
df.to_csv(out, index=False)
print(f'\nSaved: {out}', flush=True)

print('\n=== Summary by K_max ===')
agg = df.groupby('K_max').agg({
    'macro_f1': ['mean', 'std'],
    'K95_median': 'mean',
    'K95_max': 'mean',
    'K_active_max': 'mean',
})
print(agg)
