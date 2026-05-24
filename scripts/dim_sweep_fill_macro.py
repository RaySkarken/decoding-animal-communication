"""Заполнение пропущенных ячеек таблицы dim sweep macro F1.

Считаем macro F1 (5 разбиений по особям, cross-bat) для конфигураций:
  - DP-GMM full на UMAP-2D, UMAP-16D с обоими приорами (есть weighted, нет macro)
  - DP-GMM diag на UMAP-32D с обоими приорами (нет ни weighted, ни macro)

Не пересчитываем:
  - UMAP-32D с DP-GMM full (известно: не сходится)
  - BEATs-768D (отдельный foundation-модельный эксперимент из §3.6.3)

Output: docs/thesis/figures/dim_sweep_fill_macro.csv
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import f1_score
import time

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]


def load_state_and_emb(emb_filename: str):
    st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
    seg_df = st['seg_df'].reset_index(drop=True)
    emb = np.load(CACHE / emb_filename)
    return seg_df, emb


def build_vocs(seg_df):
    vocs = []
    for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
        seg_ids = g.index.to_list()
        if not seg_ids: continue
        dom_em_signed = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
        if dom_em_signed == 0: continue
        dom_em_abs = abs(dom_em_signed)
        dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
        if dom_ctx not in HP1_CTX: continue
        vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em_abs})
    return vocs


def evaluate(emb, vocs, ctx_arr, cov_type: str, prior_mode: str, n_seeds: int = 5):
    """Cross-bat eval: per-context BayesianGaussianMixture + max-likelihood classifier."""
    all_bats = sorted(set(v['em'] for v in vocs))
    macro_per_seed, weighted_per_seed = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        ba = np.array(all_bats); rng.shuffle(ba)
        test_b = set(ba[:11].tolist())
        train_v = [v for v in vocs if v['em'] not in test_b]
        test_v = [v for v in vocs if v['em'] in test_b]
        if not test_v: continue
        train_mask = np.zeros(len(emb), dtype=bool)
        for v in train_v: train_mask[v['seg_ids']] = True

        if prior_mode == 'emp':
            n_tv = len(train_v)
            log_prior = {c: np.log(max(sum(1 for v in train_v if v['ctx'] == c), 1) / n_tv)
                         for c in HP1_CTX}
        else:
            log_prior = {c: -np.log(len(HP1_CTX)) for c in HP1_CTX}

        toks = {}
        for c in HP1_CTX:
            m = train_mask & (ctx_arr == c)
            if m.sum() < 30: continue
            try:
                toks[c] = BayesianGaussianMixture(
                    n_components=15,
                    weight_concentration_prior_type='dirichlet_process',
                    weight_concentration_prior=0.1, covariance_type=cov_type,
                    max_iter=150, random_state=s
                ).fit(emb[m])
            except Exception as e:
                print(f'    skip ctx={c}: {e}')

        yt, yp = [], []
        for v in test_v:
            X = emb[v['seg_ids']]
            if len(X) == 0: continue
            best, bs = None, -np.inf
            for c, t in toks.items():
                ll = t.score_samples(X).sum() + log_prior[c]
                if ll > bs: bs = ll; best = c
            if best is None: continue
            yt.append(v['ctx']); yp.append(best)
        if not yt: continue
        yt_a = np.array(yt); yp_a = np.array(yp)
        macro = f1_score(yt_a, yp_a, average='macro', labels=HP1_CTX, zero_division=0)
        weighted = f1_score(yt_a, yp_a, average='weighted', labels=HP1_CTX, zero_division=0)
        macro_per_seed.append(macro); weighted_per_seed.append(weighted)
    macro_arr = np.array(macro_per_seed)
    weighted_arr = np.array(weighted_per_seed)
    return macro_arr.mean(), macro_arr.std(), weighted_arr.mean(), weighted_arr.std(), len(macro_arr)


# Конфигурации для пересчёта
EMB_FILES = {
    'UMAP-2D': 'umap_2d.npy',
    'UMAP-8D': 'umap_152k_21x32_md1.0_8d.npy',
    'UMAP-16D': 'umap_16d.npy',
    'UMAP-32D': 'umap_32d.npy',
}

# Что заполняем: (cov_type, prior, embedding)
TARGETS = [
    ('full', 'emp', 'UMAP-2D'), ('full', 'uni', 'UMAP-2D'),
    ('full', 'emp', 'UMAP-16D'), ('full', 'uni', 'UMAP-16D'),
    ('diag', 'emp', 'UMAP-32D'), ('diag', 'uni', 'UMAP-32D'),
]


def main():
    print('Loading state and vocs...', flush=True)
    seg_df, emb_2d = load_state_and_emb(EMB_FILES['UMAP-2D'])
    vocs = build_vocs(seg_df)
    ctx_arr = seg_df['context'].to_numpy()
    print(f'  vocs: {len(vocs)}', flush=True)

    cache_emb = {'UMAP-2D': emb_2d}
    rows = []
    for cov, prior, emb_name in TARGETS:
        if emb_name not in cache_emb:
            print(f'Loading {emb_name}...', flush=True)
            try:
                cache_emb[emb_name] = np.load(CACHE / EMB_FILES[emb_name])
            except FileNotFoundError:
                print(f'  {emb_name} not found, skip', flush=True)
                rows.append({'cov': cov, 'prior': prior, 'emb': emb_name,
                             'macro_mean': np.nan, 'macro_std': np.nan,
                             'weighted_mean': np.nan, 'weighted_std': np.nan,
                             'n_seeds': 0, 'note': 'embedding file not found'})
                continue
        e = cache_emb[emb_name]
        t0 = time.time()
        print(f'  {cov:10s} {prior:4s} {emb_name:8s}: ', end='', flush=True)
        try:
            mm, ms, wm, ws, n = evaluate(e, vocs, ctx_arr, cov, prior)
            elapsed = time.time() - t0
            print(f'macro={mm:.3f}±{ms:.3f}, weighted={wm:.3f}±{ws:.3f}, n={n}  ({elapsed:.0f}s)',
                  flush=True)
            rows.append({'cov': cov, 'prior': prior, 'emb': emb_name,
                         'macro_mean': mm, 'macro_std': ms,
                         'weighted_mean': wm, 'weighted_std': ws,
                         'n_seeds': n, 'note': ''})
        except Exception as e:
            elapsed = time.time() - t0
            print(f'failed: {e}  ({elapsed:.0f}s)', flush=True)
            rows.append({'cov': cov, 'prior': prior, 'emb': emb_name,
                         'macro_mean': np.nan, 'macro_std': np.nan,
                         'weighted_mean': np.nan, 'weighted_std': np.nan,
                         'n_seeds': 0, 'note': str(e)[:200]})

    df = pd.DataFrame(rows)
    out = Path('docs/thesis/figures/dim_sweep_fill_macro.csv')
    df.to_csv(out, index=False)
    print(f'\nSaved {out}')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
