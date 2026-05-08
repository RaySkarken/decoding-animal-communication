"""Check if classification improves by removing per-context prior bias.

In the main experiment, classifier uses log p_c(x) + log p(c) where p(c)
is estimated from training class frequency. This biases prediction toward
over-represented classes (Mating has 21% of vocalizations, Kissing has 2%).

This variant uses uniform prior: hat_c = argmax_c log p_c(x).
Keeps DP-GMM per-context tokenizers; only scoring rule changes.
Multi-seed emitter-split 30/11, F1_weighted over 8 contexts.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import f1_score, classification_report

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
emb = st['embedding']
seg_df = st['seg_df']
ctx = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAME = {2: 'Biting', 3: 'Feeding', 4: 'Fighting', 5: 'Grooming',
            6: 'Isolation', 7: 'Kissing', 9: 'Mating', 10: 'Threat'}

vocs = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids: continue
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
    if dom_ctx not in HP1_CTX: continue
    vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em})
all_emitters = sorted(set(v['em'] for v in vocs))


def run(seed, k_components=15):
    rng = np.random.default_rng(seed)
    em_arr = np.array(all_emitters); rng.shuffle(em_arr)
    test_em = set(em_arr[:11].tolist())
    train_vocs = [v for v in vocs if v['em'] not in test_em]
    test_vocs = [v for v in vocs if v['em'] in test_em]
    if not train_vocs or not test_vocs:
        return None
    train_seg_mask = np.zeros(len(emb), dtype=bool)
    for v in train_vocs:
        train_seg_mask[v['seg_ids']] = True
    n_tv = len(train_vocs)
    log_prior = {}
    tokenizers = {}
    for c in HP1_CTX:
        mask = train_seg_mask & (ctx == c)
        if mask.sum() < 30: continue
        X_c = emb[mask]
        bgm = BayesianGaussianMixture(
            n_components=k_components,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1, covariance_type='full',
            max_iter=150, random_state=seed).fit(X_c)
        tokenizers[c] = bgm
        nc = sum(1 for v in train_vocs if v['ctx'] == c)
        log_prior[c] = np.log(max(nc, 1) / n_tv)

    # inference with two prior choices
    y_true, y_pred_empirical, y_pred_uniform = [], [], []
    for v in test_vocs:
        X_seq = emb[v['seg_ids']]
        if len(X_seq) == 0: continue
        lls = {}
        for c, tok in tokenizers.items():
            lls[c] = tok.score_samples(X_seq).sum()
        y_true.append(v['ctx'])
        y_pred_empirical.append(max(lls, key=lambda c: lls[c] + log_prior[c]))
        y_pred_uniform.append(max(lls, key=lambda c: lls[c]))
    y_true = np.array(y_true)
    y_pred_empirical = np.array(y_pred_empirical)
    y_pred_uniform = np.array(y_pred_uniform)

    return {
        'seed': seed,
        'n_test': len(y_true),
        'f1_empirical_prior': round(f1_score(y_true, y_pred_empirical,
            average='weighted', labels=HP1_CTX, zero_division=0), 3),
        'f1_uniform_prior': round(f1_score(y_true, y_pred_uniform,
            average='weighted', labels=HP1_CTX, zero_division=0), 3),
        'f1_empirical_macro': round(f1_score(y_true, y_pred_empirical,
            average='macro', labels=HP1_CTX, zero_division=0), 3),
        'f1_uniform_macro': round(f1_score(y_true, y_pred_uniform,
            average='macro', labels=HP1_CTX, zero_division=0), 3),
    }, (y_true, y_pred_empirical, y_pred_uniform)


rows = []
diag = None
for s in range(5):
    print(f'seed {s}...', flush=True)
    r, d = run(s)
    rows.append(r)
    print(f"  weighted F1: empirical={r['f1_empirical_prior']}  uniform={r['f1_uniform_prior']}")
    print(f"  macro F1:    empirical={r['f1_empirical_macro']}  uniform={r['f1_uniform_macro']}")
    if s == 0:
        diag = d

df = pd.DataFrame(rows)
print('\n\n=== UNIFORM-PRIOR VARIANT SUMMARY ===')
print(df.to_string(index=False))

print('\n=== AGGREGATE (5 seeds) ===')
for col in ['f1_empirical_prior', 'f1_uniform_prior',
            'f1_empirical_macro', 'f1_uniform_macro']:
    print(f'  {col:22s} {df[col].mean():.3f} ± {df[col].std():.3f}')

Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/uniform_prior_variant.csv', index=False)
print('\nSaved to docs/thesis/figures/uniform_prior_variant.csv')

# per-class seed 0
if diag is not None:
    y_true, y_emp, y_uni = diag
    print('\n=== PER-CLASS F1 (seed 0) ===')
    print(f'{"context":<12s} {"empirical":>10s} {"uniform":>10s} {"n_test":>8s}')
    for c in HP1_CTX:
        f_emp = f1_score(y_true == c, y_emp == c, zero_division=0)
        f_uni = f1_score(y_true == c, y_uni == c, zero_division=0)
        n = (y_true == c).sum()
        print(f'{CTX_NAME[c]:<12s} {f_emp:>10.3f} {f_uni:>10.3f} {n:>8d}')
