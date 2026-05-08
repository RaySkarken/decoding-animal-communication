"""Honest baseline comparison: per-context vs Assom+RF on 18 Zhang features.

Previous main experiment compared per-context method against a WEAK baseline
(bag-of-syllables + 4 aggregates). The diagnosis (scripts/experiment_split_
diagnosis.py) showed that emitter-leakage accounts for only ~0.07 F1 drop,
so the large reported historical F1 ≈ 0.85 was driven primarily by the RICH
feature set (18 Zhang transition-based features), not by data leakage.

This script runs a HONEST apples-to-apples comparison between:
  A. Per-context DP-GMM + Bayes classification (our method)
  B. Assom labels → 18 Zhang features → RF (strong baseline)
  C. Assom labels → simple bag-of-syllables → RF (weak baseline, for reference)

Protocol: emitter-split 30/11, 5 seeds. Output: F1, gain, 95%-CI.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.sequence import compute_sequence_features

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
emb = st['embedding']
seg_df = st['seg_df']
ctx = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()
hdb_nca = st['hdb_nca_labels']

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
ALL_TYPES = sorted(set(int(l) for l in hdb_nca if l >= 0))
print(f'Assom vocabulary types: {ALL_TYPES}')

vocs = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids:
        continue
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
    if dom_ctx not in HP1_CTX:
        continue
    vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em})
all_emitters = sorted(set(v['em'] for v in vocs))
print(f'Vocalizations: {len(vocs)}, emitters: {len(all_emitters)}')


# ---- feature extractors ----
def feats_zhang18(seq):
    d = compute_sequence_features(seq, ALL_TYPES)
    return np.array([d['a_seq_length'], d['b_richness'], d['c_versatility'],
                     d['d_entropy'], d['e_linearity'], d['f_n_transitions'],
                     d['g_mean_trans_prob'], d['h_std_trans_prob'],
                     d['i_max_trans_prob'], d['j_min_trans_prob'],
                     d['k_self_loop_prob'], d['l_unique_trigrams'],
                     d['m_max_type_freq'], d['n_min_type_freq'],
                     d['o_std_type_freq'], d['p_mean_type_freq'],
                     d['q_graph_density'], d['r_n_types_total']],
                    dtype=np.float32)

V = int(np.max(hdb_nca)) + 1
def feats_simple(seq):
    c = Counter(seq); n = len(seq)
    bos = np.zeros(V, dtype=np.float32)
    for k, cnt in c.items():
        if 0 <= k < V: bos[k] = cnt / max(n, 1)
    probs = np.array(list(c.values()), dtype=np.float32) / max(n, 1)
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    richness = len(c) / max(n, 1)
    rep = max(c.values()) / max(n, 1) if c else 0.0
    return np.concatenate([bos, [n, richness, ent, rep]]).astype(np.float32)


def build_global_matrix(vocs_subset, feat_fn):
    X, y = [], []
    for v in vocs_subset:
        labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
        if not labs: continue
        X.append(feat_fn(labs))
        y.append(v['ctx'])
    return np.array(X), np.array(y)


def run_percontext_dpgmm(train_vocs, test_vocs, seed, k=15):
    train_seg_mask = np.zeros(len(emb), dtype=bool)
    for v in train_vocs:
        train_seg_mask[v['seg_ids']] = True
    n_tv = len(train_vocs)
    log_prior = {}
    tokenizers = {}
    for c in HP1_CTX:
        mask = train_seg_mask & (ctx == c)
        if mask.sum() < 30: continue
        bgm = BayesianGaussianMixture(
            n_components=k, weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1, covariance_type='full',
            max_iter=150, random_state=seed).fit(emb[mask])
        tokenizers[c] = bgm
        nc = sum(1 for v in train_vocs if v['ctx'] == c)
        log_prior[c] = np.log(max(nc, 1) / n_tv)
    y_true, y_pred = [], []
    for v in test_vocs:
        X_seq = emb[v['seg_ids']]
        if len(X_seq) == 0: continue
        best_c, best = None, -np.inf
        for c, tok in tokenizers.items():
            s = tok.score_samples(X_seq).sum() + log_prior[c]
            if s > best:
                best = s; best_c = c
        if best_c is None: continue
        y_true.append(v['ctx']); y_pred.append(best_c)
    return np.array(y_true), np.array(y_pred)


def run_rf_baseline(train_vocs, test_vocs, seed, feat_fn):
    Xt, yt = build_global_matrix(train_vocs, feat_fn)
    Xe, ye = build_global_matrix(test_vocs, feat_fn)
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=seed, n_jobs=-1)
    rf.fit(Xt, yt)
    pred = rf.predict(Xe)
    return ye, pred


rows = []
for seed in range(5):
    print(f'seed {seed}...', flush=True)
    rng = np.random.default_rng(seed)
    em_arr = np.array(all_emitters); rng.shuffle(em_arr)
    test_em = set(em_arr[:11].tolist())
    train_vocs = [v for v in vocs if v['em'] not in test_em]
    test_vocs = [v for v in vocs if v['em'] in test_em]

    yt_pc, yp_pc = run_percontext_dpgmm(train_vocs, test_vocs, seed)
    yt_z, yp_z = run_rf_baseline(train_vocs, test_vocs, seed, feats_zhang18)
    yt_s, yp_s = run_rf_baseline(train_vocs, test_vocs, seed, feats_simple)

    fm = lambda y, p: round(f1_score(y, p, average='weighted',
                                       labels=HP1_CTX, zero_division=0), 3)
    r = {
        'seed': seed,
        'pc_dpgmm_f1': fm(yt_pc, yp_pc),
        'baseline_zhang18_f1': fm(yt_z, yp_z),
        'baseline_simple_f1': fm(yt_s, yp_s),
    }
    r['gain_vs_zhang18'] = round(r['pc_dpgmm_f1'] - r['baseline_zhang18_f1'], 3)
    r['gain_vs_simple'] = round(r['pc_dpgmm_f1'] - r['baseline_simple_f1'], 3)
    rows.append(r)
    print(f'  per-context DP-GMM F1 = {r["pc_dpgmm_f1"]}')
    print(f'  baseline Zhang18   F1 = {r["baseline_zhang18_f1"]}')
    print(f'  baseline simple    F1 = {r["baseline_simple_f1"]}')
    print(f'  gain vs Zhang18     = {r["gain_vs_zhang18"]:+.3f}')
    print(f'  gain vs simple      = {r["gain_vs_simple"]:+.3f}')

df = pd.DataFrame(rows)
print('\n\n=== HONEST STRONG-BASELINE COMPARISON ===')
print(df.to_string(index=False))

print('\n=== AGGREGATE (5 seeds) ===')
for col in ['pc_dpgmm_f1', 'baseline_zhang18_f1', 'baseline_simple_f1']:
    print(f'  {col:24s} {df[col].mean():.3f} ± {df[col].std():.3f}')
for col in ['gain_vs_zhang18', 'gain_vs_simple']:
    m, s = df[col].mean(), df[col].std()
    ci_lo, ci_hi = m - 2*s, m + 2*s
    sig = 'SIG+' if ci_lo > 0 else ('SIG-' if ci_hi < 0 else 'NS')
    print(f'  {col:24s} {m:+.3f} ± {s:.3f}  95%-CI [{ci_lo:+.3f},{ci_hi:+.3f}]  {sig}')

Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/strong_baseline_comparison.csv', index=False)
print('\nSaved to docs/thesis/figures/strong_baseline_comparison.csv')
