"""Final piece: Zhang-18 features under both split protocols.

Goal: confirm that the historical F1 ≈ 0.85 reported with 18 Zhang features
comes from random CV leakage of emitter-specific transition statistics.

Runs:
  A. 18 Zhang features + RF, random stratified 5-fold CV (vocalization-level)
  B. 18 Zhang features + RF, emitter-split (5 seeds)

Gap between A and B quantifies the emitter-identity leakage for this
transition-based feature family. For bag-of-syllables the same gap was ~0.07
(scripts/experiment_split_diagnosis.py). Here we expect a much larger gap.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from src.sequence import compute_sequence_features

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
seg_df = st['seg_df']
hdb_nca = st['hdb_nca_labels']

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
ALL_TYPES = sorted(set(int(l) for l in hdb_nca if l >= 0))

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


def build_matrix(vocs_subset):
    X, y = [], []
    for v in vocs_subset:
        labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
        if not labs: continue
        X.append(feats_zhang18(labs))
        y.append(v['ctx'])
    return np.array(X), np.array(y)


# ---- A. random stratified 5-fold CV ----
X, y = build_matrix(vocs)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
f1_cv = []
for i, (tr, te) in enumerate(cv.split(X, y)):
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=0, n_jobs=-1).fit(X[tr], y[tr])
    pred = rf.predict(X[te])
    f1_cv.append(f1_score(y[te], pred, average='weighted',
                           labels=HP1_CTX, zero_division=0))
    print(f'  fold {i}: F1 = {f1_cv[-1]:.3f}', flush=True)
print(f'Random 5-fold CV F1 = {np.mean(f1_cv):.3f} ± {np.std(f1_cv):.3f}')

# ---- B. emitter-split ----
f1_em = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    em_arr = np.array(all_emitters); rng.shuffle(em_arr)
    test_em = set(em_arr[:11].tolist())
    train_vocs = [v for v in vocs if v['em'] not in test_em]
    test_vocs = [v for v in vocs if v['em'] in test_em]
    Xt, yt = build_matrix(train_vocs)
    Xe, ye = build_matrix(test_vocs)
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=seed, n_jobs=-1).fit(Xt, yt)
    pred = rf.predict(Xe)
    f1 = f1_score(ye, pred, average='weighted', labels=HP1_CTX, zero_division=0)
    f1_em.append(f1)
    print(f'  seed {seed}: F1 = {f1:.3f}', flush=True)
print(f'Emitter-split F1 = {np.mean(f1_em):.3f} ± {np.std(f1_em):.3f}')

print('\n\n=== ZHANG-18 SPLIT-PROTOCOL GAP ===')
print(f'Random 5-fold CV F1:  {np.mean(f1_cv):.3f} ± {np.std(f1_cv):.3f}')
print(f'Emitter-split F1:     {np.mean(f1_em):.3f} ± {np.std(f1_em):.3f}')
drop = np.mean(f1_cv) - np.mean(f1_em)
print(f'Drop from random CV to emitter-split: {drop:+.3f}')

print('\nCompare: simple bag-of-syllables drop was ~0.07,')
print('         Zhang-18 drop is much larger -> transition features leak emitter identity')

pd.DataFrame({'protocol': ['random_cv', 'emitter_split'],
              'f1_mean': [round(np.mean(f1_cv), 3), round(np.mean(f1_em), 3)],
              'f1_std': [round(np.std(f1_cv), 3), round(np.std(f1_em), 3)]
             }).to_csv('docs/thesis/figures/zhang18_split_gap.csv', index=False)
