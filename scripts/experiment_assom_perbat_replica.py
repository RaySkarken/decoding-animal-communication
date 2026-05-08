"""Replicate Assom's per-bat classifier protocol to recover F1 ≈ 0.9.

Assom's Exp1-Classifier.ipynb runs the HP1 protocol as follows:
  - Pick one bat (BAT_ID).
  - Take ALL its vocalizations across contexts.
  - train_test_split with stratify=y, test_size=0.25.
  - Fit RF(criterion='entropy', n_estimators=100).
  - Compute weighted F1.

This is WITHIN-BAT classification, fundamentally different from our
cross-bat emitter-split protocol.

Here we replicate it on our data with Assom's hdb_nca labels and 18 Zhang
features. If result ≈ 0.9, the historical figure is reproducible and the
gap to our cross-bat setup is what we've been trying to explain.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.sequence import compute_sequence_features

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
seg_df = st['seg_df']
hdb_nca = st['hdb_nca_labels']

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
ALL_TYPES = sorted(set(int(l) for l in hdb_nca if l >= 0))

# group vocalizations
vocs = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids: continue
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
    if dom_ctx not in HP1_CTX:
        continue
    vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em})


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


all_emitters = sorted(set(v['em'] for v in vocs))
print(f'Emitters: {len(all_emitters)}')


def build_matrix_one_bat(bat_id):
    X, y = [], []
    for v in vocs:
        if v['em'] != bat_id:
            continue
        labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
        if not labs: continue
        X.append(feats_zhang18(labs))
        y.append(v['ctx'])
    return np.array(X), np.array(y)


# Try a few bats with enough data
print('\n=== PER-BAT HP1 CLASSIFICATION (Assom-style) ===')
print(f'{"bat":>5s} {"n_vocs":>7s} {"n_ctx":>6s} {"F1_weighted":>12s} {"F1_macro":>10s}')
results = []
for bat in all_emitters:
    Xb, yb = build_matrix_one_bat(bat)
    if len(yb) < 30:
        continue
    uniq = sorted(set(yb))
    if len(uniq) < 2:
        continue
    try:
        # stratified 75/25 split (Assom's exact protocol)
        min_count = min(Counter(yb).values())
        if min_count < 2:
            # fall back to non-stratified
            Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.25, random_state=0)
        else:
            Xtr, Xte, ytr, yte = train_test_split(Xb, yb, stratify=yb,
                                                     test_size=0.25, random_state=0)
        rf = RandomForestClassifier(criterion='entropy', n_estimators=100,
                                     random_state=0, n_jobs=-1).fit(Xtr, ytr)
        pred = rf.predict(Xte)
        f_w = f1_score(yte, pred, average='weighted', zero_division=0)
        f_m = f1_score(yte, pred, average='macro', zero_division=0)
        results.append({'bat': bat, 'n_vocs': len(yb), 'n_ctx': len(uniq),
                         'f1_weighted': round(f_w, 3), 'f1_macro': round(f_m, 3)})
        print(f'{bat:>5d} {len(yb):>7d} {len(uniq):>6d} {f_w:>12.3f} {f_m:>10.3f}')
    except Exception as e:
        print(f'  bat {bat} failed: {e}')

df = pd.DataFrame(results)
print('\n=== AGGREGATE OVER BATS ===')
print(f'F1_weighted: {df["f1_weighted"].mean():.3f} ± {df["f1_weighted"].std():.3f}')
print(f'F1_macro:    {df["f1_macro"].mean():.3f} ± {df["f1_macro"].std():.3f}')
print(f'n_bats:      {len(df)}')

Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/assom_perbat_replica.csv', index=False)
print('\nSaved to docs/thesis/figures/assom_perbat_replica.csv')

print('\nIf F1_weighted ≈ 0.9: Assom protocol is within-bat classification,')
print('fundamentally different from cross-bat emitter-split used in our main experiment.')
