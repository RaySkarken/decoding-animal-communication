"""Reproduce Assom HP1 baseline numbers (F1 > 0.9 claim) on full corpus.

Paper Section 4: "The permutation test revealed that syllable order did not
affect classification performance (F1 − score > 0.9 for both original and
permuted sequences)."

Paper's protocol (Section 3, HP1_0):
- Zhang-et-al 18 sequence features (Table 1 in Appendix)
- Random Forest classifier
- Permutation test: F1_orig vs F1_perm with shuffled sequences
- NO cross-bat emitter split mentioned in paper → implies random stratified CV

Runs on full corpus (127k, 21,387 vocs):
  A) Zhang-18 + RF + random 5-fold CV (paper's implicit protocol)      — target ≥ 0.9
  B) Zhang-18 + RF + random 5-fold CV, sequences PERMUTED               — target ≥ 0.9
  C) Zhang-18 + RF + emitter-split (our protocol for per-context eval)  — fair baseline
  D) Bag-of-syllables + RF + emitter-split (our current thesis baseline) — apples-to-apples
"""
from __future__ import annotations
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd, joblib
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from src.sequence import compute_sequence_features

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
STATE_PATH = CKPT / 'ablation_state_fullcorpus.joblib'
HDB_LABELS = CKPT / 'hdb_global_labels_fullcorpus.npy'   # 8-cluster vocab

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]

print('Loading fullcorpus state...')
st = joblib.load(STATE_PATH)
seg_df = st['seg_df'].reset_index(drop=True)
hdb_nca = np.load(HDB_LABELS)
ctx = seg_df['context'].to_numpy()
em_abs = np.abs(seg_df['emitter'].to_numpy())
ALL_TYPES = sorted(set(int(l) for l in hdb_nca if l >= 0))
print(f'  N segments: {len(seg_df)}, vocab: {len(ALL_TYPES)} tokens')

# Group into vocalizations
vocs = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids: continue
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    dom_em = int(Counter(np.abs(g['emitter'].to_numpy())).most_common(1)[0][0])
    if dom_ctx not in HP1_CTX: continue
    seq = [int(hdb_nca[i]) for i in seg_ids if hdb_nca[i] >= 0]
    if not seq: continue
    vocs.append({'seq': seq, 'ctx': dom_ctx, 'em': dom_em})
all_bats = sorted(set(v['em'] for v in vocs))
print(f'  N vocs: {len(vocs)}, N bats: {len(all_bats)}')


def zhang18_feats(seq):
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


def bos_feats(seq, V=len(ALL_TYPES)):
    c = Counter(seq); n = len(seq)
    bos = np.zeros(V, dtype=np.float32)
    for k, cnt in c.items():
        if 0 <= k < V: bos[k] = cnt / max(n, 1)
    return np.concatenate([bos, [n, len(c)/max(n,1)]]).astype(np.float32)


def build_matrix(vocs, feat_fn):
    X = np.stack([feat_fn(v['seq']) for v in vocs], axis=0)
    y = np.array([v['ctx'] for v in vocs])
    em = np.array([v['em'] for v in vocs])
    return X, y, em


X18, y18, em18 = build_matrix(vocs, zhang18_feats)
Xbos, ybos, _ = build_matrix(vocs, bos_feats)
print(f'\nFeature matrices: Zhang-18={X18.shape}, BoS={Xbos.shape}')


# ─── A) Zhang-18 + random 5-fold CV ─────────────────────────────────────────
print('\n[A] Zhang-18 + Random 5-fold CV (paper protocol, target F1 > 0.9):')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
f1_A = []
for i, (tr, te) in enumerate(cv.split(X18, y18)):
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=0, n_jobs=-1).fit(X18[tr], y18[tr])
    f1_A.append(f1_score(y18[te], rf.predict(X18[te]), average='weighted',
                          labels=HP1_CTX, zero_division=0))
    print(f'  fold {i}: F1 = {f1_A[-1]:.3f}', flush=True)
print(f'  Mean F1 = {np.mean(f1_A):.3f} ± {np.std(f1_A):.3f}')


# ─── B) Zhang-18 + random 5-fold CV + PERMUTED sequences ────────────────────
print('\n[B] Zhang-18 + Random 5-fold CV + permuted sequences:')
rng = np.random.default_rng(0)
vocs_perm = [{'seq': list(rng.permutation(v['seq'])), 'ctx': v['ctx'], 'em': v['em']} for v in vocs]
X18p, y18p, _ = build_matrix(vocs_perm, zhang18_feats)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
f1_B = []
for i, (tr, te) in enumerate(cv.split(X18p, y18p)):
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=0, n_jobs=-1).fit(X18p[tr], y18p[tr])
    f1_B.append(f1_score(y18p[te], rf.predict(X18p[te]), average='weighted',
                          labels=HP1_CTX, zero_division=0))
    print(f'  fold {i}: F1 = {f1_B[-1]:.3f}', flush=True)
print(f'  Mean F1 = {np.mean(f1_B):.3f} ± {np.std(f1_B):.3f}')


# ─── C) Zhang-18 + emitter-split (our protocol) ─────────────────────────────
print('\n[C] Zhang-18 + emitter-split 30/11, 5 seeds:')
f1_C = []
for s in range(5):
    rng = np.random.default_rng(s)
    ba = np.array(all_bats); rng.shuffle(ba)
    test_set = set(ba[:11].tolist())
    tr_mask = np.array([v['em'] not in test_set for v in vocs])
    te_mask = ~tr_mask
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=s, n_jobs=-1).fit(X18[tr_mask], y18[tr_mask])
    f1 = f1_score(y18[te_mask], rf.predict(X18[te_mask]),
                   average='weighted', labels=HP1_CTX, zero_division=0)
    f1_C.append(f1)
    print(f'  seed {s}: F1 = {f1:.3f}', flush=True)
print(f'  Mean F1 = {np.mean(f1_C):.3f} ± {np.std(f1_C):.3f}')


# ─── D) Bag-of-syllables + emitter-split ────────────────────────────────────
print('\n[D] Bag-of-syllables + emitter-split 30/11, 5 seeds (current thesis baseline):')
f1_D = []
for s in range(5):
    rng = np.random.default_rng(s)
    ba = np.array(all_bats); rng.shuffle(ba)
    test_set = set(ba[:11].tolist())
    tr_mask = np.array([v['em'] not in test_set for v in vocs])
    te_mask = ~tr_mask
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=s, n_jobs=-1).fit(Xbos[tr_mask], ybos[tr_mask])
    f1 = f1_score(ybos[te_mask], rf.predict(Xbos[te_mask]),
                   average='weighted', labels=HP1_CTX, zero_division=0)
    f1_D.append(f1)
    print(f'  seed {s}: F1 = {f1:.3f}', flush=True)
print(f'  Mean F1 = {np.mean(f1_D):.3f} ± {np.std(f1_D):.3f}')


print('\n\n=== SUMMARY TABLE ===')
print(f'{"Protocol":50s} {"F1 mean":>8s} {"± SD":>8s}')
print('-' * 68)
print(f'{"A) Zhang-18 + random CV (paper protocol)":50s} {np.mean(f1_A):>8.3f} {np.std(f1_A):>8.3f}  [paper: > 0.9]')
print(f'{"B) Zhang-18 + random CV, PERMUTED":50s} {np.mean(f1_B):>8.3f} {np.std(f1_B):>8.3f}  [paper: > 0.9]')
print(f'{"C) Zhang-18 + emitter-split":50s} {np.mean(f1_C):>8.3f} {np.std(f1_C):>8.3f}')
print(f'{"D) Bag-of-syl + emitter-split (thesis baseline)":50s} {np.mean(f1_D):>8.3f} {np.std(f1_D):>8.3f}')
print(f'\nOrig vs Permuted (A - B): {np.mean(f1_A) - np.mean(f1_B):+.3f}  [paper: ≈ 0, no effect]')
print(f'Random CV vs emitter-split gap (A - C): {np.mean(f1_A) - np.mean(f1_C):+.3f}')

pd.DataFrame({
    'protocol': ['A_zhang18_random_cv', 'B_zhang18_random_cv_perm',
                 'C_zhang18_emitter_split', 'D_bos_emitter_split'],
    'f1_mean': [np.mean(f1_A), np.mean(f1_B), np.mean(f1_C), np.mean(f1_D)],
    'f1_std': [np.std(f1_A), np.std(f1_B), np.std(f1_C), np.std(f1_D)],
}).to_csv('docs/thesis/figures/hp1_baseline_reproduction_fullcorpus.csv', index=False)
print('\nSaved: docs/thesis/figures/hp1_baseline_reproduction_fullcorpus.csv')
