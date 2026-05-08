"""Diagnostic: does emitter-split really explain the baseline F1 drop?

Запускает ОДИН И ТОТ ЖЕ baseline (Assom HDBSCAN+NCA labels → bag-of-syllables
+ длина/энтропия/повтор → RandomForest) в двух протоколах:

  A. Random 5-fold stratified CV по вокализациям (утечка через эмиттера).
  B. Emitter-split 30 train / 11 test (multi-seed, без утечки).

Разница F1(A) − F1(B) = чистый эффект утечки через голос эмиттера.

Дополнительно запускается та же пара протоколов с расширенным признаковым
пространством (bag-of-syllables + bigram counts + ensemble), чтобы отделить
эффект признаков от эффекта протокола.
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

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
seg_df = st['seg_df']
ctx = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()
hdb_nca = st['hdb_nca_labels']

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]

# group vocalizations
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
print(f'Total vocalizations: {len(vocs)}')
V = int(np.max(hdb_nca)) + 1
print(f'Global vocabulary size: {V}')


def feats_simple(seq, V):
    """bag-of-syllables + 4 aggregates (same as main experiment baseline)."""
    c = Counter(seq); n = len(seq)
    bos = np.zeros(V, dtype=np.float32)
    for k, cnt in c.items():
        if 0 <= k < V:
            bos[k] = cnt / max(n, 1)
    probs = np.array(list(c.values()), dtype=np.float32) / max(n, 1)
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    richness = len(c) / max(n, 1)
    rep = max(c.values()) / max(n, 1) if c else 0.0
    return np.concatenate([bos, [n, richness, ent, rep]]).astype(np.float32)


def feats_bigram(seq, V):
    """bag-of-syllables + bigram-counts + aggregates (richer)."""
    base = feats_simple(seq, V)
    bg = np.zeros(V * V, dtype=np.float32)
    if len(seq) >= 2:
        total = len(seq) - 1
        for a, b in zip(seq[:-1], seq[1:]):
            if 0 <= a < V and 0 <= b < V:
                bg[a * V + b] += 1.0 / total
    return np.concatenate([base, bg]).astype(np.float32)


def build_matrix(vocs_subset, feat_fn):
    X, y, e = [], [], []
    for v in vocs_subset:
        labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
        if not labs:
            continue
        X.append(feat_fn(labs, V))
        y.append(v['ctx'])
        e.append(v['em'])
    return np.array(X), np.array(y), np.array(e)


# ---- Protocol A: random stratified 5-fold CV ----
def run_random_cv(feat_fn):
    X, y, _ = build_matrix(vocs, feat_fn)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    f1s = []
    for tr, te in cv.split(X, y):
        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                     random_state=0, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        pred = rf.predict(X[te])
        f1s.append(f1_score(y[te], pred, average='weighted',
                             labels=HP1_CTX, zero_division=0))
    return float(np.mean(f1s)), float(np.std(f1s))


# ---- Protocol B: emitter-split (5 seeds) ----
def run_emitter_split(feat_fn):
    all_em = sorted(set(v['em'] for v in vocs))
    f1s = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        em_arr = np.array(all_em); rng.shuffle(em_arr)
        test_em = set(em_arr[:11].tolist())
        train_vocs = [v for v in vocs if v['em'] not in test_em]
        test_vocs = [v for v in vocs if v['em'] in test_em]
        Xt, yt, _ = build_matrix(train_vocs, feat_fn)
        Xe, ye, _ = build_matrix(test_vocs, feat_fn)
        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                     random_state=seed, n_jobs=-1)
        rf.fit(Xt, yt)
        pred = rf.predict(Xe)
        f1s.append(f1_score(ye, pred, average='weighted',
                             labels=HP1_CTX, zero_division=0))
    return float(np.mean(f1s)), float(np.std(f1s))


rows = []
for feat_name, feat_fn in [('simple (BoS+4agg)', feats_simple),
                             ('bigram (BoS+bigrams+agg)', feats_bigram)]:
    print(f'\n--- features: {feat_name} ---')
    print('  random 5-fold CV ...', flush=True)
    m_rand, s_rand = run_random_cv(feat_fn)
    print(f'    F1 = {m_rand:.3f} ± {s_rand:.3f}')
    print('  emitter-split (5 seeds) ...', flush=True)
    m_em, s_em = run_emitter_split(feat_fn)
    print(f'    F1 = {m_em:.3f} ± {s_em:.3f}')
    print(f'  === drop from random CV to emitter-split: {m_rand - m_em:+.3f}')
    rows.append({'features': feat_name,
                 'random_cv_f1': round(m_rand, 3),
                 'random_cv_std': round(s_rand, 3),
                 'emitter_split_f1': round(m_em, 3),
                 'emitter_split_std': round(s_em, 3),
                 'drop_due_to_split': round(m_rand - m_em, 3)})

df = pd.DataFrame(rows)
print('\n\n=== SPLIT-PROTOCOL DIAGNOSIS ===')
print(df.to_string(index=False))

Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/split_diagnosis.csv', index=False)
print('\nSaved to docs/thesis/figures/split_diagnosis.csv')

print('\n\nInterpretation:')
print('- if drop_due_to_split >> drop_due_to_features, then emitter leakage dominates')
print('- if drops are comparable, both feature set and split matter')
