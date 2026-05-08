"""
Experiment B: Levenshtein-kNN HP1 test (Sarkar & Magimai-Doss 2025).

The existing bag-of-syllables + Zhang-18-feature HP1 test is INFLATABLE
by vocab size: more clusters → more entropy → higher F1_inv trivially.
Permutation tests show order doesn't matter, but this is unfortunately
consistent with vocab-inflation artifact too.

Levenshtein-kNN classification is ORDER-SENSITIVE BY CONSTRUCTION:
  - Distance between seqs = edit distance (order changes → distance changes)
  - kNN classifier picks up on actual sequential structure
  - Under random permutation: edit distances scramble → F1 should drop
    if real order signal exists, stay same if truly associative

Applied to:
  - Baseline HDBSCAN labels (vocab 6)
  - HDBSCAN UMAP-16D (our best AMI from Exp A)
  - DP-GMM UMAP-2D (vocab 21)

For each: compute F1_lev on original vs permuted sequences.
If delta > 0 → order matters → not purely associative.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from collections import Counter
import hdbscan
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
seg_df = st['seg_df']
embedding = st['embedding']
hdbnca = st['hdb_nca_labels']
RANDOM_STATE = 0

# Build per-file sequences
sequences_per_file, contexts_per_seq = [], []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if len(seg_ids) < 2: continue
    sequences_per_file.append(seg_ids)
    contexts_per_seq.append(int(np.bincount(g['context'].to_numpy()).argmax()))
print(f'Sequences: {len(sequences_per_file)}')

# Limit to 8 HP1 contexts per paper
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
keep = [(s, c) for s, c in zip(sequences_per_file, contexts_per_seq) if c in HP1_CTX]
sequences_per_file = [s for s, c in keep]
contexts_per_seq = [c for s, c in keep]
print(f'HP1-context sequences: {len(sequences_per_file)}')


def levenshtein(a, b):
    """Standard Levenshtein edit distance between two sequences of hashable items."""
    if len(a) < len(b): a, b = b, a
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0]*len(b)
        for j, cb in enumerate(b, 1):
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            sub = prev[j-1] + (ca != cb)
            cur[j] = min(ins, dele, sub)
        prev = cur
    return prev[-1]


def knn_lev_f1(sequences, contexts, k=5, n_splits=5, subsample=2000, permute=False, seed=0):
    """kNN classification using Levenshtein distance in stratified CV."""
    rng = np.random.default_rng(seed)
    y = np.asarray(contexts)

    # Subsample for feasibility (|seqs|^2 distances is expensive)
    if len(sequences) > subsample:
        idx = rng.choice(len(sequences), size=subsample, replace=False)
        sequences = [sequences[i] for i in idx]
        y = y[idx]

    # Permute within each sequence if requested
    if permute:
        sequences = [list(rng.permutation(s)) for s in sequences]

    # Precompute pairwise distance matrix (symmetric)
    n = len(sequences)
    D = np.zeros((n, n), dtype=np.float32)
    from tqdm.auto import tqdm
    for i in tqdm(range(n), desc='pairwise lev dist', disable=True):
        for j in range(i+1, n):
            d = levenshtein(sequences[i], sequences[j])
            D[i, j] = d
            D[j, i] = d

    # Stratified CV with kNN
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    f1s = []
    for tr, te in cv.split(range(n), y):
        y_pred = []
        for i in te:
            # nearest k training neighbours
            dists = D[i, tr]
            nn_idx = np.argsort(dists)[:k]
            nn_labels = y[tr][nn_idx]
            # majority vote
            cnt = Counter(nn_labels)
            y_pred.append(cnt.most_common(1)[0][0])
        f1s.append(f1_score(y[te], y_pred, average='weighted'))
    return float(np.mean(f1s)), float(np.std(f1s))


def build_label_sequences(labels, seqs_per_file):
    return [[int(labels[i]) for i in seg_ids if labels[i] >= 0]
             for seg_ids in seqs_per_file]


# Method 1: baseline HDBSCAN+NCA (6 atoms)
seqs_base = build_label_sequences(hdbnca, sequences_per_file)
# Filter to non-empty sequences WITH context
pairs = [(s, c) for s, c in zip(seqs_base, contexts_per_seq) if len(s) >= 2]
seqs_base = [s for s, c in pairs]
ctx_base = [c for s, c in pairs]
print(f'Baseline sequences (after filter): {len(seqs_base)}')

# Method 2: HDBSCAN on UMAP-16D (our best AMI)
_cache16 = CKPT / 'umap_16d.npy'
umap_16 = np.load(_cache16)
mcs = max(int(len(umap_16) * 0.01), 10)
hdb16 = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=20,
                         cluster_selection_epsilon=0.1,
                         cluster_selection_method='leaf').fit_predict(umap_16)
# NCA-reassign noise to nearest
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
noise = hdb16 == -1
hdb16_nca = hdb16.copy()
if noise.any() and (~noise).any():
    Xg, yg = umap_16[~noise], hdb16[~noise]
    if len(Xg) > 5000:
        idx = np.random.default_rng(0).choice(len(Xg), 5000, replace=False)
        Xg, yg = Xg[idx], yg[idx]
    pipe = Pipeline([('nca', NeighborhoodComponentsAnalysis(random_state=0)),
                      ('knn', KNeighborsClassifier(30, n_jobs=-1))])
    try:
        pipe.fit(Xg, yg)
        hdb16_nca[noise] = pipe.predict(umap_16[noise])
    except Exception:
        pass

seqs_16 = build_label_sequences(hdb16_nca, sequences_per_file)
pairs = [(s, c) for s, c in zip(seqs_16, contexts_per_seq) if len(s) >= 2]
seqs_16 = [s for s, c in pairs]
ctx_16 = [c for s, c in pairs]
print(f'HDBSCAN-16D+NCA vocab: {len(set(hdb16_nca))}, sequences: {len(seqs_16)}')

# Method 3: DP-GMM on UMAP-2D (vocab 21)
_dp_cache = CKPT / 'dp_mel_umap_k40.npy'
dp21 = np.load(_dp_cache)
seqs_21 = build_label_sequences(dp21, sequences_per_file)
pairs = [(s, c) for s, c in zip(seqs_21, contexts_per_seq) if len(s) >= 2]
seqs_21 = [s for s, c in pairs]
ctx_21 = [c for s, c in pairs]
print(f'DP-GMM-2D vocab: {len(set(dp21))}, sequences: {len(seqs_21)}')


# Run HP1 test for each
rows = []
for method, seqs, ctxs in [
    ('HDBSCAN+NCA baseline (v=6)', seqs_base, ctx_base),
    ('HDBSCAN+NCA UMAP-16D (v=10)', seqs_16, ctx_16),
    ('DP-GMM UMAP-2D (v=21)', seqs_21, ctx_21),
]:
    print(f'\n=== {method} ===')
    print('Fitting Lev-kNN original...')
    f1_orig, s_orig = knn_lev_f1(seqs, ctxs, k=5, n_splits=5, permute=False, seed=0)
    print(f'  F1 orig = {f1_orig:.3f} ± {s_orig:.3f}')
    print('Fitting Lev-kNN permuted...')
    f1_perm, s_perm = knn_lev_f1(seqs, ctxs, k=5, n_splits=5, permute=True, seed=0)
    print(f'  F1 perm = {f1_perm:.3f} ± {s_perm:.3f}')
    delta = f1_orig - f1_perm
    print(f'  Δ = {delta:+.3f}')
    rows.append({
        'method': method,
        'n_seqs': len(seqs),
        'vocab': len(set(t for s in seqs for t in s)),
        'F1_lev_orig': round(f1_orig, 3),
        'F1_lev_orig_std': round(s_orig, 3),
        'F1_lev_perm': round(f1_perm, 3),
        'F1_lev_perm_std': round(s_perm, 3),
        'delta': round(delta, 3),
    })

df = pd.DataFrame(rows)
print('\n=== LEVENSHTEIN-kNN HP1 (Sarkar & Magimai-Doss 2025) ===')
print(df.to_string(index=False))
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/hp1_levenshtein.csv', index=False)
print('\nSaved to docs/thesis/figures/hp1_levenshtein.csv')
