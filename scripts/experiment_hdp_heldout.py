"""
HDP-approx: held-out generalization test.

Split 41 emitters into train/test (stratified by having context coverage).
Fit HDP-approx only on TRAIN segments. At inference, assign TEST segments
to nearest super-centroid via 1-NN. Evaluate on test set.

Compares:
  - HDP-approx TRAIN evaluation (in-sample)  — should be high (overfit)
  - HDP-approx TEST evaluation  (out-of-sample) — the honest number
  - Baseline HDBSCAN TEST evaluation — fair comparator
  - Pure HDBSCAN refit on TRAIN then predict TEST — another fair baseline

If HDP-approx test_AMI >> baseline test_AMI → real generalization
If HDP-approx test_AMI ≈ baseline test_AMI → in-sample gain was overfitting
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from collections import Counter, defaultdict
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import pdist, squareform

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
mel = st['tf_specs'].reshape(-1, 192).astype(np.float32)
emb = st['embedding']
seg_df = st['seg_df']
ctx = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()
hdbnca = st['hdb_nca_labels']
proxy = seg_df['proxy_label'].to_numpy() if 'proxy_label' in seg_df.columns else None

# Per-file sequences
sequences_per_file, contexts_per_seq, emitters_per_seq = [], [], []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if len(seg_ids) < 2: continue
    sequences_per_file.append(seg_ids)
    contexts_per_seq.append(int(np.bincount(g['context'].to_numpy()).argmax()))
    emitters_per_seq.append(int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0]))
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]

# ─── 1. Split emitters into train/test ─────────────────────────────────
all_emitters = sorted(set(emitters))
print(f'Total emitters: {len(all_emitters)}')

# Pick 11 test emitters, stratified on emitter size (ensure both train/test have big and small bats)
emitter_sizes = {e: (emitters == e).sum() for e in all_emitters}
sorted_by_size = sorted(all_emitters, key=lambda e: -emitter_sizes[e])
# Round-robin: every 4th emitter → test
test_emitters = set(sorted_by_size[::4][:11])
train_emitters = set(all_emitters) - test_emitters
print(f'Train emitters: {len(train_emitters)} ({sum(emitter_sizes[e] for e in train_emitters)} segs)')
print(f'Test emitters:  {len(test_emitters)} ({sum(emitter_sizes[e] for e in test_emitters)} segs)')
print(f'Test emitters: {sorted(test_emitters)}')

# Create masks
train_seg_mask = np.isin(emitters, list(train_emitters))
test_seg_mask = np.isin(emitters, list(test_emitters))
print(f'Train segments: {train_seg_mask.sum()}, Test: {test_seg_mask.sum()}')

# Verify context coverage
train_ctx_set = set(ctx[train_seg_mask])
test_ctx_set = set(ctx[test_seg_mask])
print(f'Train contexts: {sorted(train_ctx_set)}')
print(f'Test contexts: {sorted(test_ctx_set)}')

# Per-sequence split
train_seq_mask = np.array([e in train_emitters for e in emitters_per_seq])
test_seq_mask = np.array([e in test_emitters for e in emitters_per_seq])


# ─── 2. Fit HDP-approx on TRAIN only ────────────────────────────────────
CTX_TO_FIT = [c for c in train_ctx_set if (train_seg_mask & (ctx == c)).sum() >= 100]
print(f'\nFitting per-context DP-GMMs (TRAIN ONLY) on contexts: {CTX_TO_FIT}')

all_components = []
for c in CTX_TO_FIT:
    mask = train_seg_mask & (ctx == c)
    global_idx = np.where(mask)[0]
    X_c = emb[mask]
    bgm = BayesianGaussianMixture(
        n_components=10,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1, covariance_type='full',
        max_iter=100, random_state=0,
    )
    sub_labels = bgm.fit_predict(X_c)
    for k in set(sub_labels):
        members_local = np.where(sub_labels == k)[0]
        if len(members_local) < 20: continue
        all_components.append({
            'ctx': int(c),
            'centroid': X_c[members_local].mean(axis=0),
            'members': global_idx[members_local],
            'size': len(members_local),
        })

centroids = np.array([a['centroid'] for a in all_components])
print(f'Total per-context components (train): {len(all_components)}')

# Merge with q=0.05
D = squareform(pdist(centroids))
np.fill_diagonal(D, np.inf)
threshold = float(np.quantile(D[D < np.inf], 0.05))
parent = list(range(len(all_components)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry: parent[rx] = ry
pairs = [(D[i,j], i, j) for i in range(len(all_components))
         for j in range(i+1, len(all_components)) if D[i,j] < threshold]
pairs.sort()
for d, i, j in pairs: union(i, j)

group_ids = {}
for i in range(len(all_components)):
    r = find(i)
    if r not in group_ids: group_ids[r] = len(group_ids)

# TRAIN labels
train_labels = -np.ones(len(mel), dtype=int)
for a_idx, a in enumerate(all_components):
    sid = group_ids[find(a_idx)]
    train_labels[a['members']] = sid
# Reassign any unassigned TRAIN segs (segments not covered by any per-ctx DP-GMM) via 1-NN
super_ids = sorted(group_ids.values())
super_centroids = {sid: emb[train_labels == sid].mean(axis=0) for sid in super_ids
                   if (train_labels == sid).sum() > 0}
if super_centroids:
    sc_ids = sorted(super_centroids.keys())
    sc_arr = np.stack([super_centroids[s] for s in sc_ids])
    knn = NearestNeighbors(n_neighbors=1).fit(sc_arr)
    orphan_train = train_seg_mask & (train_labels == -1)
    if orphan_train.any():
        _, idx = knn.kneighbors(emb[orphan_train])
        train_labels[orphan_train] = np.array(sc_ids)[idx.ravel()]

print(f'HDP-approx super-clusters (train): {len(super_ids)}')
n_specific = sum(1 for s in super_ids if len(set([a['ctx'] for a_idx, a in enumerate(all_components) if group_ids[find(a_idx)] == s])) == 1)
n_shared = len(super_ids) - n_specific
print(f'  context-specific: {n_specific}, shared: {n_shared}')


# ─── 3. Inference on TEST segments via 1-NN to super-centroid ─────────
# Recompute super-centroids with only train members
sc_arr = np.stack([emb[train_labels == s].mean(axis=0) for s in sc_ids])
knn = NearestNeighbors(n_neighbors=1).fit(sc_arr)
_, idx = knn.kneighbors(emb[test_seg_mask])
test_preds = np.array(sc_ids)[idx.ravel()]

# Full labels (train preds + test preds)
all_labels = train_labels.copy()
all_labels[test_seg_mask] = test_preds

print(f'Test predictions done. Unique tokens used: {len(set(test_preds))}')


# ─── 4. Baseline: HDBSCAN + NCA refit on TRAIN only, predict TEST via 1-NN ──
import hdbscan
mcs = max(int(train_seg_mask.sum() * 0.01), 10)
hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=20,
                        cluster_selection_epsilon=0.1,
                        cluster_selection_method='leaf').fit_predict(emb[train_seg_mask])
# Reassign noise via NCA+KNN (only on train)
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
noise_mask = hdb == -1
hdb_nca_tr = hdb.copy()
if noise_mask.any() and (~noise_mask).any():
    Xg, yg = emb[train_seg_mask][~noise_mask], hdb[~noise_mask]
    if len(Xg) > 5000:
        ri = np.random.default_rng(0).choice(len(Xg), 5000, replace=False)
        Xg, yg = Xg[ri], yg[ri]
    pipe = Pipeline([('nca', NeighborhoodComponentsAnalysis(random_state=0)),
                      ('knn', KNeighborsClassifier(30, n_jobs=-1))])
    try:
        pipe.fit(Xg, yg)
        hdb_nca_tr[noise_mask] = pipe.predict(emb[train_seg_mask][noise_mask])
    except Exception:
        pass

# TRAIN baseline labels (full array, -1 for test)
baseline_labels = -np.ones(len(mel), dtype=int)
baseline_labels[train_seg_mask] = hdb_nca_tr

# Baseline TEST preds via 1-NN to baseline centroids
base_sc_ids = sorted(set(hdb_nca_tr))
base_sc_arr = np.stack([emb[train_seg_mask][hdb_nca_tr == s].mean(axis=0) for s in base_sc_ids])
base_knn = NearestNeighbors(n_neighbors=1).fit(base_sc_arr)
_, bidx = base_knn.kneighbors(emb[test_seg_mask])
baseline_test_preds = np.array(base_sc_ids)[bidx.ravel()]
baseline_labels[test_seg_mask] = baseline_test_preds
print(f'Baseline HDBSCAN+NCA (train-only refit): {len(base_sc_ids)} clusters')


# ─── 5. Evaluate ───────────────────────────────────────────────────────
def ctx_purity(labels, c, mask):
    p, t = 0, 0
    for tid in sorted(set(labels[mask])):
        if tid < 0: continue
        m = np.where(mask & (labels == tid))[0]
        if not len(m): continue
        if max(Counter(c[m]).values()) / len(m) >= 0.5: p += len(m)
        t += len(m)
    return p / t if t else 0

def ami(a, b):
    mask = (a >= 0) & (b >= 0)
    if mask.sum() < 2 or len(set(a[mask])) < 2 or len(set(b[mask])) < 2: return np.nan
    return float(adjusted_mutual_info_score(a[mask], b[mask]))

def levenshtein(a, b):
    if len(a) < len(b): a, b = b, a
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0]*len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(cur[j-1]+1, prev[j]+1, prev[j-1]+(ca!=cb))
        prev = cur
    return prev[-1]

def knn_lev_f1(sequences, contexts, subsample=1500, permute=False, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(contexts)
    if len(sequences) > subsample:
        idx = rng.choice(len(sequences), size=subsample, replace=False)
        sequences = [sequences[i] for i in idx]
        y = y[idx]
    if permute:
        sequences = [list(rng.permutation(s)) for s in sequences]
    n = len(sequences)
    D_ = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = levenshtein(sequences[i], sequences[j])
            D_[i, j] = d; D_[j, i] = d
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    f1s = []
    for tr, te in cv.split(range(n), y):
        preds = []
        for i in te:
            nn = np.argsort(D_[i, tr])[:5]
            preds.append(Counter(y[tr][nn]).most_common(1)[0][0])
        f1s.append(f1_score(y[te], preds, average='weighted'))
    return float(np.mean(f1s))

def build_seqs(labels, mask_per_seq):
    out = []
    for seg_ids, ok in zip(sequences_per_file, mask_per_seq):
        if not ok: continue
        s = [int(labels[i]) for i in seg_ids if labels[i] >= 0]
        if len(s) >= 2: out.append(s)
    return out


def evaluate_split(labels, mask, label, run_hp1=True):
    ami_ctx = ami(labels[mask], ctx[mask])
    cp = ctx_purity(labels, ctx, mask)
    out = {'split_label': label, 'n_seg': int(mask.sum()),
           'vocab': len(set(labels[mask][labels[mask] >= 0])),
           'AMI_ctx': round(ami_ctx, 3), 'ctx_purity': round(cp, 3)}
    # Proxy AMI (per-emitter)
    if proxy is not None:
        vals = []
        for em in set(emitters[mask]):
            m = mask & (emitters == em) & (proxy >= 0) & (labels >= 0)
            if m.sum() < 20: continue
            vals.append(adjusted_mutual_info_score(labels[m], proxy[m]))
        out['AMI_proxy'] = round(float(np.mean(vals)), 3) if vals else float('nan')
    if run_hp1:
        if label.startswith('TRAIN'): seq_mask = train_seq_mask
        elif label.startswith('TEST'): seq_mask = test_seq_mask
        else: seq_mask = np.ones(len(sequences_per_file), bool)
        seqs = build_seqs(labels, seq_mask)
        ctxs = [c for s, c, ok in zip(sequences_per_file, contexts_per_seq, seq_mask)
                if ok and len([i for i in s if labels[i] >= 0]) >= 2 and c in HP1_CTX]
        seqs_hp1 = [s for s in seqs if len(s) >= 2][:len(ctxs)]
        if len(seqs_hp1) >= 50:
            f1_o = knn_lev_f1(seqs_hp1, ctxs, permute=False)
            f1_p = knn_lev_f1(seqs_hp1, ctxs, permute=True)
            out['F1_lev'] = round(f1_o, 3)
            out['F1_lev_perm'] = round(f1_p, 3)
            out['Delta_HP1'] = round(f1_o - f1_p, 3)
    return out


rows = []
print('\n=== EVALUATION ===')
print('\nHDP-approx:')
rows.append(evaluate_split(all_labels, train_seg_mask, 'TRAIN HDP-approx'))
rows.append(evaluate_split(all_labels, test_seg_mask, 'TEST HDP-approx'))

print('Baseline HDBSCAN (refit on train, predict test):')
rows.append(evaluate_split(baseline_labels, train_seg_mask, 'TRAIN HDBSCAN-baseline'))
rows.append(evaluate_split(baseline_labels, test_seg_mask, 'TEST HDBSCAN-baseline'))

# Also: original Assom labels (fit on ALL data) evaluated on test subset — reference
print('Original Assom baseline (fit on ALL, evaluated on test subset):')
rows.append(evaluate_split(hdbnca, test_seg_mask, 'TEST Assom (fit-on-all)'))

df = pd.DataFrame(rows)
print('\n=== HELD-OUT SUMMARY ===')
print(df.to_string(index=False))
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/hdp_heldout.csv', index=False)
print('\nSaved to docs/thesis/figures/hdp_heldout.csv')

# Interpretation
print('\n\n=== INTERPRETATION ===')
train_hdp = df[df['split_label'] == 'TRAIN HDP-approx'].iloc[0]
test_hdp = df[df['split_label'] == 'TEST HDP-approx'].iloc[0]
test_base = df[df['split_label'] == 'TEST HDBSCAN-baseline'].iloc[0]

drop = train_hdp['AMI_ctx'] - test_hdp['AMI_ctx']
gain = test_hdp['AMI_ctx'] - test_base['AMI_ctx']
print(f'  HDP-approx train AMI_ctx: {train_hdp["AMI_ctx"]}')
print(f'  HDP-approx TEST AMI_ctx:  {test_hdp["AMI_ctx"]}  (drop from train: {drop:+.3f})')
print(f'  Baseline TEST AMI_ctx:    {test_base["AMI_ctx"]}')
print(f'  HDP-approx advantage on TEST: {gain:+.3f} AMI')
if gain > 0.05:
    print('  → HDP-approx GENERALIZES (real improvement out-of-sample)')
elif gain > 0:
    print('  → HDP-approx marginal advantage, likely within noise')
else:
    print('  → HDP-approx advantage was IN-SAMPLE ONLY — not generalizable')
