"""
HDP-approx held-out robustness: multi-seed train/test splits.

Re-runs the held-out generalization test with several different
emitter train/test splits to check whether the negative result
(HDP-approx test advantage ≈ 0) is robust or split-dependent.

For each seed:
  - Random split 30 train / 11 test emitters
  - Fit HDP-approx on train only, predict test via 1-NN
  - Refit HDBSCAN+NCA on train only, predict test via 1-NN
  - Record AMI_ctx TRAIN vs TEST for both methods

If HDP-approx test advantage is consistently within ±0.02 AMI across
all seeds → negative result is robust (publishable).
If gain varies wildly across seeds → noise floor of the experiment.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_mutual_info_score
from scipy.spatial.distance import pdist, squareform

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
emb = st['embedding']
seg_df = st['seg_df']
ctx = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()

all_emitters = sorted(set(emitters))
emitter_sizes = {e: int((emitters == e).sum()) for e in all_emitters}


def ami(a, b):
    mask = (a >= 0) & (b >= 0)
    if mask.sum() < 2 or len(set(a[mask])) < 2 or len(set(b[mask])) < 2:
        return float('nan')
    return float(adjusted_mutual_info_score(a[mask], b[mask]))


def ctx_purity(labels, c, mask):
    p, t = 0, 0
    for tid in sorted(set(labels[mask])):
        if tid < 0: continue
        m = np.where(mask & (labels == tid))[0]
        if not len(m): continue
        if max(Counter(c[m]).values()) / len(m) >= 0.5: p += len(m)
        t += len(m)
    return p / t if t else 0.0


def run_one_split(seed):
    rng = np.random.default_rng(seed)
    em_arr = np.array(all_emitters)
    rng.shuffle(em_arr)
    test_emitters = set(em_arr[:11].tolist())
    train_emitters = set(em_arr[11:].tolist())

    train_seg_mask = np.isin(emitters, list(train_emitters))
    test_seg_mask = np.isin(emitters, list(test_emitters))
    train_ctx_set = set(ctx[train_seg_mask])
    ctx_to_fit = [c for c in train_ctx_set
                  if (train_seg_mask & (ctx == c)).sum() >= 100]

    # ── HDP-approx fit on train ──────────────────────────────────────────
    all_components = []
    for c in ctx_to_fit:
        mask = train_seg_mask & (ctx == c)
        global_idx = np.where(mask)[0]
        X_c = emb[mask]
        bgm = BayesianGaussianMixture(
            n_components=10,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1, covariance_type='full',
            max_iter=100, random_state=0,
        )
        sub = bgm.fit_predict(X_c)
        for k in set(sub):
            mem = np.where(sub == k)[0]
            if len(mem) < 20: continue
            all_components.append({
                'ctx': int(c),
                'centroid': X_c[mem].mean(axis=0),
                'members': global_idx[mem],
            })
    if not all_components:
        return None

    centroids = np.array([a['centroid'] for a in all_components])
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
    pairs = [(D[i,j], i, j)
             for i in range(len(all_components))
             for j in range(i+1, len(all_components))
             if D[i,j] < threshold]
    pairs.sort()
    for _, i, j in pairs: union(i, j)

    group_ids = {}
    for i in range(len(all_components)):
        r = find(i)
        if r not in group_ids: group_ids[r] = len(group_ids)
    train_labels = -np.ones(len(emb), dtype=int)
    for a_idx, a in enumerate(all_components):
        train_labels[a['members']] = group_ids[find(a_idx)]

    sc_ids = sorted(set(group_ids.values()))
    sc_arr = np.stack([emb[train_labels == s].mean(axis=0)
                       for s in sc_ids
                       if (train_labels == s).sum() > 0])
    knn = NearestNeighbors(n_neighbors=1).fit(sc_arr)
    # Orphan train segments
    orphan = train_seg_mask & (train_labels == -1)
    if orphan.any():
        _, idx = knn.kneighbors(emb[orphan])
        train_labels[orphan] = np.array(sc_ids)[idx.ravel()]
    # Test predictions
    _, idx = knn.kneighbors(emb[test_seg_mask])
    test_preds = np.array(sc_ids)[idx.ravel()]
    hdp_labels = train_labels.copy()
    hdp_labels[test_seg_mask] = test_preds

    # ── HDBSCAN+NCA baseline refit on train ──────────────────────────────
    import hdbscan
    from sklearn.pipeline import Pipeline
    from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
    mcs = max(int(train_seg_mask.sum() * 0.01), 10)
    hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=20,
                           cluster_selection_epsilon=0.1,
                           cluster_selection_method='leaf').fit_predict(emb[train_seg_mask])
    noise = hdb == -1
    hdb_nca = hdb.copy()
    if noise.any() and (~noise).any():
        Xg, yg = emb[train_seg_mask][~noise], hdb[~noise]
        if len(Xg) > 5000:
            ri = np.random.default_rng(0).choice(len(Xg), 5000, replace=False)
            Xg, yg = Xg[ri], yg[ri]
        try:
            pipe = Pipeline([
                ('nca', NeighborhoodComponentsAnalysis(random_state=0)),
                ('knn', KNeighborsClassifier(30, n_jobs=-1))])
            pipe.fit(Xg, yg)
            hdb_nca[noise] = pipe.predict(emb[train_seg_mask][noise])
        except Exception:
            pass
    base_labels = -np.ones(len(emb), dtype=int)
    base_labels[train_seg_mask] = hdb_nca
    base_sc_ids = sorted(set(hdb_nca))
    base_sc_arr = np.stack([emb[train_seg_mask][hdb_nca == s].mean(axis=0)
                             for s in base_sc_ids])
    bknn = NearestNeighbors(n_neighbors=1).fit(base_sc_arr)
    _, bidx = bknn.kneighbors(emb[test_seg_mask])
    base_labels[test_seg_mask] = np.array(base_sc_ids)[bidx.ravel()]

    # ── Metrics ──────────────────────────────────────────────────────────
    return {
        'seed': seed,
        'n_test': int(test_seg_mask.sum()),
        'hdp_vocab': len(sc_ids),
        'base_vocab': len(base_sc_ids),
        'hdp_ami_train': round(ami(hdp_labels[train_seg_mask], ctx[train_seg_mask]), 3),
        'hdp_ami_test':  round(ami(hdp_labels[test_seg_mask],  ctx[test_seg_mask]), 3),
        'base_ami_train': round(ami(base_labels[train_seg_mask], ctx[train_seg_mask]), 3),
        'base_ami_test':  round(ami(base_labels[test_seg_mask],  ctx[test_seg_mask]), 3),
        'hdp_cp_test':   round(ctx_purity(hdp_labels, ctx, test_seg_mask), 3),
        'base_cp_test':  round(ctx_purity(base_labels, ctx, test_seg_mask), 3),
    }


rows = []
for seed in [0, 1, 2, 3, 4]:
    print(f'Running seed {seed}...', flush=True)
    r = run_one_split(seed)
    if r is None:
        print(f'  seed {seed} FAILED')
        continue
    r['ami_gain_test'] = round(r['hdp_ami_test'] - r['base_ami_test'], 3)
    r['hdp_drop'] = round(r['hdp_ami_train'] - r['hdp_ami_test'], 3)
    r['base_drop'] = round(r['base_ami_train'] - r['base_ami_test'], 3)
    rows.append(r)
    print(f"  seed {seed}: HDP test {r['hdp_ami_test']} vs base test {r['base_ami_test']} (gain {r['ami_gain_test']:+.3f})")

df = pd.DataFrame(rows)
print('\n=== MULTI-SEED HELD-OUT SUMMARY ===')
print(df.to_string(index=False))

print('\n=== AGGREGATE ===')
print(f'HDP test AMI:       {df["hdp_ami_test"].mean():.3f} ± {df["hdp_ami_test"].std():.3f}')
print(f'Base test AMI:      {df["base_ami_test"].mean():.3f} ± {df["base_ami_test"].std():.3f}')
print(f'Gain (HDP – base):  {df["ami_gain_test"].mean():+.3f} ± {df["ami_gain_test"].std():.3f}')
print(f'HDP TRAIN→TEST drop: {df["hdp_drop"].mean():+.3f} ± {df["hdp_drop"].std():.3f}')
print(f'Base TRAIN→TEST drop: {df["base_drop"].mean():+.3f} ± {df["base_drop"].std():.3f}')

Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/hdp_heldout_multiseed.csv', index=False)
print('\nSaved to docs/thesis/figures/hdp_heldout_multiseed.csv')

mean_gain = df['ami_gain_test'].mean()
std_gain = df['ami_gain_test'].std()
if abs(mean_gain) <= 2 * std_gain:
    print(f'\n=> Gain 95%-CI crosses zero ({mean_gain:+.3f} ± {2*std_gain:.3f}).')
    print('   Negative result is robust: HDP-approx does NOT generalize.')
else:
    print(f'\n=> Gain {mean_gain:+.3f} ± {2*std_gain:.3f} is significant.')
