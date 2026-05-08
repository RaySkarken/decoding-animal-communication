"""Apples-to-apples check: baseline Assom-style bag-of-syllables + RF, but with
global clustering performed on UMAP-8D (matching the recommended per-context
configuration's embedding space).

This isolates the gain source: if the per-context UMAP-8D k-means wins not only
over UMAP-2D Assom baseline but also over UMAP-8D global k-means + RF baseline,
then the win comes from per-context decomposition, not from the embedding choice.

Two global-clustering variants are tried:
    - Global k-means (k=15 to match per-context config; optionally k=6 to match
      Assom's vocabulary size)
    - Global HDBSCAN+NCA (mimicking Assom on UMAP-8D)

For each, baseline = bag-of-syllables feature extraction over the global labels
+ RandomForestClassifier, evaluated on 5 emitter-split seeds.

Saves:
    docs/thesis/figures/baseline_umap8d.csv
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings('ignore')

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
N_SEEDS = 5
TEST_EMITTERS_PER_SEED = 11
OUT_DIR = REPO / 'docs' / 'thesis' / 'figures'


def nca_like_relabel(labels: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Mimic Assom's NCA pass: reassign noise points (label −1) to nearest cluster
    center of non-noise points. Returns relabeled array without −1."""
    mask = labels >= 0
    if mask.sum() == 0 or (~mask).sum() == 0:
        return labels.copy()
    centers = []
    ids = sorted(set(labels[mask].tolist()))
    for k in ids:
        centers.append(X[labels == k].mean(axis=0))
    centers = np.stack(centers)
    knn = NearestNeighbors(n_neighbors=1).fit(centers)
    out = labels.copy()
    _, idx = knn.kneighbors(X[~mask])
    for i, pos in zip(np.where(~mask)[0], idx.ravel()):
        out[i] = ids[pos]
    return out


def bag_of_syll(seq, V):
    c = Counter(seq); n = max(len(seq), 1)
    bos = np.zeros(V, dtype=np.float32)
    for k, cnt in c.items():
        if 0 <= k < V: bos[k] = cnt / n
    probs = np.array(list(c.values()), dtype=np.float32) / n
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    rich = len(c) / n; rep = max(c.values()) / n if c else 0.0
    return np.concatenate([bos, [n, rich, ent, rep]]).astype(np.float32)


def emitter_split(all_emitters, seed):
    rng = np.random.default_rng(seed)
    em_arr = np.array(all_emitters); rng.shuffle(em_arr)
    return set(em_arr[:TEST_EMITTERS_PER_SEED].tolist())


def main() -> int:
    if not (CKPT / 'ablation_state.joblib').exists():
        print('ERROR: T7 not mounted', file=sys.stderr); return 2
    st = joblib.load(CKPT / 'ablation_state.joblib')
    seg_df = st['seg_df']
    emb_8d = np.load(CKPT / 'umap_8d.npy')

    # Build vocalizations list
    vocs = []
    for _, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
        seg_ids = g.index.to_list()
        if not seg_ids: continue
        dc = int(np.bincount(g['context'].to_numpy()).argmax())
        if dc not in HP1_CTX: continue
        dem = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
        vocs.append({'seg_ids': seg_ids, 'ctx': dc, 'em': dem})
    all_em = sorted(set(v['em'] for v in vocs))
    print(f'{len(vocs)} vocalizations, {len(all_em)} emitters', flush=True)

    # Pre-compute global clusterings on FULL UMAP-8D data (NOT on train only,
    # because original Assom baseline clusters on full set before per-seed split)
    # Variant A: global k-means k=15
    print('Fitting global k-means (k=15)...', flush=True)
    km15 = KMeans(n_clusters=15, random_state=0, n_init=10).fit(emb_8d)
    labels_km15 = km15.labels_

    # Variant B: global k-means k=6 (matches Assom's vocab size)
    print('Fitting global k-means (k=6)...', flush=True)
    km6 = KMeans(n_clusters=6, random_state=0, n_init=10).fit(emb_8d)
    labels_km6 = km6.labels_

    # Variant C: global HDBSCAN + NCA-like relabel on UMAP-8D
    print('Fitting global HDBSCAN on UMAP-8D...', flush=True)
    hdb = HDBSCAN(min_cluster_size=max(20, int(0.01 * len(emb_8d))),
                   min_samples=20, cluster_selection_epsilon=0.1)
    hdb.fit(emb_8d)
    labels_hdb = nca_like_relabel(hdb.labels_, emb_8d)
    V_hdb = int(labels_hdb.max()) + 1
    print(f'  HDBSCAN found {V_hdb} clusters', flush=True)

    rows = []
    for seed in range(N_SEEDS):
        test_em = emitter_split(all_em, seed)
        tr = [v for v in vocs if v['em'] not in test_em]
        te = [v for v in vocs if v['em'] in test_em]

        for name, labels, V in [
            ('baseline_umap8d_kmeans15', labels_km15, 15),
            ('baseline_umap8d_kmeans6',  labels_km6, 6),
            ('baseline_umap8d_hdbscan',  labels_hdb, V_hdb),
        ]:
            Xt, yt, Xe, ye = [], [], [], []
            for v in tr:
                labs = [int(labels[i]) for i in v['seg_ids'] if labels[i] >= 0]
                if labs: Xt.append(bag_of_syll(labs, V)); yt.append(v['ctx'])
            for v in te:
                labs = [int(labels[i]) for i in v['seg_ids'] if labels[i] >= 0]
                if labs: Xe.append(bag_of_syll(labs, V)); ye.append(v['ctx'])
            rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                         random_state=seed, n_jobs=-1).fit(Xt, yt)
            yp = rf.predict(Xe); yte = np.array(ye)
            w = float(f1_score(yte, yp, average='weighted', labels=HP1_CTX, zero_division=0))
            m = float(f1_score(yte, yp, average='macro', labels=HP1_CTX, zero_division=0))
            mcc = float(matthews_corrcoef(yte, yp)) if len(set(yte)) > 1 else float('nan')
            rows.append({'method': name, 'seed': seed, 'n_test': len(yte),
                         'weighted_f1': w, 'macro_f1': m, 'mcc': mcc, 'V': V})
            print(f'seed={seed} {name:28s} V={V:2d} w={w:.3f} m={m:.3f} mcc={mcc:.3f}', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'baseline_umap8d.csv', index=False)

    print('\n=== 5-SEED AGGREGATES (apples-to-apples baselines on UMAP-8D) ===', flush=True)
    for method, g in df.groupby('method'):
        print(f"  {method:28s}  w={g['weighted_f1'].mean():.3f}±{g['weighted_f1'].std(ddof=1):.3f}   "
              f"m={g['macro_f1'].mean():.3f}±{g['macro_f1'].std(ddof=1):.3f}   "
              f"mcc={g['mcc'].mean():.3f}±{g['mcc'].std(ddof=1):.3f}", flush=True)
    print('DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
