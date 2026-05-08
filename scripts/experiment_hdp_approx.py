"""
HDP-GMM approximation via per-context DP-GMM + cross-context centroid
matching.

This is a pragmatic proxy for proper Hierarchical Dirichlet Process
Gaussian Mixture Model (Teh et al. 2006). Steps:

1. For each context c, fit independent DP-GMM on X[contexts==c]
   → per-context components
2. Collect all components across contexts into a global pool
3. Merge components with similar centroids (greedy matching via
   centroid distance) into "shared base atoms"
4. Track which original context each merged component came from
   → context-specific vs shared classification

End result: labels with an HDP-like interpretation:
  - "shared" label: used by multiple contexts
  - "context-specific" label: used by exactly 1 context

Methodological note: context here acts as GROUP structure (for fitting
per-context mixtures), NOT as supervision target. Tokens themselves
are learned from acoustic features alone; context only defines which
subsets of data to fit DP-GMMs on.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from collections import Counter, defaultdict
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, f1_score
from sklearn.model_selection import StratifiedKFold

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
mel = st['tf_specs'].reshape(-1, 192).astype(np.float32)
emb = st['embedding']
seg_df = st['seg_df']
ctx = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()
proxy = seg_df['proxy_label'].to_numpy() if 'proxy_label' in seg_df.columns else None
mel_pca = PCA(n_components=50, random_state=0).fit_transform(mel)

# Per-file sequences
sequences_per_file, contexts_per_seq = [], []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if len(seg_ids) < 2: continue
    sequences_per_file.append(seg_ids)
    contexts_per_seq.append(int(np.bincount(g['context'].to_numpy()).argmax()))
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]

# Contexts to fit per-context DP-GMM on (skip tiny ones)
CTX_TO_FIT = [c for c in set(ctx) if (ctx == c).sum() >= 100]
print(f'Contexts to fit: {CTX_TO_FIT}, sizes: {[(c, (ctx==c).sum()) for c in CTX_TO_FIT]}')


def ctx_purity(labels, c):
    p, t = 0, 0
    for tid in sorted(set(labels)):
        if tid < 0: continue
        m = np.where(labels == tid)[0]
        if not len(m): continue
        if max(Counter(c[m]).values()) / len(m) >= 0.5: p += len(m)
        t += len(m)
    return p / t if t else 0


def ami(a, b):
    mask = (a >= 0) & (b >= 0)
    if mask.sum() < 2 or len(set(a[mask])) < 2 or len(set(b[mask])) < 2: return np.nan
    return float(adjusted_mutual_info_score(a[mask], b[mask]))


def per_emitter_ami(labels, proxy, em):
    if proxy is None: return np.nan
    vals = []
    for e in set(em):
        mask = (em == e) & (proxy >= 0) & (labels >= 0)
        if mask.sum() < 20: continue
        vals.append(adjusted_mutual_info_score(labels[mask], proxy[mask]))
    return float(np.mean(vals)) if vals else np.nan


def silh(X, labels, n=6000):
    mask = labels >= 0
    n = min(n, int(mask.sum()))
    if n < 100: return np.nan
    lbls = labels[mask][:n]
    if len(set(lbls)) < 2: return np.nan
    return float(silhouette_score(X[mask][:n], lbls, random_state=0))


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
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = levenshtein(sequences[i], sequences[j])
            D[i, j] = d; D[j, i] = d
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    f1s = []
    for tr, te in cv.split(range(n), y):
        preds = []
        for i in te:
            nn = np.argsort(D[i, tr])[:5]
            preds.append(Counter(y[tr][nn]).most_common(1)[0][0])
        f1s.append(f1_score(y[te], preds, average='weighted'))
    return float(np.mean(f1s)), float(np.std(f1s))


def build_seqs(labels):
    return [[int(labels[i]) for i in seg_ids if labels[i] >= 0]
             for seg_ids in sequences_per_file]


def full_eval(labels, X_native, name):
    out = {
        'method': name,
        'vocab': len(set(labels)) - (1 if -1 in labels else 0),
        'silh_native': round(silh(X_native, labels), 3),
        'ctx_purity': round(ctx_purity(labels, ctx), 3),
        'AMI_ctx': round(ami(labels, ctx), 3),
        'AMI_proxy': round(per_emitter_ami(labels, proxy, emitters), 3),
    }
    seqs = build_seqs(labels)
    pairs = [(s, c) for s, c in zip(seqs, contexts_per_seq) if len(s) >= 2 and c in HP1_CTX]
    if pairs:
        f1_o, _ = knn_lev_f1([s for s,c in pairs], [c for s,c in pairs], permute=False)
        f1_p, _ = knn_lev_f1([s for s,c in pairs], [c for s,c in pairs], permute=True)
        out['F1_lev'] = round(f1_o, 3)
        out['Delta_HP1'] = round(f1_o - f1_p, 3)
    return out


# === HDP-APPROX: per-context DP-GMM + cross-context merging ====

def hdp_approx(X, ctx, space_name='mel_pca', k_per_context=10,
                merge_threshold_quantile=0.15, min_component_size=20):
    """Per-context DP-GMM then merge similar centroids across contexts."""
    all_components = []   # list of dicts: {ctx_of_origin, centroid, member_indices}
    for c in CTX_TO_FIT:
        mask = ctx == c
        X_c = X[mask]
        global_idx = np.where(mask)[0]
        print(f'  Fitting DP-GMM on context {c} (n={len(X_c)}) ...', end='', flush=True)
        t0 = time.time()
        bgm = BayesianGaussianMixture(
            n_components=k_per_context,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1, covariance_type='full',
            max_iter=100, random_state=0,
        )
        try:
            sub_labels = bgm.fit_predict(X_c)
        except Exception as e:
            print(f'  FAILED ({e})')
            continue
        for k in set(sub_labels):
            members_local = np.where(sub_labels == k)[0]
            if len(members_local) < min_component_size:
                continue
            members_global = global_idx[members_local]
            centroid = X_c[members_local].mean(axis=0)
            all_components.append({
                'ctx': int(c),
                'centroid': centroid,
                'members': members_global,
                'size': len(members_global),
            })
        print(f' {time.time()-t0:.0f}s, {len([c for c in all_components if c["ctx"]==c])} atoms per context')

    # All components collected; now merge across contexts
    print(f'Collected {len(all_components)} per-context components')
    centroids = np.array([a['centroid'] for a in all_components])

    # Compute pairwise distances between components
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(centroids))
    np.fill_diagonal(D, np.inf)

    # Threshold: merge pairs within merge_threshold_quantile of all distances
    finite_D = D[D < np.inf]
    if len(finite_D) == 0:
        threshold = 0
    else:
        threshold = float(np.quantile(finite_D, merge_threshold_quantile))
    print(f'Merge threshold (q={merge_threshold_quantile}): {threshold:.3f}')

    # Union-find for merging
    parent = list(range(len(all_components)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry

    # Merge pairs under threshold (greedy from closest)
    pairs = []
    for i in range(len(all_components)):
        for j in range(i+1, len(all_components)):
            if D[i, j] < threshold:
                pairs.append((D[i, j], i, j))
    pairs.sort()
    for d, i, j in pairs:
        union(i, j)

    # Group components by merged super-cluster
    group_ids = {}
    next_id = 0
    for i in range(len(all_components)):
        r = find(i)
        if r not in group_ids:
            group_ids[r] = next_id
            next_id += 1

    # Assign final labels: each segment gets super-cluster id of its
    # original per-context component
    labels = -np.ones(len(X), dtype=int)
    for a_idx, a in enumerate(all_components):
        super_id = group_ids[find(a_idx)]
        labels[a['members']] = super_id

    # Characterise each super-cluster: context-specific or shared?
    super_info = defaultdict(lambda: {'contexts': set(), 'size': 0})
    for a_idx, a in enumerate(all_components):
        sid = group_ids[find(a_idx)]
        super_info[sid]['contexts'].add(a['ctx'])
        super_info[sid]['size'] += a['size']

    print(f'\nHDP-approx result: {len(super_info)} super-clusters')
    n_specific = sum(1 for v in super_info.values() if len(v['contexts']) == 1)
    n_shared = sum(1 for v in super_info.values() if len(v['contexts']) > 1)
    print(f'  Context-specific (1 ctx):  {n_specific}')
    print(f'  Shared (≥2 ctx):           {n_shared}')
    max_sharing = max(len(v['contexts']) for v in super_info.values())
    print(f'  Most-shared cluster spans  {max_sharing} contexts')

    return labels, super_info


# ─── Run HDP-approx on mel-PCA50 and on UMAP-2D ────────────────────────
rows = []

for X_name, X in [('mel_PCA50', mel_pca), ('UMAP-2D', emb)]:
    print(f'\n=== HDP-approx on {X_name} ===')
    for k_per in [10, 20]:
        for merge_q in [0.05, 0.10, 0.15]:
            print(f'\nConfig: k_per_context={k_per}, merge_q={merge_q}')
            labels, info = hdp_approx(X, ctx, space_name=X_name,
                                         k_per_context=k_per,
                                         merge_threshold_quantile=merge_q)
            r = full_eval(labels, X, f'HDP-approx {X_name} k={k_per} q={merge_q}')
            r['shared'] = sum(1 for v in info.values() if len(v['contexts']) > 1)
            r['specific'] = sum(1 for v in info.values() if len(v['contexts']) == 1)
            rows.append(r)

df = pd.DataFrame(rows)
print('\n\n=== HDP-APPROX SUMMARY ===')
print(df.to_string(index=False))
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/hdp_approx.csv', index=False)
print('\nSaved to docs/thesis/figures/hdp_approx.csv')
