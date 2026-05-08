"""
Visualize the HDP-approx best config (UMAP-2D, k=10 per-context, q=0.05).

Generates:
  1. UMAP scatter colored by HDP super-cluster labels
  2. UMAP scatter colored by behavioural context (reference)
  3. Spectrogram gallery per super-cluster, annotated with
     shared-vs-context-specific label
  4. Context distribution bar chart per super-cluster
  5. Silhouette scores on both UMAP-2D and mel-PCA50
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
FIG = Path('docs/thesis/figures/hdp_approx')
FIG.mkdir(parents=True, exist_ok=True)

st = joblib.load(CKPT / 'ablation_state.joblib')
tf_specs = st['tf_specs']           # (N, 6, 32)
mel = tf_specs.reshape(len(tf_specs), -1).astype(np.float32)
emb = st['embedding']                # UMAP 2D
ctx = st['seg_df']['context'].to_numpy()

CTX_MAP = {0:'Unknown', 1:'Separation', 2:'Biting', 3:'Feeding',
           4:'Fighting', 5:'Grooming', 6:'Isolation', 7:'Kissing',
           8:'Landing', 9:'MatingPrt', 10:'ThreatLk', 11:'General', 12:'Sleep'}
CTX_SHORT = {k: v[:3] for k, v in CTX_MAP.items()}

# Fit HDP-approx best config
CTX_TO_FIT = [c for c in set(ctx) if (ctx == c).sum() >= 100]
print(f'Fitting per-context DP-GMMs on contexts: {CTX_TO_FIT}')

all_components = []
for c in CTX_TO_FIT:
    mask = ctx == c
    global_idx = np.where(mask)[0]
    bgm = BayesianGaussianMixture(
        n_components=10,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1, covariance_type='full',
        max_iter=100, random_state=0,
    )
    sub_labels = bgm.fit_predict(emb[mask])
    for k in set(sub_labels):
        members_local = np.where(sub_labels == k)[0]
        if len(members_local) < 20: continue
        members_global = global_idx[members_local]
        centroid = emb[mask][members_local].mean(axis=0)
        all_components.append({
            'ctx': int(c), 'centroid': centroid,
            'members': members_global, 'size': len(members_global),
        })

centroids = np.array([a['centroid'] for a in all_components])
from scipy.spatial.distance import pdist, squareform
D = squareform(pdist(centroids))
np.fill_diagonal(D, np.inf)
finite = D[D < np.inf]
threshold = float(np.quantile(finite, 0.05))
print(f'Merge threshold: {threshold:.3f}')

parent = list(range(len(all_components)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry: parent[rx] = ry

# Merge ALL pairs (including within-context) if centroid distance is small.
# Within-context over-splitting happens when DP-GMM finds multiple components
# on one context that are near-identical acoustically.
pairs = [(D[i,j], i, j) for i in range(len(all_components))
         for j in range(i+1, len(all_components)) if D[i,j] < threshold]
pairs.sort()
for d, i, j in pairs: union(i, j)

group_ids = {}
for i in range(len(all_components)):
    r = find(i)
    if r not in group_ids:
        group_ids[r] = len(group_ids)

# Assign super-cluster labels + compute per-super metadata
labels = -np.ones(len(mel), dtype=int)
super_info = defaultdict(lambda: {'contexts': [], 'size': 0})
for a_idx, a in enumerate(all_components):
    sid = group_ids[find(a_idx)]
    labels[a['members']] = sid
    super_info[sid]['contexts'].append(a['ctx'])
    super_info[sid]['size'] += a['size']

# Reassign -1 (segments from tiny contexts like Landing) to nearest super-centroid
unassigned = labels == -1
if unassigned.any():
    super_ids = sorted(super_info.keys())
    super_centroids = np.stack([
        emb[labels == sid].mean(axis=0) for sid in super_ids
    ])
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=1).fit(super_centroids)
    _, idx = knn.kneighbors(emb[unassigned])
    labels[unassigned] = np.array(super_ids)[idx.ravel()]
    print(f'Reassigned {unassigned.sum()} unassigned segments to nearest super-cluster')

# Characterise each super-cluster
super_summary = {}
for sid, info in super_info.items():
    ctx_set = set(info['contexts'])
    members = np.where(labels == sid)[0]
    if len(members):
        top_ctx = Counter(ctx[members]).most_common(3)
    else:
        top_ctx = []
    super_summary[sid] = {
        'size': len(members),
        'origin_contexts': sorted(ctx_set),
        'n_origin': len(ctx_set),
        'kind': 'context-specific' if len(ctx_set) == 1 else 'shared',
        'top_actual_contexts': top_ctx,
    }

print(f'\nHDP-approx: {len(super_summary)} super-clusters')
for sid, s in sorted(super_summary.items()):
    kind = s['kind']
    origin = [CTX_SHORT.get(c, str(c)) for c in s['origin_contexts']]
    top = ', '.join(f"{CTX_SHORT.get(int(c), str(c))}:{n/s['size']*100:.0f}%" for c, n in s['top_actual_contexts'])
    print(f'  tok {sid:2d} | {kind:15s} | origin={origin}  size={s["size"]}  actual: {top}')

# === Silhouette ===
mask = labels >= 0
n = min(8000, mask.sum())
lbl_s = labels[mask][:n]

sil_umap = silhouette_score(emb[mask][:n], lbl_s, random_state=0)
mel_pca = PCA(n_components=50, random_state=0).fit_transform(mel)
sil_mel = silhouette_score(mel_pca[mask][:n], lbl_s, random_state=0)
sil_mel_raw = silhouette_score(mel[mask][:n], lbl_s, random_state=0)

print(f'\n=== SILHOUETTE SCORES ===')
print(f'  silhouette on UMAP-2D (fit space): {sil_umap:.3f}')
print(f'  silhouette on Mel-PCA50:           {sil_mel:.3f}')
print(f'  silhouette on raw Mel-192D:        {sil_mel_raw:.3f}')

# Baseline HDBSCAN for comparison
hdb = st['hdb_nca_labels']
mask_b = hdb >= 0
n_b = min(8000, mask_b.sum())
sil_umap_b = silhouette_score(emb[mask_b][:n_b], hdb[mask_b][:n_b], random_state=0)
sil_mel_b = silhouette_score(mel_pca[mask_b][:n_b], hdb[mask_b][:n_b], random_state=0)
print(f'\n  Baseline HDBSCAN+NCA for comparison:')
print(f'    silh UMAP-2D:   {sil_umap_b:.3f}')
print(f'    silh Mel-PCA50: {sil_mel_b:.3f}')


# === VISUALIZATIONS ===

# Fig 1: UMAP scatter — cluster vs context
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
pal_lbl = sns.color_palette('husl', len(super_summary))
color_map_lbl = dict(zip(sorted(super_summary.keys()), pal_lbl))
colors_lbl = [color_map_lbl[l] for l in labels]
axes[0].scatter(emb[:, 0], emb[:, 1], c=colors_lbl, s=0.5, alpha=0.6)
# Legend for super-clusters
for sid, info in sorted(super_summary.items()):
    kind_marker = '*' if info['kind'] == 'shared' else 'o'
    origin = [CTX_SHORT.get(c, str(c)) for c in info['origin_contexts']]
    lbl_txt = f"tok {sid} ({info['kind'][:5]}, n={info['size']})"
    axes[0].scatter([], [], c=[color_map_lbl[sid]], marker=kind_marker,
                    label=lbl_txt, s=50)
axes[0].legend(fontsize=6, markerscale=1, loc='center left',
               bbox_to_anchor=(1.01, 0.5), framealpha=0.9, ncol=1)
axes[0].set_title(f'HDP-approx super-clusters ({len(super_summary)} atoms)\n'
                  f'silh_UMAP={sil_umap:.3f}, silh_Mel-PCA50={sil_mel:.3f}')
axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')

pal_ctx = sns.color_palette('husl', len(set(ctx)))
color_map_ctx = dict(zip(sorted(set(ctx)), pal_ctx))
colors_ctx = [color_map_ctx[c] for c in ctx]
axes[1].scatter(emb[:, 0], emb[:, 1], c=colors_ctx, s=0.5, alpha=0.6)
for c in sorted(set(ctx)):
    axes[1].scatter([], [], c=[color_map_ctx[c]],
                    label=CTX_MAP.get(int(c), str(c)), s=30)
axes[1].legend(fontsize=8, markerscale=1, loc='best')
axes[1].set_title('Behavioural context (reference)')
axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')
plt.tight_layout()
plt.savefig(FIG / 'umap_hdp_vs_context.png', dpi=140, bbox_inches='tight')
plt.close()
print(f'Saved {FIG}/umap_hdp_vs_context.png')

# Fig 2: Mel-spec gallery per super-cluster
uniq = sorted(super_summary.keys())
n_rows = len(uniq)
ncols = 8
fig, axes = plt.subplots(n_rows, ncols, figsize=(ncols*1.4, n_rows*1.3),
                          squeeze=False)
rng = np.random.default_rng(0)
for row_idx, sid in enumerate(uniq):
    info = super_summary[sid]
    members = np.where(labels == sid)[0]
    kind = info['kind']
    origin = [CTX_SHORT.get(c, str(c)) for c in info['origin_contexts']]
    top = ', '.join(f"{CTX_SHORT.get(int(c), str(c))}:{n/info['size']*100:.0f}%"
                     for c, n in info['top_actual_contexts'])
    row_label = f'tok {sid}\n{kind}\norigin={",".join(origin[:4])}\nsize={info["size"]}\n{top}'
    picks = rng.choice(members, size=min(ncols, len(members)), replace=False)
    for j, ax in enumerate(axes[row_idx]):
        if j < len(picks):
            ax.imshow(tf_specs[picks[j]].T, aspect='auto', origin='lower',
                      cmap='magma', interpolation='nearest')
            if j == 0:
                ax.set_ylabel(row_label, fontsize=6, rotation=0,
                              labelpad=70, va='center', ha='right')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
fig.suptitle(f'HDP-approx super-clusters (UMAP-2D k=10 q=0.05, 21 atoms)\n'
              f'Random 8 mel-spec examples per atom', fontsize=12, y=1.001)
fig.subplots_adjust(left=0.18, right=0.99, top=0.985, bottom=0.005, wspace=0.08, hspace=0.4)
fig.savefig(FIG / 'spec_gallery.png', dpi=130, bbox_inches='tight')
plt.close()
print(f'Saved {FIG}/spec_gallery.png')

# Fig 3: Context distribution bar chart per super-cluster
cols = 4
rows = (len(uniq) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*3.3, rows*1.2), squeeze=False)
for idx, sid in enumerate(uniq):
    info = super_summary[sid]
    r, c = idx // cols, idx % cols
    ax = axes[r][c]
    members = np.where(labels == sid)[0]
    cnt = Counter(ctx[members]).most_common(8)
    names = [CTX_SHORT.get(int(k), str(k)) for k, _ in cnt]
    vals = [v/len(members)*100 for _, v in cnt]
    ax.barh(range(len(names)), vals[::-1], color='steelblue')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=6)
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
    ax.tick_params(axis='x', labelsize=5)
    title_color = 'green' if info['kind'] == 'context-specific' else 'darkblue'
    ax.set_title(f'tok {sid} ({info["kind"][:5]}, n={info["size"]})',
                  fontsize=7, color=title_color)
for idx in range(len(uniq), rows*cols):
    r, c = idx // cols, idx % cols
    axes[r][c].axis('off')
fig.suptitle('HDP-approx: context distribution per super-cluster\n'
              '(green = origin from 1 context, blue = shared across contexts)',
              fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig(FIG / 'context_bars.png', dpi=130, bbox_inches='tight')
plt.close()
print(f'Saved {FIG}/context_bars.png')

# Save labels for future use
joblib.dump({
    'labels': labels,
    'super_summary': dict(super_summary),
    'silh_umap': sil_umap,
    'silh_mel_pca50': sil_mel,
    'silh_mel_raw': sil_mel_raw,
}, CKPT / 'hdp_approx_best.joblib', compress=3)
print(f'\nAll figures + state saved.')
