"""Свежие UMAP-картинки на актуальной разметке 153k 21x32 + NCA (11 кластеров)."""
import numpy as np, joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CACHE = '/Volumes/T7/cache/assom_paper_repro'
emb = np.load(f'{CACHE}/umap_152k_21x32_md1.0.npy')
hdb = np.load(f'{CACHE}/hdb_nca_labels_152k_21x32.npy')
st = joblib.load(f'{CACHE}/ablation_state_152k_21x32.joblib')
ctx = st['seg_df']['context'].to_numpy()
ctx_name = st['seg_df']['context_name'].to_numpy()

print(f'N = {len(emb)}, clusters = {len(set(hdb.tolist()))}, contexts = {len(set(ctx.tolist()))}')

# subsample for plotting (153k -> 60k)
rng = np.random.default_rng(0)
idx = rng.choice(len(emb), 60000, replace=False)
em = emb[idx]; hd = hdb[idx]; cn = ctx_name[idx]

# --- Cluster colors: 11 distinct
n_clusters = int(hdb.max()) + 1
cmap_c = plt.get_cmap('tab20', n_clusters)

fig, ax = plt.subplots(figsize=(8.5, 8))
for k in range(n_clusters):
    m = hd == k
    ax.scatter(em[m, 0], em[m, 1], s=1.2, c=[cmap_c(k)], alpha=0.55,
                label=f'{k}', linewidths=0)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
ax.set_title(f'UMAP мел-спектрограмм 21×32 — раскраска по {n_clusters} HDBSCAN-кластерам (после NCA)')
leg = ax.legend(title='кластер', markerscale=6, loc='center left',
                 bbox_to_anchor=(1.0, 0.5), frameon=False, ncol=1)
fig.tight_layout()
fig.savefig('docs/thesis/figures/umap_153k_by_cluster.pdf', bbox_inches='tight')
fig.savefig('docs/thesis/figures/umap_153k_by_cluster.png', bbox_inches='tight', dpi=140)
print('saved: umap_153k_by_cluster.{pdf,png}')

# --- Context colors
unique_ctx_names = ['Mating protest', 'Separation', 'Feeding', 'Fighting', 'Isolation',
                     'Biting', 'Threat-like', 'Grooming', 'Kissing', 'Landing']
ctx_to_idx = {n: i for i, n in enumerate(unique_ctx_names)}
cmap_x = plt.get_cmap('tab10', len(unique_ctx_names))

fig, ax = plt.subplots(figsize=(8.5, 8))
for cname, ci in ctx_to_idx.items():
    m = cn == cname
    if m.sum() == 0: continue
    ax.scatter(em[m, 0], em[m, 1], s=1.2, c=[cmap_x(ci)], alpha=0.55,
                label=cname, linewidths=0)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
ax.set_title('UMAP мел-спектрограмм 21×32 — раскраска по поведенческому контексту')
leg = ax.legend(title='контекст', markerscale=6, loc='center left',
                 bbox_to_anchor=(1.0, 0.5), frameon=False, ncol=1)
fig.tight_layout()
fig.savefig('docs/thesis/figures/umap_153k_by_context.pdf', bbox_inches='tight')
fig.savefig('docs/thesis/figures/umap_153k_by_context.png', bbox_inches='tight', dpi=140)
print('saved: umap_153k_by_context.{pdf,png}')
