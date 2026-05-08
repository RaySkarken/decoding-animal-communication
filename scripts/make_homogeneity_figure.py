"""Make figure for §3.6: V-measure of tokenizations vs context labels."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('docs/thesis/figures/cluster_homogeneity_sweep.csv')

display_names = {
    'G_hdbscan_nca': 'G HDB+NCA (V=11)',
    'KM_13': 'KM-13',
    'AGG_13': 'AGG-13',
    'KM_30': 'KM-30',
    'AGG_30': 'AGG-30',
    'KM_100': 'KM-100',
    'AGG_100': 'AGG-100',
    'PC_kmeans': 'PC-KM (V≈120)',
    'PC_dpgmm': 'PC-DP (V≈116)',
    'PC_hdbscan': 'PC-HDB (V≈69)',
}

ordered = ['G_hdbscan_nca', 'KM_13', 'AGG_13', 'KM_30', 'AGG_30',
           'KM_100', 'AGG_100', 'PC_kmeans', 'PC_dpgmm', 'PC_hdbscan']

is_pc = ['PC' in n for n in ordered]
df_ord = df.set_index('tokenization').loc[ordered]
labels = [display_names[n] for n in ordered]
vm = df_ord['V_measure'].values
nmi = df_ord['NMI(ctx,clust)'].values
ami = df_ord['AMI'].values
ari = df_ord['ARI'].values
mi = df_ord['I(ctx;clust) [bits]'].values

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

x = np.arange(len(ordered))
colors = ['#5078a0' if not pc else '#c25d5d' for pc in is_pc]

ax = axes[0]
ax.bar(x, vm, color=colors)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('V-мера')
ax.set_title('V-мера по контексту')
ax.set_ylim(0, 0.7)
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
ax.bar(x, mi, color=colors)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('I(C; K), бит')
ax.set_title('Взаимная информация (бит)')
ax.set_ylim(0, 3.2)
ax.axhline(np.log2(8), color='black', linestyle=':', linewidth=0.8, label='log2(8)=3')
ax.legend(loc='upper left', fontsize=8)
ax.grid(axis='y', alpha=0.3)

ax = axes[2]
ax.bar(x, ari, color=colors)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('ARI')
ax.set_title('Adjusted Rand Index')
ax.set_ylim(0, 0.25)
ax.grid(axis='y', alpha=0.3)

# Custom legend
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='#5078a0', label='Глобальная токенизация'),
                     Patch(color='#c25d5d', label='Per-context')],
            loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('docs/thesis/figures/cluster_homogeneity.pdf', bbox_inches='tight')
fig.savefig('docs/thesis/figures/cluster_homogeneity.png', bbox_inches='tight', dpi=150)
print('Saved cluster_homogeneity.{pdf,png}')
