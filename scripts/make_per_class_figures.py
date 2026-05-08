"""Regenerate per-class F1 bar chart for the recommended configuration.

Uses docs/thesis/figures/umap8d_kmeans_per_class.csv (5-seed per-class F1 for
UMAP-8D + kmeans_emp) and extended_metrics_per_class.csv (5-seed per-class F1 for
UMAP-2D baseline and DP-GMM variants) to produce a grouped bar chart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / 'docs' / 'thesis' / 'figures'
IMG = REPO / 'docs' / 'thesis' / 'latex' / 'images'

CTX_ORDER = ['Mating', 'Feeding', 'Isolation', 'Biting',
             'Fighting', 'Threat', 'Grooming', 'Kissing']


def main() -> int:
    # Load UMAP-8D kmeans emp per-class
    df_8d = pd.read_csv(FIG / 'umap8d_kmeans_per_class.csv')
    pc_8d = df_8d.groupby('context')['f1'].agg(['mean', 'std']).reindex(CTX_ORDER)

    # Load UMAP-2D DP-GMM and baseline per-class from extended_metrics_per_class.csv
    df_2d = pd.read_csv(FIG / 'extended_metrics_per_class.csv')
    pc_base = df_2d[df_2d['method'] == 'baseline'].groupby('context')['f1'].agg(
        ['mean', 'std']).reindex(CTX_ORDER)
    pc_dpgmm_emp = df_2d[df_2d['method'] == 'dpgmm_empirical'].groupby('context')['f1'].agg(
        ['mean', 'std']).reindex(CTX_ORDER)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(CTX_ORDER))
    w = 0.27
    bars1 = ax.bar(x - w, pc_base['mean'].values, width=w, yerr=pc_base['std'].values,
                    color='#9090e0', edgecolor='black', linewidth=0.6, capsize=2,
                    label='Assom + RF (UMAP-2D)')
    bars2 = ax.bar(x, pc_dpgmm_emp['mean'].values, width=w, yerr=pc_dpgmm_emp['std'].values,
                    color='#4ab050', edgecolor='black', linewidth=0.6, capsize=2,
                    label='DP-GMM эмпирич. (UMAP-2D)')
    bars3 = ax.bar(x + w, pc_8d['mean'].values, width=w, yerr=pc_8d['std'].values,
                    color='#2a7a30', edgecolor='black', linewidth=0.6, capsize=2,
                    label=r'$\mathbf{\mathit{k}}$-means эмпирич. (UMAP-8D) — рекомендуется')

    ax.set_xticks(x)
    ax.set_xticklabels(CTX_ORDER, rotation=25, ha='right')
    ax.set_ylabel('$F_1$-мера')
    ax.set_title('Per-class $F_1$ на контрольной выборке (5 сидов, std показано)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, frameon=True)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    IMG.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMG / 'per_class_f1_5seeds.pdf', bbox_inches='tight')
    plt.savefig(IMG / 'per_class_f1_5seeds.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Wrote {IMG / "per_class_f1_5seeds.pdf"} and .png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
