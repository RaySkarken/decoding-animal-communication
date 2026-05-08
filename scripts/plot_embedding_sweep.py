"""Plot embedding sweep results as a grouped bar chart (one panel per metric).

Reads docs/thesis/figures/embedding_sweep_5seeds.csv and produces:
    docs/thesis/latex/images/embedding_sweep.pdf
    docs/thesis/latex/images/embedding_sweep.png

Layout: 3 panels (weighted F1, macro F1, MCC). X axis: 4 embeddings.
Bars: best per-context config, baseline. Error bars = std over 5 seeds.

Run after scripts/run_embedding_sweep.py has populated the CSV.
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

EMBS = [('umap_2d',        'UMAP-2D'),
        ('umap_16d',       'UMAP-16D'),
        ('beats_768d',     'BEATs-768D'),
        ('beats_umap_16d', 'BEATs + UMAP-16D')]
METHODS = [('baseline',         'Assom+RF', '#9090e0'),
           ('dpgmm_empirical',  'DP-GMM эмп.', '#4ab050'),
           ('dpgmm_uniform',    'DP-GMM равн.', '#a0d095'),
           ('kmeans_empirical', 'k-means эмп.', '#2a7a30')]
METRICS = [('weighted_f1', 'weighted $F_1$'),
           ('macro_f1',    'macro $F_1$'),
           ('mcc',         'MCC')]


def main() -> int:
    csv = FIG / 'embedding_sweep_5seeds.csv'
    if not csv.exists():
        print(f'ERROR: {csv} not found.', file=sys.stderr)
        return 2
    df = pd.read_csv(csv)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

    x = np.arange(len(EMBS))
    n_methods = len(METHODS)
    width = 0.8 / n_methods

    for ax, (metric, label) in zip(axes, METRICS):
        for j, (method, mlabel, color) in enumerate(METHODS):
            means, stds = [], []
            for emb_key, _ in EMBS:
                g = df[(df['embedding'] == emb_key) & (df['method'] == method)]
                means.append(g[metric].mean() if len(g) else np.nan)
                stds.append(g[metric].std(ddof=1) if len(g) > 1 else 0.0)
            xpos = x - 0.4 + (j + 0.5) * width
            ax.bar(xpos, means, width=width * 0.95, yerr=stds, color=color,
                    edgecolor='black', linewidth=0.5, capsize=2, label=mlabel)
        ax.set_xticks(x)
        ax.set_xticklabels([e[1] for e in EMBS], rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=0.3)
        ax.set_title(label)
        if metric == 'weighted_f1':
            ax.axhline(0.125, color='red', linestyle=':', lw=1, label='random 1/8')
    # single legend on first axes
    axes[0].legend(loc='upper left', fontsize=7, frameon=False)
    plt.suptitle('Сравнение 4 эмбеддингов × 4 методов по 3 метрикам (5 сидов, std)',
                 y=1.02)
    plt.tight_layout()
    IMG.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMG / 'embedding_sweep.pdf', bbox_inches='tight')
    plt.savefig(IMG / 'embedding_sweep.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Wrote {IMG / "embedding_sweep.pdf"} and .png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
