"""Analyse results of scripts/run_embedding_sweep.py.

Produces:
    - docs/thesis/figures/embedding_sweep_table.tex  (LaTeX table)
    - docs/thesis/figures/embedding_sweep_summary.md  (human summary)
    - docs/thesis/figures/embedding_sweep_deltas.csv   (Δ vs baseline + CI per
      (embedding, method, metric))
    - docs/thesis/figures/embedding_sweep_best.json   (best configuration per
      metric across the 4 embeddings)

Usage:
    python scripts/analyze_embedding_sweep.py

Assumes scripts/run_embedding_sweep.py has been run and CSVs are in
docs/thesis/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'docs' / 'thesis' / 'figures'

EMB_LABEL = {
    'umap_2d':       'UMAP-2D (mel)',
    'umap_16d':      'UMAP-16D (mel)',
    'beats_768d':    'BEATs-768D (diag.~cov.)',
    'beats_umap_16d':'BEATs + UMAP-16D',
}
METHOD_LABEL = {
    'baseline':          'Assom + RF',
    'dpgmm_empirical':   'DP-GMM, эмпирич.',
    'dpgmm_uniform':     'DP-GMM, равн.',
    'hdbscan_empirical': 'HDBSCAN, эмпирич.',
    'hdbscan_uniform':   'HDBSCAN, равн.',
    'kmeans_empirical':  'k-means, эмпирич.',
    'kmeans_uniform':    'k-means, равн.',
}
METRICS = ['weighted_f1', 'macro_f1', 'mcc']
METRIC_LABEL = {'weighted_f1': 'weighted F1',
                'macro_f1': 'macro F1',
                'mcc': 'MCC'}


def main() -> int:
    csv = OUT / 'embedding_sweep_5seeds.csv'
    if not csv.exists():
        print(f'ERROR: {csv} not found — run run_embedding_sweep.py first.',
              file=sys.stderr)
        return 2

    df = pd.read_csv(csv)

    # 1. Δ (method − baseline) per (embedding, method, metric) with 95%-CI
    rows = []
    for emb, g_emb in df.groupby('embedding'):
        b = g_emb[g_emb['method'] == 'baseline'].sort_values('seed')
        for method, g_m in g_emb.groupby('method'):
            if method == 'baseline':
                continue
            a = g_m.sort_values('seed')
            for metric in METRICS:
                d = a[metric].values - b[metric].values
                m = d.mean(); s = d.std(ddof=1)
                ci_lo, ci_hi = m - 2 * s, m + 2 * s
                sig = 'SIG+' if ci_lo > 0 else ('SIG−' if ci_hi < 0 else 'n.s.')
                rows.append({
                    'embedding': emb, 'method': method, 'metric': metric,
                    'method_mean': float(a[metric].mean()),
                    'method_std':  float(a[metric].std(ddof=1)),
                    'baseline_mean': float(b[metric].mean()),
                    'baseline_std':  float(b[metric].std(ddof=1)),
                    'delta_mean': float(m),
                    'delta_std':  float(s),
                    'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
                    'significant': sig,
                })
    df_delta = pd.DataFrame(rows)
    df_delta.to_csv(OUT / 'embedding_sweep_deltas.csv', index=False)

    # 2. Best configuration per metric across all embeddings × methods (including prior)
    best = {}
    for metric in METRICS:
        d = df_delta[df_delta['metric'] == metric].sort_values('delta_mean', ascending=False)
        best[metric] = d.iloc[0].to_dict()
    with (OUT / 'embedding_sweep_best.json').open('w', encoding='utf-8') as f:
        json.dump(best, f, ensure_ascii=False, indent=2, default=float)

    # 3. Per-embedding table (method × metric, showing Δ and sig)
    # Build one pivot per metric
    with (OUT / 'embedding_sweep_summary.md').open('w', encoding='utf-8') as f:
        f.write('# Embedding sweep — Δ над baseline по трём метрикам\n\n')
        f.write('5 сидов, 30 train / 11 test emitters; формат: `Δ (95%-CI) [SIG]`.\n\n')
        for metric in METRICS:
            f.write(f'## {METRIC_LABEL[metric]}\n\n')
            d = df_delta[df_delta['metric'] == metric]
            piv = d.pivot(index='method', columns='embedding', values='delta_mean')
            pst = d.pivot(index='method', columns='embedding', values='delta_std')
            sig = d.pivot(index='method', columns='embedding', values='significant')
            embs = [c for c in ['umap_2d', 'umap_16d', 'beats_768d', 'beats_umap_16d']
                    if c in piv.columns]
            f.write('| Метод | ' + ' | '.join(EMB_LABEL[e] for e in embs) + ' |\n')
            f.write('|---|' + '|'.join(['---'] * len(embs)) + '|\n')
            for method in ['dpgmm_empirical', 'dpgmm_uniform',
                           'hdbscan_empirical', 'hdbscan_uniform',
                           'kmeans_empirical', 'kmeans_uniform']:
                if method not in piv.index:
                    continue
                row = f'| {METHOD_LABEL[method]} '
                for emb in embs:
                    m = piv.loc[method, emb]
                    s = pst.loc[method, emb]
                    ci_lo, ci_hi = m - 2*s, m + 2*s
                    tag = sig.loc[method, emb]
                    marker = '**' if tag == 'SIG+' else ''
                    row += f'| {marker}{m:+.3f}{marker} [{ci_lo:+.3f}, {ci_hi:+.3f}] {tag} '
                row += '|\n'
                f.write(row)
            f.write('\n')
        f.write('## Best configurations\n\n')
        for metric in METRICS:
            b = best[metric]
            f.write(f'- **{METRIC_LABEL[metric]}**: {EMB_LABEL.get(b["embedding"], b["embedding"])} '
                    f'+ {METHOD_LABEL.get(b["method"], b["method"])}: '
                    f'Δ = {b["delta_mean"]:+.3f} (CI [{b["ci_lo"]:+.3f}, {b["ci_hi"]:+.3f}]), '
                    f'method mean = {b["method_mean"]:.3f}, {b["significant"]}\n')

    # 4. LaTeX table — one compact overview: best Δ per (embedding, metric) across all DP-GMM/etc configs
    with (OUT / 'embedding_sweep_table.tex').open('w', encoding='utf-8') as f:
        f.write('% embedding_sweep_table.tex — best per-context configuration per embedding\n')
        f.write('\\begin{tabular}{llrrr}\n')
        f.write('\\toprule\n')
        f.write('Эмбеддинг & Лучшая конфиг. & $\\Delta$ weighted $F_1$ & $\\Delta$ macro $F_1$ & $\\Delta$ MCC \\\\\n')
        f.write('\\midrule\n')
        # For each embedding, find the method with best weighted_f1 gain
        for emb in ['umap_2d', 'umap_16d', 'beats_768d', 'beats_umap_16d']:
            d = df_delta[(df_delta['embedding'] == emb) & (df_delta['metric'] == 'weighted_f1')]
            if d.empty:
                continue
            best_m = d.sort_values('delta_mean', ascending=False).iloc[0]['method']
            row_vals = {}
            for metric in METRICS:
                r = df_delta[(df_delta['embedding'] == emb) &
                             (df_delta['method'] == best_m) &
                             (df_delta['metric'] == metric)].iloc[0]
                marker_open = '\\mathbf{' if r['significant'] == 'SIG+' else ''
                marker_close = '}' if r['significant'] == 'SIG+' else ''
                row_vals[metric] = f'${marker_open}{r["delta_mean"]:+.3f}{marker_close}$'
            row = (f"{EMB_LABEL[emb]} & {METHOD_LABEL[best_m]} & "
                   f"{row_vals['weighted_f1']} & {row_vals['macro_f1']} & {row_vals['mcc']} \\\\\n")
            f.write(row.replace('+', '{+}').replace('_', '\\_'))
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')

    # 5. Console summary
    print('\n=== Best configuration per metric (across ALL embeddings and priors) ===')
    for metric in METRICS:
        b = best[metric]
        print(f'  {METRIC_LABEL[metric]:14s}: {EMB_LABEL.get(b["embedding"], b["embedding"]):28s} '
              f'+ {METHOD_LABEL.get(b["method"], b["method"]):20s}  '
              f'Δ={b["delta_mean"]:+.3f}  [{b["ci_lo"]:+.3f}, {b["ci_hi"]:+.3f}]  '
              f'{b["significant"]}')

    print('\n=== Per-embedding best weighted F1 gain ===')
    for emb in ['umap_2d', 'umap_16d', 'beats_768d', 'beats_umap_16d']:
        d = df_delta[(df_delta['embedding'] == emb) & (df_delta['metric'] == 'weighted_f1')]
        if d.empty:
            continue
        r = d.sort_values('delta_mean', ascending=False).iloc[0]
        print(f"  {EMB_LABEL[emb]:28s}  best={METHOD_LABEL[r['method']]:20s}  "
              f"Δ weighted F1 = {r['delta_mean']:+.3f}  "
              f"method F1 = {r['method_mean']:.3f} vs baseline {r['baseline_mean']:.3f}  "
              f"{r['significant']}")

    print(f'\nWrote:')
    print(f'  {OUT / "embedding_sweep_summary.md"}')
    print(f'  {OUT / "embedding_sweep_deltas.csv"}')
    print(f'  {OUT / "embedding_sweep_best.json"}')
    print(f'  {OUT / "embedding_sweep_table.tex"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
