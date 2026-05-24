"""Сводка согласованности кластеров с акустическим DTW-прокси.

Из per_bat_context_proxy_agreement.csv:
  - ARI_global_proxy, NMI_global_proxy — oporny pipeline (HDBSCAN глобально)
  - ARI_pc_proxy, NMI_pc_proxy — per-context (DP-GMM на каждый контекст)

Агрегируем по особям и контекстам, средние/std, paired comparison.
"""
from pathlib import Path
import numpy as np, pandas as pd

CTX_NAME = {2: 'Biting', 3: 'Feeding', 4: 'Fighting', 5: 'Grooming',
            6: 'Isolation', 7: 'Kissing', 9: 'Mating', 10: 'Threat'}

df = pd.read_csv('docs/thesis/figures/per_bat_context_proxy_agreement.csv')
print(f'rows: {len(df)}, bats: {df.bat.nunique()}, contexts: {df.context.nunique()}')
df['CTX'] = df['context'].map(CTX_NAME)

# Per-context averages
print('\n=== Per-context proxy agreement (averaged over bats) ===')
print(f'{"context":12s} {"n_bats":>7s} {"ARI_glob":>10s} {"ARI_PC":>10s} {"ΔARI":>8s} {"NMI_glob":>10s} {"NMI_PC":>10s} {"ΔNMI":>8s}')
print('-' * 90)
agg_rows = []
for c in sorted(df['context'].unique()):
    g = df[df['context'] == c]
    aw_g = g['ARI_global_proxy'].mean()
    aw_p = g['ARI_pc_proxy'].mean()
    nw_g = g['NMI_global_proxy'].mean()
    nw_p = g['NMI_pc_proxy'].mean()
    name = CTX_NAME.get(c, str(c))
    print(f'{name:12s} {len(g):7d} {aw_g:10.3f} {aw_p:10.3f} {aw_p-aw_g:+8.3f} {nw_g:10.3f} {nw_p:10.3f} {nw_p-nw_g:+8.3f}')
    agg_rows.append({'context': name, 'n_bats': len(g),
                     'ARI_global': aw_g, 'ARI_PC': aw_p, 'dARI': aw_p-aw_g,
                     'NMI_global': nw_g, 'NMI_PC': nw_p, 'dNMI': nw_p-nw_g})

agg = pd.DataFrame(agg_rows)
agg.to_csv('docs/thesis/figures/proxy_agreement_percontext.csv', index=False)
print(f'\nSaved: docs/thesis/figures/proxy_agreement_percontext.csv')

# Overall
print('\n=== Overall (averaged over all bat-context pairs) ===')
print(f'  ARI global proxy: {df["ARI_global_proxy"].mean():.3f} ± {df["ARI_global_proxy"].std():.3f}')
print(f'  ARI    PC proxy: {df["ARI_pc_proxy"].mean():.3f} ± {df["ARI_pc_proxy"].std():.3f}')
print(f'  ΔARI            : {(df["ARI_pc_proxy"] - df["ARI_global_proxy"]).mean():+.3f}')
print(f'  NMI global proxy: {df["NMI_global_proxy"].mean():.3f} ± {df["NMI_global_proxy"].std():.3f}')
print(f'  NMI    PC proxy: {df["NMI_pc_proxy"].mean():.3f} ± {df["NMI_pc_proxy"].std():.3f}')
print(f'  ΔNMI            : {(df["NMI_pc_proxy"] - df["NMI_global_proxy"]).mean():+.3f}')

# Paired test
from scipy.stats import wilcoxon
ari_diff = df['ARI_pc_proxy'] - df['ARI_global_proxy']
nmi_diff = df['NMI_pc_proxy'] - df['NMI_global_proxy']
print('\n=== Paired Wilcoxon (PC vs global) ===')
w_ari, p_ari = wilcoxon(ari_diff)
w_nmi, p_nmi = wilcoxon(nmi_diff)
print(f'  ARI: stat={w_ari:.1f}, p={p_ari:.4f}, mean diff={ari_diff.mean():+.3f}')
print(f'  NMI: stat={w_nmi:.1f}, p={p_nmi:.4f}, mean diff={nmi_diff.mean():+.3f}')

# Silhouette comparison
print('\n=== Per-context silhouette (UMAP-8D) ===')
sil = pd.read_csv('docs/thesis/figures/per_context_silhouette_8d.csv')
print(sil[['context', 'n_segs', 'silhouette_baseline', 'silhouette_pc']].to_string(index=False))
print(f'\n  mean silhouette baseline: {sil["silhouette_baseline"].mean():.3f}')
print(f'  mean silhouette       PC: {sil["silhouette_pc"].mean():.3f}')
print(f'  (baseline выше — это ожидаемо: HDBSCAN оптимизирует разделимость;')
print(f'   DP-GMM моделирует плотность и допускает перекрытие кластеров.)')
