"""Сводка noise robustness результатов в латех-форматируемую таблицу."""
import pandas as pd
df = pd.read_csv('docs/thesis/figures/noise_robustness_reco.csv')
agg = df.groupby(['kind', 'param']).agg(
    f1m_mean=('f1_m', 'mean'),
    f1m_std=('f1_m', 'std'),
    f1w_mean=('f1_w', 'mean'),
    f1w_std=('f1_w', 'std'),
).reset_index()
print('=== Noise robustness: рекомендуемая конфигурация, 5 сидов ===\n')
print(f'{"kind":8s} {"param":>6s}  {"macro F1":>14s}  {"weighted F1":>14s}')
print('-' * 60)
for _, r in agg.iterrows():
    print(f'{r["kind"]:8s} {r["param"]:>6}  {r["f1m_mean"]:.3f} ± {r["f1m_std"]:.3f}  {r["f1w_mean"]:.3f} ± {r["f1w_std"]:.3f}')

# Print clean reference
clean = agg[(agg['kind']=='clean')].iloc[0]
print(f'\n[Baseline at clean: macro F1 = {clean["f1m_mean"]:.3f} ± {clean["f1m_std"]:.3f}]')

# Compute degradation
print('\n=== Деградация относительно clean ===')
for _, r in agg.iterrows():
    if r['kind'] == 'clean': continue
    d_macro = r['f1m_mean'] - clean['f1m_mean']
    print(f'  {r["kind"]:8s} {r["param"]:>6}: ΔmacroF1 = {d_macro:+.3f}')

agg.to_csv('docs/thesis/figures/noise_robustness_summary.csv', index=False)
print('\nSaved: docs/thesis/figures/noise_robustness_summary.csv')
