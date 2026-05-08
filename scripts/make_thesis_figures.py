"""Generate figures for thesis Chapter 3 from experiment CSVs."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = Path('docs/thesis/latex/images')
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


# ---------- Figure 1: methods comparison bar chart ----------
df_var = pd.read_csv('docs/thesis/figures/percontext_variants.csv')
df_strong = pd.read_csv('docs/thesis/figures/strong_baseline_comparison.csv')

methods = [
    ('Zhang-18 + RF', df_strong['baseline_zhang18_f1'].values, '#b0b0b0'),
    ('bag-of-syll + RF', df_var['baseline_rf_f1'].values, '#9090e0'),
    ('per-context HDBSCAN', df_var['pc_hdbscan_f1'].values, '#a0d095'),
    ('per-context DP-GMM', df_var['pc_dpgmm_f1'].values, '#4ab050'),
    ('per-context k-means', df_var['pc_kmeans_f1'].values, '#2a7a30'),
]
means = [v.mean() for _, v, _ in methods]
stds = [v.std() for _, v, _ in methods]
labels = [m for m, _, _ in methods]
colors = [c for _, _, c in methods]

fig, ax = plt.subplots(figsize=(7.5, 4.2))
y = np.arange(len(methods))
bars = ax.barh(y, means, xerr=stds, color=colors, edgecolor='black',
                linewidth=0.6, capsize=4)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel('средневзвешенная $F_1$-мера на контрольной выборке')
ax.set_xlim(0, 0.6)
ax.axvline(0.125, color='red', linestyle=':', linewidth=1, alpha=0.7,
           label='случайное угадывание (1/8)')
ax.grid(axis='x', linestyle=':', alpha=0.4)
ax.invert_yaxis()
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(m + s + 0.008, i, f'{m:.3f}', va='center', fontsize=9)
ax.legend(loc='lower right', fontsize=9)
ax.set_title('Сравнение методов классификации типа коммуникации\n(5 разбиений по особям, 30 train / 11 test)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'methods_comparison.pdf')
plt.savefig(FIG_DIR / 'methods_comparison.png')
plt.close()
print(f'Saved {FIG_DIR}/methods_comparison.pdf')


# ---------- Figure 2: per-class F1 (seed 0, DP-GMM vs baseline) ----------
df_pc = pd.read_csv('docs/thesis/figures/percontext_perclass_seed0.csv')
x = np.arange(len(df_pc))
w = 0.38
fig, ax = plt.subplots(figsize=(7.8, 3.8))
ax.bar(x - w/2, df_pc['dpgmm_f1'], w, label='per-context DP-GMM',
       color='#4ab050', edgecolor='black', linewidth=0.5)
ax.bar(x + w/2, df_pc['baseline_f1'], w, label='bag-of-syll + RF (baseline)',
       color='#9090e0', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{r['context']}\n(n={r['n_test']})"
                    for _, r in df_pc.iterrows()],
                   fontsize=9)
ax.set_ylabel('$F_1$-мера на контрольной выборке (seed 0)')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.legend(loc='upper right', fontsize=9)
ax.set_title('Качество классификации по контекстам')
plt.tight_layout()
plt.savefig(FIG_DIR / 'per_class_f1.pdf')
plt.savefig(FIG_DIR / 'per_class_f1.png')
plt.close()
print(f'Saved {FIG_DIR}/per_class_f1.pdf')


# ---------- Figure 3: confusion matrix heatmap (seed 0, DP-GMM) ----------
df_cm = pd.read_csv('docs/thesis/figures/percontext_confusion_seed0.csv',
                    index_col=0)
cm = df_cm.values.astype(float)
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(df_cm.columns)))
ax.set_yticks(np.arange(len(df_cm.index)))
ax.set_xticklabels(df_cm.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(df_cm.index, fontsize=9)
ax.set_xlabel('предсказанный контекст')
ax.set_ylabel('истинный контекст')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm_norm[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                color='white' if val > 0.5 else 'black', fontsize=8)
ax.set_title('Матрица ошибок: per-context DP-GMM (seed 0, нормирование по строке)')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(FIG_DIR / 'confusion_matrix.pdf')
plt.savefig(FIG_DIR / 'confusion_matrix.png')
plt.close()
print(f'Saved {FIG_DIR}/confusion_matrix.pdf')


# ---------- Figure 4: gain summary with 95%-CIs ----------
gains = [
    ('vs bag-of-syll', df_var['gain_dpgmm'].values, '#4ab050'),
    ('vs bag-of-syll\n(HDBSCAN variant)', df_var['gain_hdbscan'].values, '#a0d095'),
    ('vs bag-of-syll\n(k-means variant)', df_var['gain_kmeans'].values, '#2a7a30'),
    ('vs Zhang-18', df_strong['gain_vs_zhang18'].values, '#f0a040'),
]
means = [g.mean() for _, g, _ in gains]
stds = [g.std() for _, g, _ in gains]
cis_lo = [m - 2*s for m, s in zip(means, stds)]
cis_hi = [m + 2*s for m, s in zip(means, stds)]
labels = [l for l, _, _ in gains]
colors = [c for _, _, c in gains]

fig, ax = plt.subplots(figsize=(7.5, 3.6))
y = np.arange(len(gains))
for i, (m, lo, hi, col) in enumerate(zip(means, cis_lo, cis_hi, colors)):
    ax.plot([lo, hi], [i, i], color=col, lw=3)
    ax.plot(m, i, 'o', color=col, markersize=9, markeredgecolor='black')
ax.axvline(0, color='red', linestyle=':', linewidth=1.2)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('прирост $F_1$-меры над глобальным baseline (mean, 95\\%-ДИ)')
ax.set_xlim(-0.05, 0.32)
ax.grid(axis='x', linestyle=':', alpha=0.4)
ax.invert_yaxis()
ax.set_title('Выигрыш контекстно-зависимой токенизации\n(5 сидов, 95\\%-ДИ не пересекает 0 для всех вариантов)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'gain_summary.pdf')
plt.savefig(FIG_DIR / 'gain_summary.png')
plt.close()
print(f'Saved {FIG_DIR}/gain_summary.pdf')


# ---------- Figure 5: within-bat (Assom) vs cross-bat protocol ----------
df_ab = pd.read_csv('docs/thesis/figures/assom_perbat_replica.csv')

fig, ax = plt.subplots(figsize=(7.5, 3.8))
n_ctx = df_ab['n_ctx'].values
f1_w = df_ab['f1_weighted'].values
sc = ax.scatter(n_ctx + np.random.default_rng(0).uniform(-0.15, 0.15, len(n_ctx)),
                f1_w, alpha=0.7, s=60, c='#4080b0', edgecolor='black')
ax.axhline(df_ab['f1_weighted'].mean(), color='#4080b0', linestyle='--',
           linewidth=1.2, label=f"Assom per-bat, среднее = {df_ab['f1_weighted'].mean():.2f}")
ax.axhline(df_var['pc_dpgmm_f1'].mean(), color='#4ab050', linestyle='--',
           linewidth=1.2, label=f"per-context DP-GMM (cross-bat) = {df_var['pc_dpgmm_f1'].mean():.2f}")
ax.axhline(df_var['baseline_rf_f1'].mean(), color='#9090e0', linestyle='--',
           linewidth=1.2, label=f"bag-of-syll + RF (cross-bat) = {df_var['baseline_rf_f1'].mean():.2f}")
ax.axhline(0.125, color='red', linestyle=':', linewidth=1,
           label='случайное угадывание')
ax.set_xlabel('число контекстов у особи')
ax.set_ylabel('$F_1$-мера')
ax.set_ylim(0, 1)
ax.set_xticks(range(2, 9))
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.legend(loc='upper right', fontsize=9)
ax.set_title('Within-bat (Assom) vs cross-bat (настоящая работа)\nточка — один bat в within-bat протоколе')
plt.tight_layout()
plt.savefig(FIG_DIR / 'withinbat_vs_crossbat.pdf')
plt.savefig(FIG_DIR / 'withinbat_vs_crossbat.png')
plt.close()
print(f'Saved {FIG_DIR}/withinbat_vs_crossbat.pdf')

print('\nAll figures generated.')
