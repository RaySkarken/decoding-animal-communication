"""Regenerate thesis figures from the final notebook-produced CSV/JSON.

Inputs: docs/thesis/figures/main_experiment_5seeds.csv
        docs/thesis/figures/main_experiment_summary.json
Outputs: docs/thesis/latex/images/*.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

FIG = Path('docs/thesis/latex/images')
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

df = pd.read_csv('docs/thesis/figures/main_experiment_5seeds.csv')
summary = json.load(open('docs/thesis/figures/main_experiment_summary.json'))


# ---- Figure: methods comparison ----
methods = ['baseline', 'DP-GMM', 'HDBSCAN', 'k-means']
labels = ['Assom + RF (baseline)', 'per-context DP-GMM', 'per-context HDBSCAN', 'per-context k-means']
colors = ['#9090e0', '#4ab050', '#a0d095', '#2a7a30']
means = [df[m].mean() for m in methods]
stds = [df[m].std() for m in methods]

fig, ax = plt.subplots(figsize=(7.5, 3.8))
y = np.arange(len(methods))
ax.barh(y, means, xerr=stds, color=colors, edgecolor='black', linewidth=0.5, capsize=4)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(m + s + 0.008, i, f'{m:.3f}', va='center', fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel('средневзвешенная F1 на контрольной выборке (5 сидов)')
ax.axvline(0.125, color='red', linestyle=':', linewidth=1, label='случайное угадывание 1/8')
ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim(0, 0.6)
plt.tight_layout()
plt.savefig(FIG / 'methods_comparison.pdf')
plt.close()
print(f'Saved {FIG}/methods_comparison.pdf')


# ---- Figure: gain vs baseline with 95% CI ----
variants = ['DP-GMM', 'HDBSCAN', 'k-means']
labels_g = ['per-context DP-GMM', 'per-context HDBSCAN', 'per-context k-means']
cols = ['#4ab050', '#a0d095', '#2a7a30']
means_g, stds_g = [], []
for v in variants:
    d = df[v] - df['baseline']
    means_g.append(d.mean()); stds_g.append(d.std())

fig, ax = plt.subplots(figsize=(7.5, 3.2))
y = np.arange(len(variants))
for i, (m, s, c) in enumerate(zip(means_g, stds_g, cols)):
    lo, hi = m - 2*s, m + 2*s
    ax.plot([lo, hi], [i, i], color=c, lw=3)
    ax.plot(m, i, 'o', color=c, markersize=9, markeredgecolor='black')
    ax.text(hi + 0.005, i, f'Δ={m:+.3f}', va='center', fontsize=9)
ax.axvline(0, color='red', linestyle=':', linewidth=1.2)
ax.set_yticks(y); ax.set_yticklabels(labels_g)
ax.set_xlabel('Δ F1 над baseline (mean, 95%-ДИ, 5 сидов)')
ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
ax.set_xlim(-0.04, 0.22)
plt.tight_layout()
plt.savefig(FIG / 'gain_summary.pdf')
plt.close()
print(f'Saved {FIG}/gain_summary.pdf')


# ---- Figure: HDP-approx vs per-context DP-GMM ----
hdp = pd.DataFrame(summary['hdp_approx_comparison'])
fig, ax = plt.subplots(figsize=(7.5, 3.6))
x = np.arange(len(hdp))
w = 0.38
ax.bar(x - w/2, hdp['F1_per_context_dpgmm'], w, label='per-context DP-GMM (полностью раздельные)',
       color='#4ab050', edgecolor='black', linewidth=0.5)
ax.bar(x + w/2, hdp['F1_hdp_approx'], w, label='HDP-approx (объединение атомов)',
       color='#f0a040', edgecolor='black', linewidth=0.5)
for i, (a, b) in enumerate(zip(hdp['F1_per_context_dpgmm'], hdp['F1_hdp_approx'])):
    ax.text(i - w/2, a + 0.01, f'{a:.2f}', ha='center', fontsize=9)
    ax.text(i + w/2, b + 0.01, f'{b:.2f}', ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f'seed {s}' for s in hdp['seed']])
ax.set_ylabel('средневзвешенная F1')
ax.set_ylim(0, 0.75)
ax.grid(axis='y', alpha=0.3)
ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig(FIG / 'hdp_vs_percontext.pdf')
plt.close()
print(f'Saved {FIG}/hdp_vs_percontext.pdf')


# ---- Figure: vocabulary size per context ----
voc_means = summary['vocabulary_sizes_mean_per_context']
ctx_order = sorted(voc_means, key=lambda c: voc_means[c])
sizes = [voc_means[c] for c in ctx_order]

fig, ax = plt.subplots(figsize=(7.5, 3.2))
y = np.arange(len(ctx_order))
ax.barh(y, sizes, color='#a0d095', edgecolor='black', linewidth=0.5)
for i, s in enumerate(sizes):
    ax.text(s + 0.15, i, f'{s:.1f}', va='center', fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(ctx_order)
ax.set_xlabel('среднее число активных прототипов |V_c| (5 сидов)')
ax.grid(axis='x', alpha=0.3)
ax.set_title('Размеры контекстно-зависимых словарей различаются между контекстами')
plt.tight_layout()
plt.savefig(FIG / 'vocab_sizes.pdf')
plt.close()
print(f'Saved {FIG}/vocab_sizes.pdf')


# ---- Figure: prior ablation ----
pr = pd.DataFrame(summary['prior_ablation'])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, (metric_emp, metric_uni, title) in zip(axes, [
    ('F1_weighted_emp', 'F1_weighted_uni', 'weighted F1'),
    ('F1_macro_emp', 'F1_macro_uni', 'macro F1'),
]):
    x = np.arange(len(pr))
    w = 0.38
    ax.bar(x - w/2, pr[metric_emp], w, label='эмпирический prior',
           color='#4ab050', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, pr[metric_uni], w, label='равномерный prior',
           color='#80b0c0', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f's{s}' for s in pr['seed']])
    ax.set_ylabel(title)
    ax.set_ylim(0, max(pr[[metric_emp, metric_uni]].max().max() * 1.15, 0.1))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(FIG / 'prior_ablation.pdf')
plt.close()
print(f'Saved {FIG}/prior_ablation.pdf')


# ---- Figure: per-class F1 (seed 0) ----
# Recompute from saved data since per-class info is not in summary
CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
emb = st['embedding']
seg_df = st['seg_df']
ctx_arr = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()
hdb_nca = st['hdb_nca_labels']
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAME = {2: 'Biting', 3: 'Feeding', 4: 'Fighting', 5: 'Grooming',
            6: 'Isolation', 7: 'Kissing', 9: 'Mating', 10: 'Threat'}

import sys
sys.path.insert(0, '.')
from src.per_context_tokenizer import PerContextFamily
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from collections import Counter as _Counter

vocs = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids: continue
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    dom_em = int(_Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
    if dom_ctx not in HP1_CTX: continue
    vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em})
all_emitters = sorted(set(v['em'] for v in vocs))
seed0 = 0
rng = np.random.default_rng(seed0)
em_arr = np.array(all_emitters); rng.shuffle(em_arr)
test_em = set(em_arr[:11].tolist())
train_vocs = [v for v in vocs if v['em'] not in test_em]
test_vocs = [v for v in vocs if v['em'] in test_em]

tr_seg = np.concatenate([v['seg_ids'] for v in train_vocs])
prior_counts = _Counter(v['ctx'] for v in train_vocs)
fam = PerContextFamily(variant='dpgmm', tokenizer_kwargs={'n_components': 15, 'max_iter': 150})
fam.fit(emb[tr_seg], ctx_arr[tr_seg], HP1_CTX, seed=seed0, prior_counts=dict(prior_counts))
yt_pc, yp_pc = [], []
for v in test_vocs:
    X = emb[v['seg_ids']]
    if len(X) == 0: continue
    yt_pc.append(v['ctx']); yp_pc.append(fam.predict_context(X))
yt_pc, yp_pc = np.array(yt_pc), np.array(yp_pc)

# baseline
V_GLOBAL = int(hdb_nca.max()) + 1
def bag_feats(seq, V=V_GLOBAL):
    c = _Counter(seq); n = len(seq)
    bos = np.zeros(V, dtype=np.float32)
    for k, cnt in c.items():
        if 0 <= k < V: bos[k] = cnt / max(n, 1)
    probs = np.array(list(c.values()), dtype=np.float32) / max(n, 1)
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    richness = len(c) / max(n, 1)
    rep = max(c.values()) / max(n, 1) if c else 0.0
    return np.concatenate([bos, [n, richness, ent, rep]]).astype(np.float32)

Xt, yt, Xe, ye = [], [], [], []
for v in train_vocs:
    labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
    if not labs: continue
    Xt.append(bag_feats(labs)); yt.append(v['ctx'])
for v in test_vocs:
    labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
    if not labs: continue
    Xe.append(bag_feats(labs)); ye.append(v['ctx'])
rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                             random_state=seed0, n_jobs=-1).fit(Xt, yt)
yp_bl = rf.predict(Xe)
yt_bl, yp_bl = np.array(ye), np.array(yp_bl)

per_class = []
for c in HP1_CTX:
    n_c = int((yt_pc == c).sum())
    f_dp = f1_score(yt_pc == c, yp_pc == c, zero_division=0)
    f_bl = f1_score(yt_bl == c, yp_bl == c, zero_division=0)
    per_class.append({'context': CTX_NAME[c], 'n_test': n_c,
                       'dpgmm_f1': round(f_dp, 3),
                       'baseline_f1': round(f_bl, 3)})
pc_df = pd.DataFrame(per_class)
pc_df.to_csv('docs/thesis/figures/per_class_seed0.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 3.6))
x = np.arange(len(pc_df))
w = 0.38
ax.bar(x - w/2, pc_df['dpgmm_f1'], w, label='per-context DP-GMM',
       color='#4ab050', edgecolor='black', linewidth=0.5)
ax.bar(x + w/2, pc_df['baseline_f1'], w, label='Assom + RF (baseline)',
       color='#9090e0', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{r['context']}\n(n={r['n_test']})" for _, r in pc_df.iterrows()], fontsize=9)
ax.set_ylabel('F1 (seed 0)')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(FIG / 'per_class_f1.pdf')
plt.close()
print(f'Saved {FIG}/per_class_f1.pdf')

# confusion matrix
cm = confusion_matrix(yt_pc, yp_pc, labels=HP1_CTX)
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
names = [CTX_NAME[c] for c in HP1_CTX]
ax.set_xticks(np.arange(len(HP1_CTX))); ax.set_yticks(np.arange(len(HP1_CTX)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('предсказанный контекст')
ax.set_ylabel('истинный контекст')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        v = cm_norm[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                color='white' if v > 0.5 else 'black', fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(FIG / 'confusion_matrix.pdf')
plt.close()
print(f'Saved {FIG}/confusion_matrix.pdf')

print('\nAll thesis figures regenerated.')
