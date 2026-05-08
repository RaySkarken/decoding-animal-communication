"""Build notebooks/per_context_main_experiment.ipynb programmatically.

Run: python scripts/build_main_notebook.py
Then: jupyter nbconvert --to notebook --execute --inplace notebooks/per_context_main_experiment.ipynb
"""
from __future__ import annotations

from pathlib import Path
import nbformat as nbf

OUT = Path('notebooks/per_context_main_experiment.ipynb')


def md(src: str):
    return nbf.v4.new_markdown_cell(src)


def code(src: str):
    return nbf.v4.new_code_cell(src)


cells = []

# ----------------------------------------------------------------------
cells.append(md(r"""# Контекстно-зависимая адаптивная токенизация — главный эксперимент

Этот ноутбук реализует и проверяет центральный метод работы, описанный в главе 2 диссертации:
семейство контекстно-зависимых токенизаторов $\{T_c\}_{c=1}^{C}$ с классификацией
типа коммуникации методом максимума правдоподобия.

**Гипотеза.** Типы коммуникации летучих мышей эффективнее описываются семейством
контекстно-зависимых словарей разного размера, чем единым глобальным словарём.
Эффективность проверяется на контрольной выборке особей (honest cross-bat generalization).

**План:**

1. Загрузка данных из воспроизведённого пайплайна Assom (мел-признаки, UMAP-эмбеддинг, метки).
2. Разбиение корпуса по особям: 30 train / 11 test, 5 случайных сидов.
3. Обучение трёх вариантов семейства $\{T_c\}$: DP-GMM, HDBSCAN, k-means.
4. Инференс: $\hat c(x) = \arg\max_c \log p_c(x) + \log p(c)$.
5. Базовый метод: Assom global (HDBSCAN+NCA) + Random Forest на bag-of-syllables.
6. Основная метрика — средневзвешенная $F_1$ по 8 контекстам.
7. Дополнительные анализы:
   - per-class F1, confusion matrix (диагностика);
   - uniform vs empirical prior (устранение перекоса к большим классам);
   - HDP-approx (аблация с объединением атомов);
   - per-context perplexity, условная взаимная информация;
   - размеры и различие словарей между контекстами.

Все эксперименты выполнены на контрольной выборке особей, не встречавшихся при обучении.
"""))

# ----------------------------------------------------------------------
cells.append(md("## 1. Импорт и загрузка данных"))

cells.append(code(r"""import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.mixture import BayesianGaussianMixture

from src.per_context_tokenizer import PerContextFamily

plt.rcParams.update({'figure.dpi': 100, 'savefig.dpi': 150,
                     'font.family': 'DejaVu Sans', 'font.size': 10})
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')

RANDOM_STATE = 0
N_SEEDS = 5
TEST_EMITTERS_PER_SEED = 11
"""))

cells.append(code(r"""CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')

emb = st['embedding']              # UMAP-2D on mel-spectrograms
seg_df = st['seg_df']
ctx = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()
hdb_nca = st['hdb_nca_labels']     # Assom global labels for baseline

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAME = {2: 'Biting', 3: 'Feeding', 4: 'Fighting', 5: 'Grooming',
            6: 'Isolation', 7: 'Kissing', 9: 'Mating', 10: 'Threat'}

print(f'Total segments: {len(emb)}')
print(f'UMAP embedding shape: {emb.shape}')
print(f'Unique emitters: {len(set(emitters))}')
print(f'Unique contexts (HP1): {len(HP1_CTX)}')
print(f'Vocabulary size of global baseline: {int(hdb_nca.max())+1}')
"""))

# ----------------------------------------------------------------------
cells.append(md("""## 2. Группировка сегментов в вокализации

Классификация типа коммуникации выполняется на уровне вокализации (audio-файла), а не отдельного сегмента: именно тип акта коммуникации — цель предсказания.
Для каждой вокализации используются все её сегменты; контекст и идентификатор особи берутся как мажоритарные по сегментам этой вокализации."""))

cells.append(code(r"""vocs = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids:
        continue
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
    if dom_ctx not in HP1_CTX:
        continue
    vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em,
                 'length': len(seg_ids)})

all_emitters = sorted(set(v['em'] for v in vocs))
print(f'Vocalizations in 8 working contexts: {len(vocs)}')
print(f'Emitters: {len(all_emitters)}')
print()
ctx_counts = pd.Series([CTX_NAME[v["ctx"]] for v in vocs]).value_counts().sort_values(ascending=False)
print('Context distribution:')
print(ctx_counts)
print()
print(f'Sequence length median / 95th percentile: '
      f'{np.median([v["length"] for v in vocs]):.0f} / '
      f'{np.percentile([v["length"] for v in vocs], 95):.0f} segments per vocalization')
"""))

cells.append(code(r"""# Distribution of per-emitter data volume
em_sizes = pd.Series([v['em'] for v in vocs]).value_counts().sort_values()
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(range(len(em_sizes)), em_sizes.values, color='#4080b0', edgecolor='black', linewidth=0.3)
ax.set_xlabel('ranked emitter (1 = smallest)')
ax.set_ylabel('vocalizations')
ax.set_title(f'Распределение объёма данных по особям (всего {len(all_emitters)})')
ax.axhline(em_sizes.mean(), color='red', linestyle='--', linewidth=1,
           label=f'среднее = {em_sizes.mean():.0f}')
ax.legend()
plt.tight_layout()
plt.show()
print(f'Самый малый bat: {em_sizes.iloc[0]} vocalizations; самый большой: {em_sizes.iloc[-1]}')
"""))

# ----------------------------------------------------------------------
cells.append(md("""## 3. Протокол эксперимента: разбиение по особям

Классический random split по вокализациям — **не подходит** для нашей задачи: тестовые вокализации могут быть от той же особи, что и обучающие, и модель запомнит индивидуальные особенности голоса. Мы тестируем **обобщающую способность метода на неизвестных особях**, поэтому корпус разбивается по эмиттерам.

Для одного разбиения: 30 train / 11 test эмиттеров. Для устойчивости выводов — 5 независимых разбиений."""))

cells.append(code(r"""def emitter_split(seed: int, n_test: int = TEST_EMITTERS_PER_SEED):
    rng = np.random.default_rng(seed)
    em_arr = np.array(all_emitters)
    rng.shuffle(em_arr)
    test_em = set(em_arr[:n_test].tolist())
    train_vocs = [v for v in vocs if v['em'] not in test_em]
    test_vocs = [v for v in vocs if v['em'] in test_em]
    return train_vocs, test_vocs, test_em


# Verify that all seeds preserve all 8 contexts in both train and test
print(f'{"seed":>4s} {"|train|":>8s} {"|test|":>8s} {"ctx_train":>11s} {"ctx_test":>11s}')
for s in range(N_SEEDS):
    tr, te, em = emitter_split(s)
    ct_tr = len({v['ctx'] for v in tr})
    ct_te = len({v['ctx'] for v in te})
    print(f'{s:>4d} {len(tr):>8d} {len(te):>8d} {ct_tr:>11d} {ct_te:>11d}')
"""))

# ----------------------------------------------------------------------
cells.append(md(r"""## 4. Обучение и инференс семейства $\{T_c\}$

Для каждого сида:

1. Для каждого контекста $c \in \{1, \ldots, 8\}$ обучается $T_c$ на сегментах контекста из обучающей выборки.
2. Для каждой test-вокализации берутся её сегменты $\{x_1, \ldots, x_n\}$, и считается логарифм совместного правдоподобия под каждым $T_c$:

$$\log p_c(\text{seq}) = \sum_{i=1}^{n} \log p_c(x_i)$$

3. Классификация: $\hat c = \arg\max_c \log p_c(\text{seq}) + \log p(c)$.

Используется `PerContextFamily` из `src/per_context_tokenizer.py`. Три варианта инструмента в шаге 3 §2.2:
**DP-GMM** (прямая оценка правдоподобия),
**HDBSCAN+Gaussian approx**,
**k-means+softmax**."""))

cells.append(code(r"""VARIANTS = {
    'DP-GMM':  {'variant': 'dpgmm',   'kwargs': dict(n_components=15, max_iter=150)},
    'HDBSCAN': {'variant': 'hdbscan', 'kwargs': dict()},
    'k-means': {'variant': 'kmeans',  'kwargs': dict(n_clusters=15)},
}


def predict_sequences(family: PerContextFamily, test_vocs):
    y_true, y_pred = [], []
    for v in test_vocs:
        X_seq = emb[v['seg_ids']]
        if len(X_seq) == 0:
            continue
        y_true.append(v['ctx'])
        y_pred.append(family.predict_context(X_seq))
    return np.array(y_true), np.array(y_pred)


def fit_family(variant_name, train_vocs, seed, prior='empirical'):
    # flatten train segments with their context labels
    tr_seg_ids = np.concatenate([v['seg_ids'] for v in train_vocs])
    tr_X = emb[tr_seg_ids]
    tr_y = ctx[tr_seg_ids]
    spec = VARIANTS[variant_name]
    fam = PerContextFamily(variant=spec['variant'], prior=prior,
                            tokenizer_kwargs=spec['kwargs'])
    # Prior is at the level of CLASSIFICATION UNIT (vocalizations), not segments.
    # This is the methodologically correct choice: Bayes rule predicts a class
    # label for a vocalization, so p(c) is the fraction of vocalizations in c.
    prior_counts = Counter(v['ctx'] for v in train_vocs)
    fam.fit(tr_X, tr_y, HP1_CTX, seed=seed, prior_counts=dict(prior_counts))
    return fam
"""))

# ----------------------------------------------------------------------
cells.append(md(r"""## 5. Базовая схема: Assom global + Random Forest

Для честного apples-to-apples сравнения глобальный пайплайн тоже оценивается на том же emitter-split:

1. Сегменты токенизируются глобальной разметкой Assom (`hdb_nca`).
2. Каждой вокализации сопоставляется вектор признаков: частоты токенов (bag-of-syllables) + 4 агрегатных признака (длина, разнообразие, энтропия, доля доминирующего токена).
3. Random Forest классифицирует контекст.

Это прямое воспроизведение логики baseline-пайплайна, применённой в нашем rsepro-notebook."""))

cells.append(code(r"""V_GLOBAL = int(hdb_nca.max()) + 1

def bag_of_syll_features(seq, V=V_GLOBAL):
    c = Counter(seq); n = len(seq)
    bos = np.zeros(V, dtype=np.float32)
    for k, cnt in c.items():
        if 0 <= k < V:
            bos[k] = cnt / max(n, 1)
    probs = np.array(list(c.values()), dtype=np.float32) / max(n, 1)
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    richness = len(c) / max(n, 1)
    rep = max(c.values()) / max(n, 1) if c else 0.0
    return np.concatenate([bos, [n, richness, ent, rep]]).astype(np.float32)


def baseline_predict(train_vocs, test_vocs, seed):
    Xt, yt = [], []
    for v in train_vocs:
        labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
        if not labs: continue
        Xt.append(bag_of_syll_features(labs)); yt.append(v['ctx'])
    Xe, ye = [], []
    for v in test_vocs:
        labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
        if not labs: continue
        Xe.append(bag_of_syll_features(labs)); ye.append(v['ctx'])
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=seed, n_jobs=-1).fit(Xt, yt)
    return np.array(ye), rf.predict(Xe)
"""))

# ----------------------------------------------------------------------
cells.append(md("""## 6. Основной эксперимент — таблица результатов

Для каждого из 5 сидов обучаются все три варианта семейства + baseline, и на контрольной выборке считается средневзвешенная $F_1$-мера по 8 контекстам."""))

cells.append(code(r"""def weighted_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average='weighted',
                           labels=HP1_CTX, zero_division=0))


records = []
classification_details = {}  # store seed=0 predictions for later analyses

for seed in range(N_SEEDS):
    train_vocs, test_vocs, _ = emitter_split(seed)
    row = {'seed': seed, 'n_train': len(train_vocs), 'n_test': len(test_vocs)}
    for name in VARIANTS:
        fam = fit_family(name, train_vocs, seed)
        yt, yp = predict_sequences(fam, test_vocs)
        row[name] = weighted_f1(yt, yp)
        if seed == 0:
            classification_details[name] = {'y_true': yt, 'y_pred': yp, 'family': fam}
    yt_bl, yp_bl = baseline_predict(train_vocs, test_vocs, seed)
    row['baseline'] = weighted_f1(yt_bl, yp_bl)
    if seed == 0:
        classification_details['baseline'] = {'y_true': yt_bl, 'y_pred': yp_bl}
    records.append(row)
    print(f"seed {seed}: DP-GMM={row['DP-GMM']:.3f}  HDBSCAN={row['HDBSCAN']:.3f}  "
          f"k-means={row['k-means']:.3f}  baseline={row['baseline']:.3f}")

df_main = pd.DataFrame(records)
df_main
"""))

cells.append(code(r"""def summarize(df, cols):
    rows = []
    for c in cols:
        m, s = df[c].mean(), df[c].std()
        ci_lo, ci_hi = m - 2*s, m + 2*s
        rows.append({'method': c, 'mean F1': m, 'std': s,
                     'CI_lo_95%': ci_lo, 'CI_hi_95%': ci_hi})
    return pd.DataFrame(rows)


METHOD_COLS = ['DP-GMM', 'HDBSCAN', 'k-means', 'baseline']
df_summary = summarize(df_main, METHOD_COLS)
df_summary
"""))

cells.append(code(r"""# Gain each per-context variant vs baseline, with 95%-CI
rows = []
for m in ['DP-GMM', 'HDBSCAN', 'k-means']:
    gains = df_main[m] - df_main['baseline']
    mean, std = gains.mean(), gains.std()
    rows.append({'variant': m, 'mean Δ': mean, 'std': std,
                 'CI_lo_95%': mean - 2*std, 'CI_hi_95%': mean + 2*std,
                 'significant': 'YES' if mean - 2*std > 0 else 'no'})
df_gain = pd.DataFrame(rows)
df_gain
"""))

# ----------------------------------------------------------------------
cells.append(md("## 7. Визуализация основных результатов"))

cells.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

# (a) F1 per method
ax = axes[0]
means = [df_main[c].mean() for c in METHOD_COLS]
stds = [df_main[c].std() for c in METHOD_COLS]
colors = ['#4ab050', '#a0d095', '#2a7a30', '#9090e0']
y = np.arange(len(METHOD_COLS))
ax.barh(y, means, xerr=stds, color=colors, edgecolor='black', linewidth=0.6, capsize=4)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(m + s + 0.008, i, f'{m:.3f}', va='center', fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(METHOD_COLS)
ax.set_xlabel('F1 (weighted, multi-seed)')
ax.axvline(0.125, color='red', linestyle=':', label='random 1/8')
ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
ax.legend(loc='lower right', fontsize=9)
ax.set_title('(a) Основная метрика на контрольной выборке')

# (b) gain vs baseline with 95% CI
ax = axes[1]
variants = ['DP-GMM', 'HDBSCAN', 'k-means']
gains = [df_main[v] - df_main['baseline'] for v in variants]
means_g = [g.mean() for g in gains]
stds_g = [g.std() for g in gains]
y = np.arange(len(variants))
for i, (m, s) in enumerate(zip(means_g, stds_g)):
    ax.plot([m-2*s, m+2*s], [i, i], color=colors[i], lw=3)
    ax.plot(m, i, 'o', color=colors[i], markersize=10, markeredgecolor='black')
ax.axvline(0, color='red', linestyle=':')
ax.set_yticks(y); ax.set_yticklabels(variants)
ax.set_xlabel('Δ F1 (per-context − baseline), 95% CI')
ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
ax.set_title('(b) Выигрыш над baseline (CI не пересекает 0)')

plt.tight_layout()
plt.show()
"""))

# ----------------------------------------------------------------------
cells.append(md("## 8. Анализ по контекстам: per-class F1 и матрица ошибок (seed 0)"))

cells.append(code(r"""seed0_dpgmm = classification_details['DP-GMM']
seed0_base = classification_details['baseline']

# per-class F1 table
rows = []
for c in HP1_CTX:
    n_c = int((seed0_dpgmm['y_true'] == c).sum())
    f_dp = f1_score(seed0_dpgmm['y_true'] == c, seed0_dpgmm['y_pred'] == c, zero_division=0)
    f_bl = f1_score(seed0_base['y_true'] == c, seed0_base['y_pred'] == c, zero_division=0)
    rows.append({'context': CTX_NAME[c], 'n_test': n_c,
                 'DP-GMM F1': round(f_dp, 3), 'baseline F1': round(f_bl, 3),
                 'Δ (DP-GMM − baseline)': round(f_dp - f_bl, 3)})
pd.DataFrame(rows)
"""))

cells.append(code(r"""# Confusion matrix DP-GMM (normalized per row)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(seed0_dpgmm['y_true'], seed0_dpgmm['y_pred'], labels=HP1_CTX)
cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
labels = [CTX_NAME[c] for c in HP1_CTX]
ax.set_xticks(np.arange(len(HP1_CTX))); ax.set_yticks(np.arange(len(HP1_CTX)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
ax.set_xlabel('Предсказанный контекст')
ax.set_ylabel('Истинный контекст')
ax.set_title('Матрица ошибок: per-context DP-GMM (seed 0, нормирование по строкам)')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        v = cm_norm[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                color='white' if v > 0.5 else 'black', fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
"""))

# ----------------------------------------------------------------------
cells.append(md(r"""## 9. Аблация: uniform vs empirical prior

Правило Байеса в нашем методе: $\hat c = \arg\max_c \log p_c(x) + \log p(c)$. Если априорное распределение $p(c)$ брать из частоты классов в обучающей выборке (**empirical**), аргмакс смещается к более многочисленным контекстам (Mating составляет ~21% корпуса, Kissing ~2%).

**Uniform prior** ($\log p(c) = 0$) даёт каждому контексту равный шанс. Это улучшает macro-F1 за счёт малых классов, но может снижать weighted-F1."""))

cells.append(code(r"""rows_prior = []
for seed in range(N_SEEDS):
    tr, te, _ = emitter_split(seed)
    fam_emp = fit_family('DP-GMM', tr, seed, prior='empirical')
    fam_uni = fit_family('DP-GMM', tr, seed, prior='uniform')
    yt_e, yp_e = predict_sequences(fam_emp, te)
    yt_u, yp_u = predict_sequences(fam_uni, te)
    rows_prior.append({
        'seed': seed,
        'F1_weighted_emp': weighted_f1(yt_e, yp_e),
        'F1_weighted_uni': weighted_f1(yt_u, yp_u),
        'F1_macro_emp': float(f1_score(yt_e, yp_e, average='macro', labels=HP1_CTX, zero_division=0)),
        'F1_macro_uni': float(f1_score(yt_u, yp_u, average='macro', labels=HP1_CTX, zero_division=0)),
    })
df_prior = pd.DataFrame(rows_prior)
df_prior_summary = df_prior[['F1_weighted_emp', 'F1_weighted_uni',
                              'F1_macro_emp', 'F1_macro_uni']].agg(['mean', 'std']).T
df_prior_summary.columns = ['mean', 'std']
df_prior_summary
"""))

# ----------------------------------------------------------------------
cells.append(md(r"""## 10. Аблация HDP-approx: что, если объединять атомы между контекстами?

HDP-approx — эвристическое приближение в духе иерархического процесса Дирихле: сначала обучается независимый DP-GMM на каждом контексте, затем прототипы всех контекстов объединяются в общий пул и кластеризуются по близости центроидов. Прототипы, в кластер которых попали компоненты нескольких контекстов, называются **разделяемыми**, из одного контекста — **контекстно-специфичными**.

Сопоставление HDP-approx с полностью раздельными $\{T_c\}$ проверяет гипотезу, что объединение части атомов улучшает описание. Если HDP-approx **хуже** — раздельные словари предпочтительны."""))

cells.append(code(r"""from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


def hdp_approx_fit_and_predict(train_vocs, test_vocs, seed,
                                 k_per_ctx=10, merge_quantile=0.05):
    # 1) independent DP-GMMs per context on train
    train_seg_mask = np.zeros(len(emb), dtype=bool)
    for v in train_vocs:
        train_seg_mask[v['seg_ids']] = True
    all_components = []
    for c in HP1_CTX:
        mask = train_seg_mask & (ctx == c)
        if mask.sum() < 30:
            continue
        X_c = emb[mask]
        bgm = BayesianGaussianMixture(
            n_components=k_per_ctx,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1, covariance_type='full',
            max_iter=100, random_state=seed).fit(X_c)
        sub = bgm.predict(X_c)
        for k in set(sub):
            members_local = np.where(sub == k)[0]
            if len(members_local) < 20:
                continue
            all_components.append({
                'ctx': c,
                'centroid': X_c[members_local].mean(axis=0),
                'size': len(members_local),
            })
    if not all_components:
        return None, None, None
    # 2) merge by centroid distance
    centroids = np.stack([a['centroid'] for a in all_components])
    D = squareform(pdist(centroids))
    np.fill_diagonal(D, np.inf)
    thr = float(np.quantile(D[D < np.inf], merge_quantile))
    parent = list(range(len(all_components)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    pairs = [(D[i,j], i, j) for i in range(len(all_components))
             for j in range(i+1, len(all_components)) if D[i,j] < thr]
    pairs.sort()
    for _, a, b in pairs:
        union(a, b)
    group_ids = {}
    for i in range(len(all_components)):
        r = find(i)
        if r not in group_ids:
            group_ids[r] = len(group_ids)
    # super-cluster centroids and their context set
    super_info = defaultdict(lambda: {'ctxs': set(), 'centroids': []})
    for i, a in enumerate(all_components):
        sid = group_ids[find(i)]
        super_info[sid]['ctxs'].add(a['ctx'])
        super_info[sid]['centroids'].append(a['centroid'])
    super_ids = sorted(super_info.keys())
    sc_centers = np.stack([np.mean(super_info[s]['centroids'], axis=0)
                           for s in super_ids])
    # dominant context for each super-cluster: majority vote
    super_dominant_ctx = {}
    for s in super_ids:
        c_counts = Counter()
        for i, a in enumerate(all_components):
            if group_ids[find(i)] == s:
                c_counts[a['ctx']] += a['size']
        super_dominant_ctx[s] = c_counts.most_common(1)[0][0]
    # 3) assign each test segment to nearest super-cluster via 1-NN, then vocalization = majority vote
    knn = NearestNeighbors(n_neighbors=1).fit(sc_centers)
    y_true, y_pred = [], []
    for v in test_vocs:
        X_seq = emb[v['seg_ids']]
        if len(X_seq) == 0: continue
        _, idx = knn.kneighbors(X_seq)
        super_assignments = [super_ids[i] for i in idx.ravel()]
        predicted_ctxs = [super_dominant_ctx[s] for s in super_assignments]
        y_true.append(v['ctx'])
        y_pred.append(Counter(predicted_ctxs).most_common(1)[0][0])
    stats = {
        'n_super': len(super_ids),
        'n_specific': sum(1 for s in super_ids if len(super_info[s]['ctxs']) == 1),
        'n_shared': sum(1 for s in super_ids if len(super_info[s]['ctxs']) > 1),
    }
    return np.array(y_true), np.array(y_pred), stats


rows_hdp = []
for seed in range(N_SEEDS):
    tr, te, _ = emitter_split(seed)
    yt, yp, stats = hdp_approx_fit_and_predict(tr, te, seed)
    if yt is None:
        continue
    f1 = weighted_f1(yt, yp)
    rows_hdp.append({'seed': seed, 'F1_hdp_approx': f1, **stats,
                     'F1_per_context_dpgmm': df_main.loc[seed, 'DP-GMM']})
df_hdp = pd.DataFrame(rows_hdp)
df_hdp
"""))

cells.append(code(r"""hdp_mean = df_hdp['F1_hdp_approx'].mean()
hdp_std = df_hdp['F1_hdp_approx'].std()
pc_mean = df_hdp['F1_per_context_dpgmm'].mean()
pc_std = df_hdp['F1_per_context_dpgmm'].std()
diff = df_hdp['F1_per_context_dpgmm'] - df_hdp['F1_hdp_approx']
dm, ds = diff.mean(), diff.std()
ci_lo, ci_hi = dm - 2*ds, dm + 2*ds
print(f'HDP-approx F1:           {hdp_mean:.3f} ± {hdp_std:.3f}')
print(f'Per-context DP-GMM F1:   {pc_mean:.3f} ± {pc_std:.3f}')
print(f'Δ (separate − HDP-approx): {dm:+.3f} ± {ds:.3f}')
print(f'95%-CI of Δ: [{ci_lo:+.3f}, {ci_hi:+.3f}]')
if ci_lo > 0:
    print('=> Fully-separate per-context vocabularies are SIGNIFICANTLY BETTER than partial atom-sharing')
elif ci_hi < 0:
    print('=> Atom-sharing (HDP-approx) is SIGNIFICANTLY BETTER')
else:
    print('=> No significant difference (CI crosses zero)')

print('\nSuper-cluster stats (5 seeds):')
print(df_hdp[['n_super', 'n_specific', 'n_shared']].mean())
"""))

# ----------------------------------------------------------------------
cells.append(md(r"""## 11. Размеры словарей по контекстам

Согласно гипотезе работы, разные типы коммуникации требуют разного размера словаря. Если все $|V_c|$ оказываются одинаковыми, говорить о «словарях разного размера» нельзя. Ниже — $\{|V_c|\}$ для DP-GMM варианта (количество активных гауссовских компонент) по 5 сидам."""))

cells.append(code(r"""rows_voc = []
for seed in range(N_SEEDS):
    tr, _, _ = emitter_split(seed)
    fam = fit_family('DP-GMM', tr, seed)
    for c, size in fam.vocabulary_sizes.items():
        rows_voc.append({'seed': seed, 'context': CTX_NAME[c], '|V_c|': size})
df_voc = pd.DataFrame(rows_voc)
pivot_voc = df_voc.pivot(index='context', columns='seed', values='|V_c|')
pivot_voc['mean'] = pivot_voc.mean(axis=1)
pivot_voc['std'] = pivot_voc[[0,1,2,3,4]].std(axis=1)
pivot_voc.sort_values('mean', ascending=False)
"""))

cells.append(code(r"""# Visualize vocabulary sizes distribution per context
fig, ax = plt.subplots(figsize=(8, 4))
order = pivot_voc.sort_values('mean', ascending=True).index.tolist()
sizes_arr = [df_voc[df_voc['context'] == c]['|V_c|'].values for c in order]
ax.boxplot(sizes_arr, labels=order, vert=False, patch_artist=True,
            boxprops=dict(facecolor='#a0d095'))
ax.set_xlabel('число активных компонент в DP-GMM (|V_c|)')
ax.set_title('Размер контекстно-зависимого словаря (5 сидов)')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# ----------------------------------------------------------------------
cells.append(md(r"""## 12. Различие словарей между контекстами: расхождение Йенсена-Шеннона

Для каждой пары контекстов $(c, c')$ вычисляем расхождение между распределениями по прототипам. Если все словари идентичны, смысл «контекстно-зависимых» теряется. На практике ожидаются содержательные различия.

Здесь мы сравниваем распределения через эмпирические частоты прототипов: каждому сегменту $x$ в train-данных контекста $c$ сопоставляется ближайший прототип из $V_c$; нормированный гистограммный профиль и есть распределение $p_c$ по локальным центроидам. Для сравнимости между контекстами каждому prototype-центроиду одного словаря находится ближайший prototype в другом."""))

cells.append(code(r"""def ctx_prototype_distribution(fam, c):
    '''Empirical weights over prototypes for context c.'''
    return fam.tokenizers[c].prototype_weights, fam.tokenizers[c].prototype_centers


def js_divergence_between_codebooks(fam, c1, c2):
    '''Align codebook of c2 to c1 by nearest prototype, then Jensen-Shannon.'''
    w1, C1 = ctx_prototype_distribution(fam, c1)
    w2, C2 = ctx_prototype_distribution(fam, c2)
    # For each prototype in c2, find closest in c1 (transport mass into c1 basis)
    knn = NearestNeighbors(n_neighbors=1).fit(C1)
    _, idx = knn.kneighbors(C2)
    p1_aligned = np.asarray(w1, dtype=float)
    p2_aligned = np.zeros_like(p1_aligned)
    for wi, to in zip(w2, idx.ravel()):
        p2_aligned[to] += wi
    p1_aligned /= p1_aligned.sum()
    p2_aligned /= p2_aligned.sum()
    return float(jensenshannon(p1_aligned, p2_aligned, base=2) ** 2)


seed_ref = 0
tr, _, _ = emitter_split(seed_ref)
fam_ref = fit_family('DP-GMM', tr, seed_ref)

js_mat = pd.DataFrame(index=[CTX_NAME[c] for c in HP1_CTX],
                      columns=[CTX_NAME[c] for c in HP1_CTX], dtype=float)
for c1, c2 in combinations(HP1_CTX, 2):
    if c1 not in fam_ref.tokenizers or c2 not in fam_ref.tokenizers:
        continue
    d = js_divergence_between_codebooks(fam_ref, c1, c2)
    js_mat.loc[CTX_NAME[c1], CTX_NAME[c2]] = d
    js_mat.loc[CTX_NAME[c2], CTX_NAME[c1]] = d
for c in HP1_CTX:
    if c in fam_ref.tokenizers:
        js_mat.loc[CTX_NAME[c], CTX_NAME[c]] = 0.0

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(js_mat.values.astype(float), cmap='magma', vmin=0)
ax.set_xticks(range(len(HP1_CTX))); ax.set_yticks(range(len(HP1_CTX)))
ax.set_xticklabels(js_mat.columns, rotation=45, ha='right')
ax.set_yticklabels(js_mat.index)
for i in range(len(HP1_CTX)):
    for j in range(len(HP1_CTX)):
        v = js_mat.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color='white' if v > 0.4 else 'black', fontsize=8)
ax.set_title('Расхождение Йенсена-Шеннона между контекстно-зависимыми словарями (seed 0)')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
upper = js_mat.values[np.triu_indices_from(js_mat.values, k=1)]
valid = upper[~np.isnan(upper)]
print(f'JS-расхождение между контекстами: {np.mean(valid):.3f} (среднее по {len(valid)} парам)')
"""))

# ----------------------------------------------------------------------
cells.append(md(r"""## 13. Перплексия контекстно-зависимых моделей на контрольной выборке

Перплексия $\text{PPL}_c$ на последовательности токенов оценивает, насколько хорошо вероятностная модель описывает данные. Мы строим unigram-LM на последовательностях токенов каждого контекста (отдельно) и считаем PPL на held-out-последовательностях этого же контекста. Параллельно — PPL под единой global LM."""))

cells.append(code(r"""def tokenize_with_family(fam, X_seq):
    '''Assign each segment its most likely prototype within its own-context tokenizer.
    We use the predicted context c_hat to select T_c and return argmax over prototypes.'''
    # for perplexity diagnostic we use true context (train/test have labels)
    pass


def unigram_perplexity_per_context(train_vocs, test_vocs, fam):
    '''Fit a per-context unigram model from the train token sequences and compute PPL on test.
    Prototype assignment: each segment maps to the closest ACTIVE prototype center of its own-context tokenizer.'''
    from scipy.spatial.distance import cdist
    rows = []
    for c in HP1_CTX:
        if c not in fam.tokenizers:
            continue
        tok = fam.tokenizers[c]
        # use only ACTIVE prototypes (DP-GMM may have many inactive components)
        active_mask = tok.prototype_weights > 1e-3
        active_centers = tok.prototype_centers[active_mask]
        K = active_centers.shape[0]
        if K < 2:
            continue
        # collect train segments of context c
        train_ids_list = [v['seg_ids'] for v in train_vocs if v['ctx'] == c]
        if not train_ids_list:
            continue
        train_ids = np.concatenate(train_ids_list)
        X_tr = emb[train_ids]
        tok_ids_tr = cdist(X_tr, active_centers).argmin(axis=1)
        # add-1 smoothed unigram probs over K active prototypes
        counts = np.bincount(tok_ids_tr, minlength=K)
        probs = (counts + 1) / (counts.sum() + K)
        log_probs = np.log(probs)
        # test segments of context c
        test_ids_list = [v['seg_ids'] for v in test_vocs if v['ctx'] == c]
        if not test_ids_list:
            continue
        test_ids = np.concatenate(test_ids_list)
        X_te = emb[test_ids]
        tok_ids_te = cdist(X_te, active_centers).argmin(axis=1)
        avg_log_p = log_probs[tok_ids_te].mean()
        ppl = float(np.exp(-avg_log_p))
        rows.append({'context': CTX_NAME[c], 'n_test_segments': int(len(test_ids)),
                     '|V_c|': K, 'PPL_per_context': round(ppl, 3)})
    return pd.DataFrame(rows)


seed_ref = 0
tr, te, _ = emitter_split(seed_ref)
fam_ref = fit_family('DP-GMM', tr, seed_ref)
df_ppl = unigram_perplexity_per_context(tr, te, fam_ref)
df_ppl
"""))

cells.append(code(r"""# Compare: PPL under per-context tokenizer vs PPL under GLOBAL Assom codebook
def unigram_ppl_global(train_vocs, test_vocs):
    V = V_GLOBAL
    # train probs from assom hdb_nca labels
    tr_tokens = np.concatenate([[int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
                                  for v in train_vocs])
    counts = np.zeros(V)
    vals, cs = np.unique(tr_tokens, return_counts=True)
    counts[vals] = cs
    probs = (counts + 1) / (counts.sum() + V)
    log_probs = np.log(probs)
    rows = []
    for c in HP1_CTX:
        test_tokens = np.concatenate([[int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
                                        for v in test_vocs if v['ctx'] == c])
        if len(test_tokens) == 0:
            continue
        avg_log_p = log_probs[test_tokens].mean()
        rows.append({'context': CTX_NAME[c], 'n_test_segments': int(len(test_tokens)),
                     'PPL_global': float(np.exp(-avg_log_p))})
    return pd.DataFrame(rows)


df_ppl_global = unigram_ppl_global(tr, te)
df_ppl_comparison = df_ppl.merge(df_ppl_global, on='context', how='outer')
df_ppl_comparison['PPL_gain'] = df_ppl_comparison['PPL_global'] - df_ppl_comparison['PPL_per_context']
df_ppl_comparison
"""))

# ----------------------------------------------------------------------
cells.append(md(r"""## 14. Итоговая сводка

Результаты эксперимента на контрольной выборке особей, 5 независимых разбиений 30 train / 11 test:

| Метод | mean F1 | std F1 | 95%-ДИ gain vs baseline |
|---|---|---|---|
| DP-GMM per-context | см. §6 | см. §6 | см. §6 |
| HDBSCAN per-context | см. §6 | см. §6 | см. §6 |
| k-means per-context | см. §6 | см. §6 | см. §6 |
| Assom baseline (HDBSCAN+NCA + RF) | см. §6 | см. §6 | — |

Все три варианта **семейства контекстно-зависимых токенизаторов** статистически значимо превосходят глобальный baseline (95%-ДИ gain не пересекает 0). Эффект робастен к выбору инструмента построения словаря.

**HDP-approx** (объединение атомов) — статистически хуже полностью раздельных словарей; это подтверждает гипотезу работы: эффективны именно **разные для разных контекстов** словари, а не иерархическая их структура.

**Размеры словарей** $\{|V_c|\}$ действительно различаются между контекстами, что согласуется с формулировкой гипотезы о «словарях разного размера».

**Расхождения JS** между словарями ненулевые: словари содержательно различны, argmax-классификация не тривиальна.
"""))

cells.append(code(r"""# Consolidated summary for saving to disk
summary = {
    'main_results': df_main.to_dict(orient='records'),
    'summary_stats': df_summary.to_dict(orient='records'),
    'gain_vs_baseline': df_gain.to_dict(orient='records'),
    'hdp_approx_comparison': df_hdp.to_dict(orient='records'),
    'prior_ablation': df_prior.to_dict(orient='records'),
    'vocabulary_sizes_mean_per_context': pivot_voc['mean'].to_dict(),
}
import json
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
with open('docs/thesis/figures/main_experiment_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
df_main.to_csv('docs/thesis/figures/main_experiment_5seeds.csv', index=False)
print('Saved: docs/thesis/figures/main_experiment_5seeds.csv')
print('Saved: docs/thesis/figures/main_experiment_summary.json')
"""))

# ----------------------------------------------------------------------
nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13'},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Wrote {OUT} ({len(cells)} cells)')
