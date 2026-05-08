"""Main experiment: per-context tokenizer + Bayes classification.

Реализует центральный метод из главы 2 thesis:
  1. Разбиение эмиттеров 30 train / 11 test (multi-seed).
  2. Per-context DP-GMM: для каждого контекста c из 8 рабочих обучается
     независимая смесь гауссиан с процессом Дирихле на train сегментах
     этого контекста.
  3. Инференс на test vocalization: для каждой последовательности сегментов
     (x_1,...,x_n) вычисляется log p_c(sequence) = sum_i log p_c(x_i),
     прибавляется log p(c), выбирается argmax_c.
  4. Метрика: средневзвешенная F1-мера по 8 контекстам.

Baseline (apples-to-apples сравнение): Assom-style global pipeline --
HDBSCAN+NCA labels -> Zhang 18 sequence features -> Random Forest.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
emb = st['embedding']               # UMAP-2D
seg_df = st['seg_df']
ctx = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()
hdb_nca = st['hdb_nca_labels']      # Assom baseline global labels

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]  # 8 working contexts
CTX_NAME = {2:'Biting', 3:'Feeding', 4:'Fighting', 5:'Grooming',
            6:'Isolation', 7:'Kissing', 9:'Mating', 10:'Threat'}

# ----- group segments into vocalizations (per file) -----
vocs = []  # list of dicts: {seg_ids, context, emitter}
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if len(seg_ids) < 1:
        continue
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
    if dom_ctx not in HP1_CTX:
        continue
    vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em})
print(f'Total vocalizations in HP1 contexts: {len(vocs)}')
print(f'  context distribution: {Counter([v["ctx"] for v in vocs])}')

all_emitters = sorted(set(v['em'] for v in vocs))
print(f'Total emitters: {len(all_emitters)}')


# ----- Zhang-18-like sequence features (bag-of-syllables + length/entropy) -----
def seq_features(seq, vocab_size):
    """Short features used by global+RF baseline."""
    c = Counter(seq)
    n = len(seq)
    # bag-of-syllables counts
    bos = np.zeros(vocab_size, dtype=np.float32)
    for k, v in c.items():
        if 0 <= k < vocab_size:
            bos[k] = v / max(n, 1)
    # 4 simple aggregates: length, richness, entropy, repetition ratio
    richness = len(c) / max(n, 1)
    probs = np.array(list(c.values()), dtype=np.float32) / max(n, 1)
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    rep = max(c.values()) / max(n, 1) if c else 0.0
    return np.concatenate([bos, [n, richness, ent, rep]]).astype(np.float32)


def run_one_seed(seed, k_components=15):
    rng = np.random.default_rng(seed)
    em_arr = np.array(all_emitters)
    rng.shuffle(em_arr)
    test_emitters = set(em_arr[:11].tolist())
    train_emitters = set(em_arr[11:].tolist())

    train_vocs = [v for v in vocs if v['em'] in train_emitters]
    test_vocs = [v for v in vocs if v['em'] in test_emitters]

    if not train_vocs or not test_vocs:
        return None

    train_seg_mask = np.zeros(len(emb), dtype=bool)
    for v in train_vocs:
        train_seg_mask[v['seg_ids']] = True

    # ---- per-context DP-GMM fit on train ----
    tokenizers = {}
    log_prior = {}
    n_train_vocs = len(train_vocs)
    for c in HP1_CTX:
        mask = train_seg_mask & (ctx == c)
        if mask.sum() < 30:
            continue
        X_c = emb[mask]
        bgm = BayesianGaussianMixture(
            n_components=k_components,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1, covariance_type='full',
            max_iter=150, random_state=seed,
        )
        bgm.fit(X_c)
        tokenizers[c] = bgm
        n_c = sum(1 for v in train_vocs if v['ctx'] == c)
        log_prior[c] = np.log(max(n_c, 1) / n_train_vocs)

    # ---- per-context classification on test ----
    y_true, y_pred = [], []
    for v in test_vocs:
        X_seq = emb[v['seg_ids']]
        if len(X_seq) == 0:
            continue
        best_c, best_score = None, -np.inf
        for c, bgm in tokenizers.items():
            ll = bgm.score_samples(X_seq).sum()  # sum log p_c(x_i)
            score = ll + log_prior[c]
            if score > best_score:
                best_score = score
                best_c = c
        if best_c is None:
            continue
        y_true.append(v['ctx'])
        y_pred.append(best_c)

    pc_f1 = f1_score(y_true, y_pred, average='weighted',
                     labels=HP1_CTX, zero_division=0)

    # ---- baseline: global HDBSCAN+NCA labels + RF on sequence features ----
    vocab_size = int(np.max(hdb_nca)) + 1
    X_tr, y_tr = [], []
    for v in train_vocs:
        labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
        if len(labs) < 1:
            continue
        X_tr.append(seq_features(labs, vocab_size))
        y_tr.append(v['ctx'])
    X_te, y_te = [], []
    for v in test_vocs:
        labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
        if len(labs) < 1:
            continue
        X_te.append(seq_features(labs, vocab_size))
        y_te.append(v['ctx'])
    X_tr = np.array(X_tr); X_te = np.array(X_te)
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    bl_pred = rf.predict(X_te)
    bl_f1 = f1_score(y_te, bl_pred, average='weighted',
                     labels=HP1_CTX, zero_division=0)

    return {
        'seed': seed,
        'n_test_vocs_pc': len(y_pred),
        'n_test_vocs_baseline': len(bl_pred),
        'pc_f1': round(pc_f1, 3),
        'baseline_f1': round(bl_f1, 3),
        'gain': round(pc_f1 - bl_f1, 3),
        'n_tokenizers': len(tokenizers),
    }


rows = []
for seed in [0, 1, 2, 3, 4]:
    print(f'\n=== seed {seed} ===')
    r = run_one_seed(seed)
    if r is None:
        print('  FAILED (empty split)')
        continue
    print(f'  per-context F1 = {r["pc_f1"]}')
    print(f'  baseline F1    = {r["baseline_f1"]}')
    print(f'  gain           = {r["gain"]:+.3f}')
    rows.append(r)

df = pd.DataFrame(rows)
print('\n\n=== MAIN EXPERIMENT SUMMARY ===')
print(df.to_string(index=False))

print('\n=== AGGREGATE (5 seeds) ===')
print(f'Per-context F1:  {df["pc_f1"].mean():.3f} ± {df["pc_f1"].std():.3f}')
print(f'Baseline F1:     {df["baseline_f1"].mean():.3f} ± {df["baseline_f1"].std():.3f}')
print(f'Gain:            {df["gain"].mean():+.3f} ± {df["gain"].std():.3f}')

Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/main_experiment_percontext.csv', index=False)
print('\nSaved to docs/thesis/figures/main_experiment_percontext.csv')

mean_gain = df['gain'].mean()
std_gain = df['gain'].std()
print(f'\n95%-CI of gain: [{mean_gain - 2*std_gain:+.3f}, {mean_gain + 2*std_gain:+.3f}]')
if mean_gain - 2*std_gain > 0:
    print('=> Per-context method significantly BETTER than global baseline')
elif mean_gain + 2*std_gain < 0:
    print('=> Per-context method significantly WORSE than global baseline')
else:
    print('=> No significant difference (CI crosses zero)')
