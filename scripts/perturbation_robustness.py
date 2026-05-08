"""Perturbation analysis: насколько устойчивы методы к шуму и обрезанию.

Три типа возмущений на тестовых сегментах:
  1. Гауссов аддитивный шум на мел-признаках (σ ∈ {0.1, 0.3, 0.5})
  2. Drop-out сегментов в вокализации (p_drop ∈ {0.1, 0.3, 0.5})
  3. Обрезание длинных вокализаций до первых N сегментов (N ∈ {1, 3, 5})

Сравниваются три метода по weighted F1:
  per-context k-means + Bayes (наш основной)
  Assom (G + RF)
  RF на агрегатах сырого мел (no tokens)

Чувствительность измеряется как ΔF1 при возмущении относительно чистого теста.
Цель — выявить, какой метод деградирует медленнее.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, time
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from scipy.special import logsumexp

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX2IDX = {c: i for i, c in enumerate(HP1_CTX)}
SEEDS = list(range(5))

print('Loading state...')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
mel = st['tf_specs'].reshape(len(seg_df), -1).astype(np.float32)  # (N, 672)
emb_8d = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
hdb_nca = np.load(CACHE / 'hdb_nca_labels_152k_21x32.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy()
pos_arr = seg_df['pos_segment'].to_numpy()


def split_by_emitter(seed):
    rng = np.random.default_rng(seed)
    em = sorted(set(em_arr[em_arr != 0].tolist()))
    em = np.array(em); rng.shuffle(em)
    test_em = set(em[:11].tolist())
    return test_em


# Vocalizations
def build_vocs():
    df = pd.DataFrame({'idx': np.arange(len(seg_df)), 'file': file_arr,
                       'pos': pos_arr, 'ctx': ctx, 'em': em_arr})
    df = df[df['em'] != 0]
    df = df[df['ctx'].isin(HP1_CTX)]
    vocs = []
    for fname, g in df.sort_values('pos').groupby('file', sort=False):
        idxs = g['idx'].to_numpy()
        if len(idxs) < 1: continue
        vocs.append({
            'idxs': idxs,
            'ctx': CTX2IDX[int(np.bincount(g['ctx'].to_numpy()).argmax())],
            'em': int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0])
        })
    return vocs


vocs = build_vocs()
print(f'  vocs: {len(vocs)}')


def perturb_mel(features, sigma, rng):
    """Add Gaussian noise."""
    return features + rng.normal(0, sigma, features.shape).astype(np.float32)


def perturb_drop(idxs, p_drop, rng):
    """Drop random segments."""
    keep = rng.random(len(idxs)) >= p_drop
    if keep.sum() == 0:
        keep[rng.integers(len(idxs))] = True
    return idxs[keep]


def perturb_truncate(idxs, n_max):
    """Keep first n_max segments."""
    return idxs[:max(1, min(n_max, len(idxs)))]


# --- per-context Bayes (наш метод) ---
def fit_per_context_bayes(train_vocs, K=15):
    models = {}
    for c in range(len(HP1_CTX)):
        idxs = np.concatenate([v['idxs'] for v in train_vocs if v['ctx'] == c])
        if len(idxs) < 50: continue
        bgm = KMeans(n_clusters=min(K, len(idxs)//5), n_init=10, random_state=0).fit(emb_8d[idxs])
        # Gaussian per cluster (diag cov for speed)
        labels = bgm.labels_
        K_eff = bgm.n_clusters
        mu = np.array([emb_8d[idxs[labels == k]].mean(0) for k in range(K_eff)])
        var = np.array([emb_8d[idxs[labels == k]].var(0) + 1e-3 for k in range(K_eff)])
        pi = np.array([(labels == k).sum() / len(labels) for k in range(K_eff)])
        models[c] = {'mu': mu, 'var': var, 'pi': pi}
    return models


def voc_logp_under_c(idxs_pert_emb, model_c):
    """Sum_i log sum_k pi_k N(x_i | mu_k, diag(var_k))."""
    # idxs_pert_emb: (n_seg, 8)
    mu, var, pi = model_c['mu'], model_c['var'], model_c['pi']
    # log N(x | mu, diag(var)) = -0.5 sum [log(2pi*var) + (x-mu)^2/var]
    diff = idxs_pert_emb[:, None, :] - mu[None, :, :]  # (n, K, d)
    log_pdf = -0.5 * (np.log(2 * np.pi * var)[None, :, :] +
                       diff ** 2 / var[None, :, :]).sum(axis=2)
    log_p = logsumexp(log_pdf + np.log(pi)[None, :], axis=1)  # (n,)
    return log_p.sum()


def predict_pc_bayes(test_vocs_with_emb, models, prior):
    pred = []
    for v in test_vocs_with_emb:
        logp = []
        for c in range(len(HP1_CTX)):
            if c not in models: logp.append(-1e9); continue
            logp.append(voc_logp_under_c(v['emb'], models[c]) + np.log(prior[c] + 1e-9))
        pred.append(int(np.argmax(logp)))
    return np.array(pred)


# --- Assom (G + RF) ---
def assom_features(idxs, hdb_arr, n_clust):
    toks = hdb_arr[idxs]
    toks = toks[toks >= 0]
    if len(toks) == 0:
        return np.zeros(n_clust + 4, dtype=np.float32)
    freq = np.bincount(toks, minlength=n_clust) / len(toks)
    ent = -np.sum(freq * np.log(freq + 1e-12))
    return np.concatenate([freq, [len(toks), ent, freq.max(),
                                    float(np.count_nonzero(freq))]]).astype(np.float32)


def fit_assom_rf(train_vocs):
    n_clust = int(hdb_nca.max()) + 1
    X = np.array([assom_features(v['idxs'], hdb_nca, n_clust) for v in train_vocs])
    y = np.array([v['ctx'] for v in train_vocs])
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                 random_state=0, n_jobs=-1).fit(X, y)
    return rf, n_clust


def predict_assom_rf(rf, test_vocs, n_clust):
    X = np.array([assom_features(v['idxs'], hdb_nca, n_clust) for v in test_vocs])
    return rf.predict(X)


# --- RF on raw mel agg (no tokens) ---
def fit_rf_mel(train_vocs):
    X = np.array([np.concatenate([mel[v['idxs']].mean(0), mel[v['idxs']].std(0)])
                   for v in train_vocs])
    y = np.array([v['ctx'] for v in train_vocs])
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                 random_state=0, n_jobs=-1).fit(X, y)
    return rf


def predict_rf_mel(rf, test_vocs_pert):
    X = np.array([np.concatenate([m_pert.mean(0), m_pert.std(0)])
                   for m_pert in test_vocs_pert])
    return rf.predict(X)


PERTURBATIONS = [
    ('clean', 'none', 0),
    ('noise', 'gauss', 0.1),
    ('noise', 'gauss', 0.3),
    ('noise', 'gauss', 0.5),
    ('drop', 'p_drop', 0.1),
    ('drop', 'p_drop', 0.3),
    ('drop', 'p_drop', 0.5),
    ('trunc', 'n_max', 1),
    ('trunc', 'n_max', 3),
    ('trunc', 'n_max', 5),
]


print('\n=== Perturbation analysis ===')
results = []
for seed in SEEDS:
    test_em = split_by_emitter(seed)
    train = [v for v in vocs if v['em'] not in test_em]
    test = [v for v in vocs if v['em'] in test_em]
    print(f'\n--- seed {seed}: train={len(train)}, test={len(test)} ---', flush=True)

    # Fit models on clean train
    pc_models = fit_per_context_bayes(train)
    counts = np.bincount([v['ctx'] for v in train], minlength=len(HP1_CTX))
    pc_prior = counts / counts.sum()
    rf_assom, n_clust_assom = fit_assom_rf(train)
    rf_mel = fit_rf_mel(train)

    y_test = np.array([v['ctx'] for v in test])

    for kind, pname, pval in PERTURBATIONS:
        rng = np.random.default_rng(seed * 100 + hash((kind, pname, pval)) % 1000)
        # Construct perturbed test
        pert_emb = []  # for PC Bayes
        pert_idxs = []  # for Assom (still indices, but maybe truncated/dropped)
        pert_mel = []  # for RF mel
        for v in test:
            idxs = v['idxs']
            if kind == 'drop':
                idxs = perturb_drop(idxs, pval, rng)
            elif kind == 'trunc':
                idxs = perturb_truncate(idxs, pval)
            # For 'noise': mel and emb get noise; idxs unchanged
            if kind == 'noise':
                e = perturb_mel(emb_8d[idxs], pval, rng)
                m = perturb_mel(mel[idxs], pval, rng)
            else:
                e = emb_8d[idxs]; m = mel[idxs]
            pert_emb.append({'emb': e, 'ctx': v['ctx']})
            pert_idxs.append({'idxs': idxs})
            pert_mel.append(m)

        pred_pc = predict_pc_bayes(pert_emb, pc_models, pc_prior)
        pred_assom = predict_assom_rf(rf_assom, pert_idxs, n_clust_assom)
        pred_mel = predict_rf_mel(rf_mel, pert_mel)

        f1w_pc = f1_score(y_test, pred_pc, average='weighted', zero_division=0)
        f1w_assom = f1_score(y_test, pred_assom, average='weighted', zero_division=0)
        f1w_mel = f1_score(y_test, pred_mel, average='weighted', zero_division=0)
        for method, f1w in [('pc-bayes', f1w_pc), ('assom-rf', f1w_assom), ('rf-mel', f1w_mel)]:
            results.append({'seed': seed, 'kind': kind, 'param': pname,
                            'value': pval, 'method': method, 'f1_w': f1w})
        print(f'  {kind}({pname}={pval}): pc={f1w_pc:.3f}, assom={f1w_assom:.3f}, mel={f1w_mel:.3f}',
              flush=True)


df = pd.DataFrame(results)
df.to_csv('docs/thesis/figures/perturbation_results.csv', index=False)
agg = df.groupby(['kind', 'value', 'method']).agg(f1w_mean=('f1_w', 'mean'),
                                                    f1w_std=('f1_w', 'std')).reset_index()
agg.to_csv('docs/thesis/figures/perturbation_summary.csv', index=False)
print('\n=== Summary ===')
print(agg.to_string(index=False))
print('\nSaved: docs/thesis/figures/perturbation_*.csv')
