"""Per-context classification: three tokenizer variants + confusion matrix.

Расширение главного эксперимента: проверяет, что positive gain не привязан
к конкретному инструменту построения словаря. Реализует три варианта
шага 3 из §2.2:

  A. DP-GMM-per-context          (как в главном эксперименте)
  B. HDBSCAN-per-context         (гауссова аппроксимация для правдоподобия)
  C. VQ (k-means)-per-context    (мягкое назначение по расстояниям)

Классификация во всех трёх случаях: argmax_c [log p_c(seq) + log p(c)].
Baseline: Assom global + Random Forest на bag-of-syllables признаках.

На seed=0 сохраняется confusion matrix варианта A для диагностики.
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
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
emb = st['embedding']
seg_df = st['seg_df']
ctx = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()
hdb_nca = st['hdb_nca_labels']

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAME = {2: 'Biting', 3: 'Feeding', 4: 'Fighting', 5: 'Grooming',
            6: 'Isolation', 7: 'Kissing', 9: 'Mating', 10: 'Threat'}

# group by file
vocs = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids:
        continue
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
    if dom_ctx not in HP1_CTX:
        continue
    vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em})

all_emitters = sorted(set(v['em'] for v in vocs))
print(f'Total vocalizations: {len(vocs)}, emitters: {len(all_emitters)}')


# ----- helpers for gaussian mixture scoring -----
def _gauss_logpdf(X, mu, Sigma):
    """log N(x|mu,Sigma) per row, with jitter for singular cov."""
    D = X.shape[1]
    Sigma = Sigma + 1e-6 * np.eye(D)
    sign, logdet = np.linalg.slogdet(Sigma)
    inv = np.linalg.inv(Sigma)
    diff = X - mu
    mahal = np.einsum('ij,jk,ik->i', diff, inv, diff)
    return -0.5 * (D * np.log(2 * np.pi) + logdet + mahal)


def _logmix(comps, X):
    """log sum_k pi_k N(x|mu_k, Sigma_k) for each row of X."""
    if not comps:
        return np.full(len(X), -1e10)
    logs = np.stack([np.log(p + 1e-12) + _gauss_logpdf(X, mu, S)
                     for (p, mu, S) in comps], axis=1)
    m = logs.max(axis=1)
    return m + np.log(np.exp(logs - m[:, None]).sum(axis=1))


# ----- Tokenizer A: DP-GMM -----
def fit_dpgmm(X, seed, k=15):
    bgm = BayesianGaussianMixture(
        n_components=k,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1, covariance_type='full',
        max_iter=150, random_state=seed)
    bgm.fit(X)
    return bgm

def score_dpgmm(tok, X):
    return tok.score_samples(X)


# ----- Tokenizer B: HDBSCAN + gaussian per-cluster approximation -----
def fit_hdbscan(X, seed, min_cluster_size=None):
    if min_cluster_size is None:
        min_cluster_size = max(20, int(0.02 * len(X)))
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10,
                   cluster_selection_epsilon=0.05).fit(X)
    comps = []
    for k in sorted(set(hdb.labels_)):
        if k < 0:
            continue
        m = hdb.labels_ == k
        if m.sum() < 5:
            continue
        Xi = X[m]
        mu = Xi.mean(axis=0)
        Sigma = np.cov(Xi.T) if Xi.shape[0] > 1 else np.eye(X.shape[1]) * 0.1
        comps.append((m.sum() / len(X), mu, Sigma))
    if not comps:
        mu = X.mean(axis=0)
        Sigma = np.cov(X.T) if X.shape[0] > 1 else np.eye(X.shape[1]) * 0.1
        comps.append((1.0, mu, Sigma))
    return comps

def score_hdbscan(comps, X):
    return _logmix(comps, X)


# ----- Tokenizer C: k-means with gaussian aprox -----
def fit_kmeans(X, seed, K=15):
    K = min(K, max(2, len(X) // 10))
    km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X)
    comps = []
    for k in range(K):
        m = km.labels_ == k
        if m.sum() < 2:
            continue
        Xi = X[m]
        mu = Xi.mean(axis=0)
        Sigma = np.cov(Xi.T) if Xi.shape[0] > 1 else np.eye(X.shape[1]) * 0.1
        comps.append((m.sum() / len(X), mu, Sigma))
    return comps

def score_kmeans(comps, X):
    return _logmix(comps, X)


# ----- Classification loop -----
def classify_percontext(tokenizers, log_prior, test_vocs, score_fn):
    y_true, y_pred = [], []
    for v in test_vocs:
        X_seq = emb[v['seg_ids']]
        if len(X_seq) == 0:
            continue
        best_c, best = None, -np.inf
        for c, tok in tokenizers.items():
            ll = score_fn(tok, X_seq).sum()
            score = ll + log_prior[c]
            if score > best:
                best = score
                best_c = c
        if best_c is None:
            continue
        y_true.append(v['ctx'])
        y_pred.append(best_c)
    return np.array(y_true), np.array(y_pred)


# ----- Baseline: global Assom + RF on bag-of-syllables -----
def seq_features(seq, V):
    c = Counter(seq)
    n = len(seq)
    bos = np.zeros(V, dtype=np.float32)
    for k, cnt in c.items():
        if 0 <= k < V:
            bos[k] = cnt / max(n, 1)
    richness = len(c) / max(n, 1)
    probs = np.array(list(c.values()), dtype=np.float32) / max(n, 1)
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    rep = max(c.values()) / max(n, 1) if c else 0.0
    return np.concatenate([bos, [n, richness, ent, rep]]).astype(np.float32)


def baseline_global_rf(train_vocs, test_vocs, seed):
    V = int(np.max(hdb_nca)) + 1
    Xtr, ytr = [], []
    for v in train_vocs:
        labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
        if not labs: continue
        Xtr.append(seq_features(labs, V)); ytr.append(v['ctx'])
    Xte, yte = [], []
    for v in test_vocs:
        labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
        if not labs: continue
        Xte.append(seq_features(labs, V)); yte.append(v['ctx'])
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    return np.array(yte), pred


# ----- Main loop -----
def run_seed(seed):
    rng = np.random.default_rng(seed)
    em_arr = np.array(all_emitters)
    rng.shuffle(em_arr)
    test_emitters = set(em_arr[:11].tolist())
    train_emitters = set(em_arr[11:].tolist())
    train_vocs = [v for v in vocs if v['em'] in train_emitters]
    test_vocs = [v for v in vocs if v['em'] in test_emitters]
    if not train_vocs or not test_vocs:
        return None, None
    train_seg_mask = np.zeros(len(emb), dtype=bool)
    for v in train_vocs:
        train_seg_mask[v['seg_ids']] = True

    n_tv = len(train_vocs)
    log_prior = {}
    for c in HP1_CTX:
        nc = sum(1 for v in train_vocs if v['ctx'] == c)
        log_prior[c] = np.log(max(nc, 1) / n_tv)

    # fit per-context variants
    tok_dp, tok_hdb, tok_km = {}, {}, {}
    for c in HP1_CTX:
        mask = train_seg_mask & (ctx == c)
        if mask.sum() < 30:
            continue
        X_c = emb[mask]
        tok_dp[c] = fit_dpgmm(X_c, seed)
        tok_hdb[c] = fit_hdbscan(X_c, seed)
        tok_km[c] = fit_kmeans(X_c, seed)

    yt_dp, yp_dp = classify_percontext(tok_dp, log_prior, test_vocs, score_dpgmm)
    yt_hdb, yp_hdb = classify_percontext(tok_hdb, log_prior, test_vocs, score_hdbscan)
    yt_km, yp_km = classify_percontext(tok_km, log_prior, test_vocs, score_kmeans)
    yt_bl, yp_bl = baseline_global_rf(train_vocs, test_vocs, seed)

    f = lambda y, p: round(f1_score(y, p, average='weighted',
                                      labels=HP1_CTX, zero_division=0), 3)
    r = {
        'seed': seed,
        'n_test': len(yt_dp),
        'pc_dpgmm_f1': f(yt_dp, yp_dp),
        'pc_hdbscan_f1': f(yt_hdb, yp_hdb),
        'pc_kmeans_f1': f(yt_km, yp_km),
        'baseline_rf_f1': f(yt_bl, yp_bl),
    }
    r['gain_dpgmm'] = round(r['pc_dpgmm_f1'] - r['baseline_rf_f1'], 3)
    r['gain_hdbscan'] = round(r['pc_hdbscan_f1'] - r['baseline_rf_f1'], 3)
    r['gain_kmeans'] = round(r['pc_kmeans_f1'] - r['baseline_rf_f1'], 3)
    return r, (yt_dp, yp_dp, yt_bl, yp_bl)


rows = []
cm_seed0 = None
for s in range(5):
    print(f'seed {s}...', flush=True)
    r, cm_data = run_seed(s)
    if r is None:
        continue
    rows.append(r)
    print(f'  DP-GMM   F1={r["pc_dpgmm_f1"]}  (gain {r["gain_dpgmm"]:+.3f})')
    print(f'  HDBSCAN  F1={r["pc_hdbscan_f1"]}  (gain {r["gain_hdbscan"]:+.3f})')
    print(f'  k-means  F1={r["pc_kmeans_f1"]}  (gain {r["gain_kmeans"]:+.3f})')
    print(f'  baseline F1={r["baseline_rf_f1"]}')
    if s == 0:
        cm_seed0 = cm_data

df = pd.DataFrame(rows)
print('\n\n=== PER-CONTEXT VARIANTS SUMMARY ===')
print(df.to_string(index=False))

print('\n=== AGGREGATE (5 seeds, mean ± std) ===')
for col in ['pc_dpgmm_f1', 'pc_hdbscan_f1', 'pc_kmeans_f1', 'baseline_rf_f1']:
    print(f'  {col:22s} {df[col].mean():.3f} ± {df[col].std():.3f}')
for col in ['gain_dpgmm', 'gain_hdbscan', 'gain_kmeans']:
    m, s = df[col].mean(), df[col].std()
    ci_lo, ci_hi = m - 2*s, m + 2*s
    sig = 'SIG+' if ci_lo > 0 else ('SIG-' if ci_hi < 0 else 'NS')
    print(f'  {col:22s} {m:+.3f} ± {s:.3f}  95%-CI [{ci_lo:+.3f},{ci_hi:+.3f}]  {sig}')

Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/percontext_variants.csv', index=False)
print('\nSaved to docs/thesis/figures/percontext_variants.csv')

# ----- per-class analysis seed 0 -----
if cm_seed0 is not None:
    yt_dp, yp_dp, yt_bl, yp_bl = cm_seed0
    print('\n\n=== PER-CLASS F1 (seed 0) ===')
    print('context       | DP-GMM | baseline')
    print('-' * 40)
    per_class_rows = []
    for c in HP1_CTX:
        f_dp = f1_score(yt_dp == c, yp_dp == c, zero_division=0)
        f_bl = f1_score(yt_bl == c, yp_bl == c, zero_division=0)
        n = (yt_dp == c).sum()
        print(f'{CTX_NAME[c]:13s} | {f_dp:.3f}  | {f_bl:.3f}   (n_test={n})')
        per_class_rows.append({
            'context': CTX_NAME[c], 'n_test': int(n),
            'dpgmm_f1': round(f_dp, 3), 'baseline_f1': round(f_bl, 3),
        })
    pd.DataFrame(per_class_rows).to_csv(
        'docs/thesis/figures/percontext_perclass_seed0.csv', index=False)

    print('\n=== CONFUSION MATRIX (DP-GMM, seed 0) — rows: true, cols: pred ===')
    cm = confusion_matrix(yt_dp, yp_dp, labels=HP1_CTX)
    header = 'true\\pred  | ' + '  '.join(f'{CTX_NAME[c][:4]:>5s}' for c in HP1_CTX)
    print(header)
    print('-' * len(header))
    for i, c in enumerate(HP1_CTX):
        row = f'{CTX_NAME[c]:10s} | ' + '  '.join(f'{cm[i,j]:>5d}' for j in range(len(HP1_CTX)))
        print(row)
    pd.DataFrame(cm, index=[CTX_NAME[c] for c in HP1_CTX],
                 columns=[CTX_NAME[c] for c in HP1_CTX]
                 ).to_csv('docs/thesis/figures/percontext_confusion_seed0.csv')
