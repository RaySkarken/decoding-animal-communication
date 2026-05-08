"""Main per-context experiment re-run on FULL corpus (127k, all 82 emitter IDs).

Mirrors scripts/experiment_percontext_variants.py but reads
ablation_state_fullcorpus.joblib instead of ablation_state.joblib.

Key differences:
- tf_specs: (127040, 6, 32); UMAP embedding loaded from cached .npy
- Cross-bat emitter grouping by |Emitter| (–215 and 215 are same physical bat)
- Global HDBSCAN baseline config picked from sweep: frac=0.010, ms=20, eps=0.05
  (8 clusters, silhouette 0.697 on full corpus)
- Train/test split: 30/11 physical bats, same protocol as original experiment

Output: docs/thesis/figures/percontext_variants_fullcorpus.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, joblib
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
STATE_PATH = CKPT / 'ablation_state_fullcorpus.joblib'
UMAP_PATH = CKPT / 'umap_fullcorpus_nn30_md0.3.npy'
HDB_GLOBAL_LABELS_PATH = CKPT / 'hdb_global_labels_fullcorpus.npy'

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAME = {2: 'Biting', 3: 'Feeding', 4: 'Fighting', 5: 'Grooming',
            6: 'Isolation', 7: 'Kissing', 9: 'Mating', 10: 'Threat'}

# Global HDBSCAN config for baseline (chosen from reproduction sweep)
HDB_FRAC = 0.010
HDB_MS = 20
HDB_EPS = 0.05

N_TEST_BATS = 11
N_TRAIN_BATS = 30
N_SEEDS = 5


print(f'[1/5] Loading fullcorpus state...')
st = joblib.load(STATE_PATH)
seg_df = st['seg_df']
tf_specs = st['tf_specs']
emb = np.load(UMAP_PATH)
ctx = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()
em_abs = np.abs(emitters)

print(f'  N segments: {len(seg_df)}')
print(f'  tf_specs: {tf_specs.shape}  |  UMAP emb: {emb.shape}')
print(f'  unique |emitters| (physical bats): {len(set(em_abs.tolist()))}')

# ----- Global HDBSCAN for baseline vocabulary (+ KNN reassignment) -----
if HDB_GLOBAL_LABELS_PATH.exists():
    print(f'[2/5] Loading cached global HDBSCAN labels...')
    hdb_nca = np.load(HDB_GLOBAL_LABELS_PATH)
else:
    print(f'[2/5] Fitting global HDBSCAN (frac={HDB_FRAC}, ms={HDB_MS}, eps={HDB_EPS})...')
    import hdbscan as hdbscan_lib
    mcs = max(10, int(HDB_FRAC * len(emb)))
    hdb = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=HDB_MS,
                               cluster_selection_epsilon=HDB_EPS,
                               cluster_selection_method='leaf',
                               metric='euclidean', core_dist_n_jobs=-1).fit(emb)
    raw = hdb.labels_
    n_cl = len(set(raw)) - (1 if -1 in raw else 0)
    print(f'  raw: {n_cl} clusters, {(raw==-1).sum()/len(raw):.1%} noise')
    # KNN reassignment of noise
    nn = raw >= 0
    knn = KNeighborsClassifier(n_neighbors=30, weights='uniform', n_jobs=-1)
    knn.fit(emb[nn], raw[nn])
    hdb_nca = raw.copy()
    hdb_nca[~nn] = knn.predict(emb[~nn])
    np.save(HDB_GLOBAL_LABELS_PATH, hdb_nca)
    print(f'  saved global HDBSCAN+KNN labels: {HDB_GLOBAL_LABELS_PATH}')

print(f'  global vocabulary size: {len(set(hdb_nca.tolist()))}')

# ----- Group segments into vocalizations (by file_name) -----
print(f'\n[3/5] Grouping segments into vocalizations...')
vocs = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids: continue
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    dom_em = int(Counter(np.abs(g['emitter'].to_numpy())).most_common(1)[0][0])
    if dom_ctx not in HP1_CTX: continue
    vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em})

all_bats = sorted(set(v['em'] for v in vocs))
print(f'  total vocalizations: {len(vocs)}')
print(f'  unique bats (|ID|): {len(all_bats)}')


# ----- Gaussian mixture log-likelihood helpers -----
def _gauss_logpdf(X, mu, Sigma):
    D = X.shape[1]
    Sigma = Sigma + 1e-6 * np.eye(D)
    _, logdet = np.linalg.slogdet(Sigma)
    inv = np.linalg.inv(Sigma)
    diff = X - mu
    mahal = np.einsum('ij,jk,ik->i', diff, inv, diff)
    return -0.5 * (D * np.log(2 * np.pi) + logdet + mahal)

def _logmix(comps, X):
    if not comps: return np.full(len(X), -1e10)
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
        max_iter=150, random_state=seed).fit(X)
    return bgm
def score_dpgmm(tok, X): return tok.score_samples(X)


# ----- Tokenizer B: HDBSCAN + per-cluster gaussian -----
def fit_hdbscan_tok(X, seed):
    mcs = max(20, int(0.02 * len(X)))
    hdb = HDBSCAN(min_cluster_size=mcs, min_samples=10,
                   cluster_selection_epsilon=0.05).fit(X)
    comps = []
    for k in sorted(set(hdb.labels_)):
        if k < 0: continue
        m = hdb.labels_ == k
        if m.sum() < 5: continue
        Xi = X[m]
        mu = Xi.mean(axis=0)
        Sigma = np.cov(Xi.T) if Xi.shape[0] > 1 else np.eye(X.shape[1]) * 0.1
        comps.append((m.sum() / len(X), mu, Sigma))
    if not comps:
        mu = X.mean(axis=0)
        Sigma = np.cov(X.T) if X.shape[0] > 1 else np.eye(X.shape[1]) * 0.1
        comps.append((1.0, mu, Sigma))
    return comps
def score_hdbscan_tok(comps, X): return _logmix(comps, X)


# ----- Tokenizer C: k-means + per-cluster gaussian -----
def fit_kmeans_tok(X, seed, K=15):
    K = min(K, max(2, len(X) // 10))
    km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X)
    comps = []
    for k in range(K):
        m = km.labels_ == k
        if m.sum() < 2: continue
        Xi = X[m]
        mu = Xi.mean(axis=0)
        Sigma = np.cov(Xi.T) if Xi.shape[0] > 1 else np.eye(X.shape[1]) * 0.1
        comps.append((m.sum() / len(X), mu, Sigma))
    return comps
def score_kmeans_tok(comps, X): return _logmix(comps, X)


# ----- Classification -----
def classify_pc(tokenizers, log_prior, test_vocs, score_fn):
    y_true, y_pred = [], []
    for v in test_vocs:
        X_seq = emb[v['seg_ids']]
        if len(X_seq) == 0: continue
        best_c, best = None, -np.inf
        for c, tok in tokenizers.items():
            ll = score_fn(tok, X_seq).sum()
            score = ll + log_prior[c]
            if score > best: best = score; best_c = c
        if best_c is None: continue
        y_true.append(v['ctx']); y_pred.append(best_c)
    return np.array(y_true), np.array(y_pred)


# ----- Baseline: global HDBSCAN + RF on bag-of-syllables -----
def seq_features(seq, V):
    c = Counter(seq)
    n = len(seq)
    bos = np.zeros(V, dtype=np.float32)
    for k, cnt in c.items():
        if 0 <= k < V: bos[k] = cnt / max(n, 1)
    richness = len(c) / max(n, 1)
    probs = np.array(list(c.values()), dtype=np.float32) / max(n, 1)
    ent = float(-(probs * np.log(probs + 1e-12)).sum())
    rep = max(c.values()) / max(n, 1) if c else 0.0
    return np.concatenate([bos, [n, richness, ent, rep]]).astype(np.float32)

def baseline_rf(train_vocs, test_vocs, seed):
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
    if not Xtr or not Xte:
        return np.array([]), np.array([])
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    return np.array(yte), rf.predict(Xte)


# ----- Main loop -----
def run_seed(seed):
    rng = np.random.default_rng(seed)
    bat_arr = np.array(all_bats)
    rng.shuffle(bat_arr)
    test_bats = set(bat_arr[:N_TEST_BATS].tolist())
    train_bats = set(bat_arr[N_TEST_BATS:N_TEST_BATS+N_TRAIN_BATS].tolist())
    train_vocs = [v for v in vocs if v['em'] in train_bats]
    test_vocs = [v for v in vocs if v['em'] in test_bats]
    if not train_vocs or not test_vocs: return None, None

    train_seg_mask = np.zeros(len(emb), dtype=bool)
    for v in train_vocs:
        train_seg_mask[v['seg_ids']] = True

    # vocalization-level prior p(c)
    n_tv = len(train_vocs)
    log_prior = {}
    for c in HP1_CTX:
        nc = sum(1 for v in train_vocs if v['ctx'] == c)
        log_prior[c] = np.log(max(nc, 1) / n_tv)

    tok_dp, tok_hdb, tok_km = {}, {}, {}
    for c in HP1_CTX:
        mask = train_seg_mask & (ctx == c)
        if mask.sum() < 30: continue
        X_c = emb[mask]
        tok_dp[c] = fit_dpgmm(X_c, seed)
        tok_hdb[c] = fit_hdbscan_tok(X_c, seed)
        tok_km[c] = fit_kmeans_tok(X_c, seed)

    yt_dp, yp_dp = classify_pc(tok_dp, log_prior, test_vocs, score_dpgmm)
    yt_hdb, yp_hdb = classify_pc(tok_hdb, log_prior, test_vocs, score_hdbscan_tok)
    yt_km, yp_km = classify_pc(tok_km, log_prior, test_vocs, score_kmeans_tok)
    yt_bl, yp_bl = baseline_rf(train_vocs, test_vocs, seed)

    f = lambda y, p: round(f1_score(y, p, average='weighted',
                                      labels=HP1_CTX, zero_division=0), 3) if len(y) else 0.0
    r = {
        'seed': seed, 'n_train_vocs': len(train_vocs), 'n_test_vocs': len(test_vocs),
        'pc_dpgmm_f1': f(yt_dp, yp_dp),
        'pc_hdbscan_f1': f(yt_hdb, yp_hdb),
        'pc_kmeans_f1': f(yt_km, yp_km),
        'baseline_rf_f1': f(yt_bl, yp_bl),
    }
    r['gain_dpgmm'] = round(r['pc_dpgmm_f1'] - r['baseline_rf_f1'], 3)
    r['gain_hdbscan'] = round(r['pc_hdbscan_f1'] - r['baseline_rf_f1'], 3)
    r['gain_kmeans'] = round(r['pc_kmeans_f1'] - r['baseline_rf_f1'], 3)
    return r, (yt_dp, yp_dp, yt_bl, yp_bl)


print(f'\n[4/5] Running {N_SEEDS} seeds with {N_TRAIN_BATS}/{N_TEST_BATS} cross-bat split...')
rows = []
cm_seed0 = None
for s in range(N_SEEDS):
    print(f'\nseed {s}...', flush=True)
    r, cm_data = run_seed(s)
    if r is None: print('  FAIL'); continue
    rows.append(r)
    print(f'  n_train_vocs={r["n_train_vocs"]}, n_test_vocs={r["n_test_vocs"]}')
    print(f'  DP-GMM   F1={r["pc_dpgmm_f1"]}  (gain {r["gain_dpgmm"]:+.3f})')
    print(f'  HDBSCAN  F1={r["pc_hdbscan_f1"]}  (gain {r["gain_hdbscan"]:+.3f})')
    print(f'  k-means  F1={r["pc_kmeans_f1"]}  (gain {r["gain_kmeans"]:+.3f})')
    print(f'  baseline F1={r["baseline_rf_f1"]}')
    if s == 0: cm_seed0 = cm_data

df = pd.DataFrame(rows)
print('\n\n=== SUMMARY (5 seeds) ===')
print(df.to_string(index=False))

print('\n=== AGGREGATE (mean ± std) ===')
for col in ['pc_dpgmm_f1', 'pc_hdbscan_f1', 'pc_kmeans_f1', 'baseline_rf_f1']:
    print(f'  {col:22s} {df[col].mean():.3f} ± {df[col].std():.3f}')
for col in ['gain_dpgmm', 'gain_hdbscan', 'gain_kmeans']:
    m, s = df[col].mean(), df[col].std()
    ci_lo, ci_hi = m - 2*s, m + 2*s
    sig = 'SIG+' if ci_lo > 0 else ('SIG-' if ci_hi < 0 else 'NS')
    print(f'  {col:22s} {m:+.3f} ± {s:.3f}  95%-CI [{ci_lo:+.3f},{ci_hi:+.3f}]  {sig}')

print(f'\n[5/5] Saving...')
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/percontext_variants_fullcorpus.csv', index=False)
print(f'Saved: docs/thesis/figures/percontext_variants_fullcorpus.csv')

if cm_seed0 is not None:
    yt_dp, yp_dp, yt_bl, yp_bl = cm_seed0
    print('\n=== PER-CLASS F1 (seed 0) — full corpus ===')
    print('context       | DP-GMM | baseline | n_test')
    print('-' * 45)
    pc_rows = []
    for c in HP1_CTX:
        f_dp = f1_score(yt_dp == c, yp_dp == c, zero_division=0)
        f_bl = f1_score(yt_bl == c, yp_bl == c, zero_division=0) if len(yt_bl) else 0
        n = int((yt_dp == c).sum())
        print(f'{CTX_NAME[c]:13s} | {f_dp:.3f}  | {f_bl:.3f}    | {n}')
        pc_rows.append({'context': CTX_NAME[c], 'n_test': n,
                         'dpgmm_f1': round(f_dp, 3), 'baseline_f1': round(f_bl, 3)})
    pd.DataFrame(pc_rows).to_csv(
        'docs/thesis/figures/percontext_perclass_seed0_fullcorpus.csv', index=False)
