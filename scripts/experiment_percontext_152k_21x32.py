"""Main per-context experiment on 153k corpus (21x32 mel, paper-faithful state).

Baseline state: ablation_state_152k_21x32.joblib (153,366 sub-segments, 41 physical
bats + ~26k id=0 segments). For cross-bat protocol we exclude id=0 (cannot assign
to train/test by physical bat).

Pipeline (per-context):
  - Per-context DP-GMM / HDBSCAN-tok / k-means tokenizers fit on UMAP-2D
  - Bayes classification with vocalization-level empirical prior
  - 5 seeds × 30 train / 11 test physical bats
Baseline:
  - Global HDBSCAN labels (11-cluster, Assom defaults) → bag-of-syllables features
  - Random Forest on those features

Output: docs/thesis/figures/percontext_152k_21x32_results.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd, joblib
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans, HDBSCAN as sk_HDBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
STATE = CACHE / 'ablation_state_152k_21x32.joblib'
UMAP_PATH = CACHE / 'umap_152k_21x32_md1.0.npy'
HDB_NCA_PATH = CACHE / 'hdb_nca_labels_152k_21x32.npy'

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAME = {2:'Biting', 3:'Feeding', 4:'Fighting', 5:'Grooming',
            6:'Isolation', 7:'Kissing', 9:'Mating', 10:'Threat'}

N_TEST_BATS = 11
N_TRAIN_BATS = 30
N_SEEDS = 5


print('[1/4] Loading state...')
st = joblib.load(STATE)
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(UMAP_PATH)
hdb_nca = np.load(HDB_NCA_PATH)
ctx = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()
em_abs = np.abs(emitters)
print(f'  Total segments: {len(seg_df)}')
print(f'  embedding: {emb.shape}, HDBSCAN labels: {len(set(hdb_nca.tolist()))} syllables')
print(f'  unique |emitters| (physical bats): {len(set(em_abs[emitters != 0].tolist()))}')
print(f'  segments with id=0 (excluded for cross-bat): {(emitters==0).sum()}')


# ── Group segments into vocalizations (only those with id != 0) ─────────────
print(f'\n[2/4] Grouping into vocalizations...')
vocs = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids: continue
    dom_em_signed = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
    if dom_em_signed == 0: continue   # cannot use for cross-bat
    dom_em_abs = abs(dom_em_signed)
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    if dom_ctx not in HP1_CTX: continue
    vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em_abs})
all_bats = sorted(set(v['em'] for v in vocs))
print(f'  N vocs (HP1 contexts, identified bats): {len(vocs)}')
print(f'  N physical bats: {len(all_bats)}')
print(f'  contexts: {Counter(v["ctx"] for v in vocs)}')


# ── Tokenizer helpers ──────────────────────────────────────────────────────
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


def fit_dpgmm(X, seed, k=15):
    bgm = BayesianGaussianMixture(
        n_components=k, weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1, covariance_type='full',
        max_iter=150, random_state=seed).fit(X)
    return bgm
def score_dpgmm(tok, X): return tok.score_samples(X)


def fit_hdbscan_tok(X, seed):
    mcs = max(20, int(0.02 * len(X)))
    hdb = sk_HDBSCAN(min_cluster_size=mcs, min_samples=10,
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


def classify_pc(tokenizers, log_prior, test_vocs, score_fn):
    y_true, y_pred = [], []
    for v in test_vocs:
        X_seq = emb[v['seg_ids']]
        if len(X_seq) == 0: continue
        best_c, best = None, -np.inf
        for c, tok in tokenizers.items():
            ll = score_fn(tok, X_seq).sum() + log_prior[c]
            if ll > best: best = ll; best_c = c
        if best_c is None: continue
        y_true.append(v['ctx']); y_pred.append(best_c)
    return np.array(y_true), np.array(y_pred)


def seq_features(seq, V):
    c = Counter(seq); n = len(seq)
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
    if not Xtr or not Xte: return np.array([]), np.array([])
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                 random_state=seed, n_jobs=-1).fit(Xtr, ytr)
    return np.array(yte), rf.predict(Xte)


# ── Main loop ──────────────────────────────────────────────────────────────
def run_seed(seed):
    rng = np.random.default_rng(seed)
    bat_arr = np.array(all_bats); rng.shuffle(bat_arr)
    test_bats = set(bat_arr[:N_TEST_BATS].tolist())
    train_bats = set(bat_arr[N_TEST_BATS:N_TEST_BATS+N_TRAIN_BATS].tolist())
    train_vocs = [v for v in vocs if v['em'] in train_bats]
    test_vocs = [v for v in vocs if v['em'] in test_bats]
    if not train_vocs or not test_vocs: return None

    train_seg_mask = np.zeros(len(emb), dtype=bool)
    for v in train_vocs: train_seg_mask[v['seg_ids']] = True

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
    return {
        'seed': seed, 'n_train_vocs': len(train_vocs), 'n_test_vocs': len(test_vocs),
        'pc_dpgmm_f1': f(yt_dp, yp_dp),
        'pc_hdbscan_f1': f(yt_hdb, yp_hdb),
        'pc_kmeans_f1': f(yt_km, yp_km),
        'baseline_rf_f1': f(yt_bl, yp_bl),
        'gain_dpgmm': round(f(yt_dp, yp_dp) - f(yt_bl, yp_bl), 3),
        'gain_hdbscan': round(f(yt_hdb, yp_hdb) - f(yt_bl, yp_bl), 3),
        'gain_kmeans': round(f(yt_km, yp_km) - f(yt_bl, yp_bl), 3),
    }


print(f'\n[3/4] Running {N_SEEDS} seeds...')
rows = []
for s in range(N_SEEDS):
    print(f'\nseed {s}...', flush=True)
    r = run_seed(s)
    if r is None:
        print('  FAIL'); continue
    rows.append(r)
    print(f'  n_train={r["n_train_vocs"]}, n_test={r["n_test_vocs"]}')
    print(f'  DP-GMM   F1={r["pc_dpgmm_f1"]}  gain {r["gain_dpgmm"]:+.3f}')
    print(f'  HDBSCAN  F1={r["pc_hdbscan_f1"]}  gain {r["gain_hdbscan"]:+.3f}')
    print(f'  k-means  F1={r["pc_kmeans_f1"]}  gain {r["gain_kmeans"]:+.3f}')
    print(f'  baseline F1={r["baseline_rf_f1"]}')


df = pd.DataFrame(rows)
print('\n\n=== SUMMARY (5 seeds, 153k corpus, 21x32 mel) ===')
print(df.to_string(index=False))

print('\n=== AGGREGATE ===')
for col in ['pc_dpgmm_f1', 'pc_hdbscan_f1', 'pc_kmeans_f1', 'baseline_rf_f1']:
    print(f'  {col:22s} {df[col].mean():.3f} ± {df[col].std():.3f}')
for col in ['gain_dpgmm', 'gain_hdbscan', 'gain_kmeans']:
    m, s = df[col].mean(), df[col].std()
    ci_lo, ci_hi = m - 2*s, m + 2*s
    sig = 'SIG+' if ci_lo > 0 else ('SIG-' if ci_hi < 0 else 'NS')
    print(f'  {col:22s} {m:+.3f} ± {s:.3f}  95%-CI [{ci_lo:+.3f},{ci_hi:+.3f}]  {sig}')

print(f'\n[4/4] Saving...')
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/percontext_152k_21x32_results.csv', index=False)
print('Saved: docs/thesis/figures/percontext_152k_21x32_results.csv')
