"""Diagnostics: why does per-context lose on full corpus?

Tests 3 hypotheses:
  A) Positive-ID-only subset of fullcorpus — isolate data-change from methodology change
  B) Baseline with 7-cluster vocab (matches paper reproducibility) — is 8-cluster win spurious?
  C) Uniform prior for per-context — is prior distribution skewing results?

Each runs 5 seeds, reports gain mean ± 95%-CI.
"""
from __future__ import annotations
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd, joblib
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans, HDBSCAN as sk_HDBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import hdbscan

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
STATE_PATH = CKPT / 'ablation_state_fullcorpus.joblib'
UMAP_PATH = CKPT / 'umap_fullcorpus_nn30_md0.3.npy'

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]


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


def run_experiment(seg_df, emb, hdb_nca, *, name, seeds=(0,1,2,3,4), uniform_prior=False):
    ctx = seg_df['context'].to_numpy()
    emitters = seg_df['emitter'].to_numpy()
    em_abs = np.abs(emitters)

    vocs = []
    for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
        seg_ids = g.index.to_list()
        if not seg_ids: continue
        dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
        dom_em = int(Counter(np.abs(g['emitter'].to_numpy())).most_common(1)[0][0])
        if dom_ctx not in HP1_CTX: continue
        vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em})
    all_bats = sorted(set(v['em'] for v in vocs))
    print(f'  N vocs: {len(vocs)}, N bats: {len(all_bats)}')

    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        ba = np.array(all_bats); rng.shuffle(ba)
        test_bats = set(ba[:11].tolist())
        train_bats = set(ba[11:41].tolist())
        train_vocs = [v for v in vocs if v['em'] in train_bats]
        test_vocs = [v for v in vocs if v['em'] in test_bats]
        if not train_vocs or not test_vocs: continue

        train_seg_mask = np.zeros(len(emb), dtype=bool)
        for v in train_vocs: train_seg_mask[v['seg_ids']] = True

        n_tv = len(train_vocs)
        log_prior = {}
        for c in HP1_CTX:
            if uniform_prior:
                log_prior[c] = 0.0
            else:
                nc = sum(1 for v in train_vocs if v['ctx'] == c)
                log_prior[c] = np.log(max(nc, 1) / n_tv)

        # DP-GMM tokenizers per context
        tok = {}
        for c in HP1_CTX:
            mask = train_seg_mask & (ctx == c)
            if mask.sum() < 30: continue
            X_c = emb[mask]
            bgm = BayesianGaussianMixture(
                n_components=15, weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.1, covariance_type='full',
                max_iter=150, random_state=s).fit(X_c)
            tok[c] = bgm

        # classify
        yt, yp = [], []
        for v in test_vocs:
            X_seq = emb[v['seg_ids']]
            if len(X_seq) == 0: continue
            best_c, best = None, -np.inf
            for c, bgm in tok.items():
                ll = bgm.score_samples(X_seq).sum() + log_prior[c]
                if ll > best: best = ll; best_c = c
            if best_c is None: continue
            yt.append(v['ctx']); yp.append(best_c)
        pc_f1 = f1_score(yt, yp, average='weighted', labels=HP1_CTX, zero_division=0)

        # baseline
        V = int(np.max(hdb_nca)) + 1
        Xtr, ytr = [], []
        for v in train_vocs:
            labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
            if labs:
                Xtr.append(seq_features(labs, V)); ytr.append(v['ctx'])
        Xte, yte = [], []
        for v in test_vocs:
            labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
            if labs:
                Xte.append(seq_features(labs, V)); yte.append(v['ctx'])
        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                      random_state=s, n_jobs=-1).fit(Xtr, ytr)
        bl_f1 = f1_score(yte, rf.predict(Xte), average='weighted',
                          labels=HP1_CTX, zero_division=0)

        rows.append({'seed': s, 'pc_f1': pc_f1, 'bl_f1': bl_f1, 'gain': pc_f1 - bl_f1})
    df = pd.DataFrame(rows)
    m, sd = df.gain.mean(), df.gain.std()
    ci_lo, ci_hi = m - 2*sd, m + 2*sd
    sig = 'SIG+' if ci_lo > 0 else ('SIG-' if ci_hi < 0 else 'NS')
    print(f'  {name}: pc={df.pc_f1.mean():.3f}±{df.pc_f1.std():.3f}, '
          f'bl={df.bl_f1.mean():.3f}±{df.bl_f1.std():.3f}, '
          f'gain={m:+.3f}±{sd:.3f} CI[{ci_lo:+.3f},{ci_hi:+.3f}] {sig}')
    return df


print('Loading full-corpus state...')
st = joblib.load(STATE_PATH)
seg_df_full = st['seg_df'].reset_index(drop=True)
emb_full = np.load(UMAP_PATH)

# Precompute global HDBSCAN labels for 8-cluster vocab (already cached)
hdb_nca_8 = np.load(CKPT / 'hdb_global_labels_fullcorpus.npy')

# Precompute 7-cluster vocab (from reproduction: frac=0.014, ms=20, eps=0.05)
CACHE_7 = CKPT / 'hdb_global_labels_fullcorpus_7c.npy'
if CACHE_7.exists():
    hdb_nca_7 = np.load(CACHE_7)
    print(f'Loaded 7-cluster cached vocab: {len(set(hdb_nca_7.tolist()))} unique')
else:
    print('Fitting 7-cluster HDBSCAN (frac=0.014, ms=20, eps=0.05) + KNN reassign...')
    mcs = max(10, int(0.014 * len(emb_full)))
    h = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=20,
                        cluster_selection_epsilon=0.05,
                        cluster_selection_method='leaf',
                        metric='euclidean', core_dist_n_jobs=-1).fit(emb_full)
    raw = h.labels_
    nn = raw >= 0
    knn = KNeighborsClassifier(n_neighbors=30, weights='uniform', n_jobs=-1).fit(emb_full[nn], raw[nn])
    hdb_nca_7 = raw.copy(); hdb_nca_7[~nn] = knn.predict(emb_full[~nn])
    np.save(CACHE_7, hdb_nca_7)
    print(f'Saved: {CACHE_7}')

# ====================================================================
# A) Full corpus, 8-cluster vocab, empirical prior — baseline
print('\n=== [A] Full corpus | 8-cluster baseline | empirical prior ===')
df_A = run_experiment(seg_df_full, emb_full, hdb_nca_8, name='A', uniform_prior=False)

# B) Full corpus, 7-cluster vocab, empirical prior
print('\n=== [B] Full corpus | 7-cluster baseline | empirical prior ===')
df_B = run_experiment(seg_df_full, emb_full, hdb_nca_7, name='B', uniform_prior=False)

# C) Full corpus, 8-cluster vocab, UNIFORM prior
print('\n=== [C] Full corpus | 8-cluster baseline | uniform prior ===')
df_C = run_experiment(seg_df_full, emb_full, hdb_nca_8, name='C', uniform_prior=True)

# D) POSITIVE-ID SUBSET of full corpus — isolate data vs methodology change
print('\n=== [D] Positive-ID subset of fullcorpus | 8-cluster vocab | empirical prior ===')
pos_mask = seg_df_full['emitter'].values > 0
seg_pos = seg_df_full[pos_mask].reset_index(drop=True)
emb_pos = emb_full[pos_mask]
hdb_pos = hdb_nca_8[pos_mask]
print(f'  positive subset: {len(seg_pos)} segs ({pos_mask.mean():.1%} of full)')
df_D = run_experiment(seg_pos, emb_pos, hdb_pos, name='D', uniform_prior=False)

summary = pd.DataFrame({
    'A_fullcorpus_8c_emp': df_A.gain.tolist(),
    'B_fullcorpus_7c_emp': df_B.gain.tolist(),
    'C_fullcorpus_8c_unif': df_C.gain.tolist(),
    'D_positive_8c_emp': df_D.gain.tolist(),
})
summary.to_csv('docs/thesis/figures/percontext_diagnose_fullcorpus.csv', index=False)
print(f'\nSaved: docs/thesis/figures/percontext_diagnose_fullcorpus.csv')
print('\n=== SUMMARY TABLE ===')
print(summary.describe().round(3).to_string())
