"""Per-context syllable consistency analysis on 153k corpus.

For each of 5 seeds, fit per-context DP-GMM on train. Then assign syllable IDs
to ALL segments using ORACLE context (true label of segment) → argmax over
that context's components. Compute:
  - silhouette in UMAP space
  - per-emitter ARI/NMI vs DTW-MFCC proxy

Compare against global HDBSCAN baseline labels (already computed, 11 clusters).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, joblib
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans, HDBSCAN as sk_HDBSCAN
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                              normalized_mutual_info_score)

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
print('Loading state...')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0.npy')
hdb_nca = np.load(CACHE / 'hdb_nca_labels_152k_21x32.npy')
proxy = np.load(CACHE / 'proxy_label_152k_21x32.npy')
ctx = seg_df['context'].to_numpy()
em_abs = np.abs(seg_df['emitter'].to_numpy())
em_arr = seg_df['emitter'].to_numpy()
HP1_CTX = [2,3,4,5,6,7,9,10]

# ── helpers ────────────────────────────────────────────────────────────────
def _gauss_logpdf(X, mu, Sigma):
    D = X.shape[1]
    Sigma = Sigma + 1e-6 * np.eye(D)
    _, logdet = np.linalg.slogdet(Sigma)
    inv = np.linalg.inv(Sigma)
    diff = X - mu
    mahal = np.einsum('ij,jk,ik->i', diff, inv, diff)
    return -0.5 * (D * np.log(2*np.pi) + logdet + mahal)


def assign_dpgmm(X, bgm):
    """Component argmax for DP-GMM (already fitted)."""
    return bgm.predict(X)


def assign_comps(X, comps):
    """Component argmax from list of (pi, mu, Sigma)."""
    if not comps: return np.zeros(len(X), dtype=int)
    logs = np.stack([np.log(p+1e-12) + _gauss_logpdf(X, mu, S) for (p, mu, S) in comps], axis=1)
    return np.argmax(logs, axis=1)


def fit_dpgmm(X, seed, k=15):
    return BayesianGaussianMixture(
        n_components=k, weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1, covariance_type='full',
        max_iter=150, random_state=seed).fit(X)


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
        comps.append((m.sum()/len(X), mu, Sigma))
    if not comps:
        mu = X.mean(axis=0)
        Sigma = np.cov(X.T) if X.shape[0] > 1 else np.eye(X.shape[1]) * 0.1
        comps.append((1.0, mu, Sigma))
    return comps


def fit_kmeans_tok(X, seed, K=15):
    K = min(K, max(2, len(X)//10))
    km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X)
    comps = []
    for k in range(K):
        m = km.labels_ == k
        if m.sum() < 2: continue
        Xi = X[m]
        mu = Xi.mean(axis=0)
        Sigma = np.cov(Xi.T) if Xi.shape[0] > 1 else np.eye(X.shape[1])*0.1
        comps.append((m.sum()/len(X), mu, Sigma))
    return comps


# Group vocs to identify train bats per seed
vocs = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids: continue
    dom_em_signed = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
    if dom_em_signed == 0: continue
    dom_em_abs = abs(dom_em_signed)
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    if dom_ctx not in HP1_CTX: continue
    vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em_abs})
all_bats = sorted(set(v['em'] for v in vocs))
print(f'Vocs: {len(vocs)}, bats: {len(all_bats)}')


def peremit_ari_nmi(lbl):
    aris, nmis = [], []
    for b in sorted(set(em_abs[em_arr != 0].tolist())):
        m = (em_abs == b) & (proxy >= 0)
        if m.sum() < 30: continue
        aris.append(adjusted_rand_score(proxy[m], lbl[m]))
        nmis.append(normalized_mutual_info_score(proxy[m], lbl[m]))
    if not aris: return np.nan, np.nan, np.nan, np.nan, 0
    return np.mean(aris), np.std(aris), np.mean(nmis), np.std(nmis), len(aris)


def silh(lbl, sample=20000):
    nn = lbl >= 0
    if nn.sum() < 100: return np.nan
    rng = np.random.default_rng(0)
    ix = rng.choice(np.where(nn)[0], min(sample, nn.sum()), replace=False)
    if len(set(lbl[ix].tolist())) < 2: return np.nan
    return silhouette_score(emb[ix], lbl[ix])


# Baseline reference
print('\n=== BASELINE: global HDBSCAN labels ===')
sb = silh(hdb_nca)
ari_m, ari_s, nmi_m, nmi_s, n_em = peremit_ari_nmi(hdb_nca)
print(f'  silhouette = {sb:.3f}')
print(f'  per-emitter ARI = {ari_m:.3f} ± {ari_s:.3f}, NMI = {nmi_m:.3f} ± {nmi_s:.3f}, n_bats={n_em}')

# Per-context tokenizers — assign syllable IDs via ORACLE context
SEEDS = [0, 1, 2, 3, 4]
N_TEST = 11

results = []
for seed in SEEDS:
    print(f'\n=== seed {seed} ===')
    rng = np.random.default_rng(seed)
    bat_arr = np.array(all_bats); rng.shuffle(bat_arr)
    test_bats = set(bat_arr[:N_TEST].tolist())
    train_bats = set(bat_arr[N_TEST:N_TEST+30].tolist())
    train_vocs = [v for v in vocs if v['em'] in train_bats]
    train_seg_mask = np.zeros(len(emb), dtype=bool)
    for v in train_vocs: train_seg_mask[v['seg_ids']] = True

    # Fit each tokenizer on train segments per context
    tok_dp, tok_hdb, tok_km = {}, {}, {}
    for c in HP1_CTX:
        mask = train_seg_mask & (ctx == c)
        if mask.sum() < 30: continue
        X_c = emb[mask]
        tok_dp[c] = fit_dpgmm(X_c, seed)
        tok_hdb[c] = fit_hdbscan_tok(X_c, seed)
        tok_km[c] = fit_kmeans_tok(X_c, seed)

    # Assign syllable IDs to ALL segments using oracle context
    def oracle_assign(tokenizers, kind):
        labels = np.full(len(emb), -1, dtype=np.int32)
        # Encode (context, component) → unique global ID
        offset = 0
        ctx_to_offset = {}
        for c in sorted(tokenizers.keys()):
            ctx_to_offset[c] = offset
            if kind == 'dpgmm':
                # Find max component used by this tokenizer
                K = tokenizers[c].n_components
            else:
                K = len(tokenizers[c])
            offset += K
        for c, tok in tokenizers.items():
            mask = (ctx == c)
            if mask.sum() == 0: continue
            X = emb[mask]
            if kind == 'dpgmm':
                comp_id = assign_dpgmm(X, tok)
            else:
                comp_id = assign_comps(X, tok)
            labels[mask] = comp_id + ctx_to_offset[c]
        return labels

    lbl_dp = oracle_assign(tok_dp, 'dpgmm')
    lbl_hdb = oracle_assign(tok_hdb, 'comps')
    lbl_km = oracle_assign(tok_km, 'comps')

    for name, lbl in [('DP-GMM', lbl_dp), ('HDBSCAN-tok', lbl_hdb), ('k-means', lbl_km)]:
        n_unique = len(set(lbl[lbl >= 0].tolist()))
        s = silh(lbl)
        ari_m_, ari_s_, nmi_m_, nmi_s_, n_em_ = peremit_ari_nmi(lbl)
        print(f'  {name:12s}: vocab={n_unique}, sil={s:.3f}, '
              f'ARI={ari_m_:.3f}±{ari_s_:.3f}, NMI={nmi_m_:.3f}±{nmi_s_:.3f}')
        results.append({
            'seed': seed, 'method': name,
            'vocab_size': n_unique,
            'silhouette': s,
            'ARI_pe_mean': ari_m_, 'ARI_pe_std': ari_s_,
            'NMI_pe_mean': nmi_m_, 'NMI_pe_std': nmi_s_,
        })

# Add baseline as reference rows
for seed in [0]:
    results.append({
        'seed': 'baseline', 'method': 'global HDBSCAN',
        'vocab_size': len(set(hdb_nca.tolist())),
        'silhouette': sb,
        'ARI_pe_mean': ari_m, 'ARI_pe_std': ari_s,
        'NMI_pe_mean': nmi_m, 'NMI_pe_std': nmi_s,
    })

df = pd.DataFrame(results)
print('\n=== AGGREGATE per method (5 seeds) ===')
agg = df[df.seed != 'baseline'].groupby('method').agg(
    vocab_size=('vocab_size', 'mean'),
    sil_mean=('silhouette', 'mean'),
    sil_std=('silhouette', 'std'),
    ARI=('ARI_pe_mean', 'mean'),
    ARI_std=('ARI_pe_mean', 'std'),
    NMI=('NMI_pe_mean', 'mean'),
    NMI_std=('NMI_pe_mean', 'std'),
).round(3)
print(agg.to_string())

print(f'\nBaseline (global HDBSCAN): vocab={len(set(hdb_nca.tolist()))}, '
      f'sil={sb:.3f}, ARI={ari_m:.3f}, NMI={nmi_m:.3f}')

df.to_csv('docs/thesis/figures/percontext_consistency_152k.csv', index=False)
print('\nSaved: docs/thesis/figures/percontext_consistency_152k.csv')
