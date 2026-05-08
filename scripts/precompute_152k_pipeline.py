"""Pre-compute UMAP + HDBSCAN + NCA + DTW-MFCC proxy for 152k state.

Populates caches so the notebook cells load instantly.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, joblib, librosa
import umap, hdbscan
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                              normalized_mutual_info_score)
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
STATE = CACHE / 'ablation_state_152k.joblib'
EMB   = CACHE / 'umap_152k_nn30_md0.3.npy'
HDB   = CACHE / 'hdb_labels_152k.npy'
NCA   = CACHE / 'hdb_nca_labels_152k.npy'
PROXY = CACHE / 'proxy_label_152k.npy'

# HDBSCAN config — picked from sweep on 127k corpus
HDB_FRAC, HDB_MS, HDB_EPS = 0.016, 20, 0.05   # tuned for 7 clusters on 153k

# Proxy config
N_PER_EM = 400
Q_PROXY  = 0.05
N_MFCC   = 13


def main():
    print(f'Loading {STATE.name}...')
    st = joblib.load(STATE)
    seg_df = st['seg_df']; tf_specs = st['tf_specs']
    print(f'  N segments: {len(seg_df)}, tf_specs: {tf_specs.shape}')

    # ── UMAP ──────────────────────────────────────────────────────────────────
    if EMB.exists():
        embedding = np.load(EMB)
        print(f'[1/4] UMAP: cached → {EMB.name}')
    else:
        print(f'[1/4] UMAP fit on {len(tf_specs)} points (n_neighbors=30, min_dist=0.3)...')
        X = tf_specs.reshape(len(tf_specs), -1)
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                            metric='euclidean', random_state=0)
        embedding = reducer.fit_transform(X).astype(np.float32)
        np.save(EMB, embedding)
        print(f'  saved: {EMB}')
    print(f'  embedding: {embedding.shape}')

    # ── HDBSCAN ──────────────────────────────────────────────────────────────
    if HDB.exists():
        labels = np.load(HDB)
        print(f'[2/4] HDBSCAN: cached → {HDB.name}')
    else:
        mcs = max(10, int(HDB_FRAC * len(embedding)))
        print(f'[2/4] HDBSCAN(min_cluster_size={mcs}, ms={HDB_MS}, eps={HDB_EPS})...')
        h = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=HDB_MS,
                            cluster_selection_epsilon=HDB_EPS,
                            cluster_selection_method='leaf',
                            metric='euclidean', core_dist_n_jobs=-1)
        labels = h.fit_predict(embedding).astype(int)
        np.save(HDB, labels)
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    nn = labels >= 0
    sil = silhouette_score(embedding[nn][:20000], labels[nn][:20000]) if nn.sum() > 100 else np.nan
    print(f'  clusters: {n_cl}, noise: {n_noise/len(labels):.1%}, silhouette: {sil:.3f}')

    # ── NCA noise reassign ───────────────────────────────────────────────────
    if NCA.exists():
        nca = np.load(NCA)
        print(f'[3/4] NCA: cached → {NCA.name}')
    else:
        print(f'[3/4] KNN(k=30) noise reassign...')
        nn_mask = labels >= 0
        knn = KNeighborsClassifier(n_neighbors=30, weights='uniform', n_jobs=-1)
        knn.fit(embedding[nn_mask], labels[nn_mask])
        nca = labels.copy(); nca[~nn_mask] = knn.predict(embedding[~nn_mask])
        np.save(NCA, nca)
    print(f'  reassigned. global vocabulary: {len(set(nca.tolist()))} syllables')

    # ── DTW-MFCC qt_ward proxy on identified bats ────────────────────────────
    em = seg_df['emitter'].to_numpy()
    em_abs = np.abs(em)
    bats = sorted(set(em_abs[em != 0].tolist()))
    print(f'[4/4] DTW-MFCC qt_ward proxy on {len(bats)} identified bats...')

    if PROXY.exists():
        proxy = np.load(PROXY)
        print(f'  cached → {PROXY.name}')
    else:
        rng = np.random.default_rng(0)
        proxy = np.full(len(seg_df), -1, dtype=np.int32)
        offset = 0
        per_em_stats = []
        for b in tqdm(bats, desc='bats'):
            ix = np.where(em_abs == b)[0]
            if len(ix) < 5: continue
            if len(ix) > N_PER_EM:
                ix = rng.choice(ix, size=N_PER_EM, replace=False)
            ix = np.sort(ix)
            # MFCC from tf_specs (treat as log-mel after transposing to (n_mels, T))
            mfccs = [librosa.feature.mfcc(S=tf_specs[i].T, n_mfcc=N_MFCC) for i in ix]
            n = len(mfccs)
            D = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                for j in range(i+1, n):
                    D_, wp = librosa.sequence.dtw(X=mfccs[i], Y=mfccs[j], metric='euclidean')
                    d = float(D_[-1,-1]) / max(len(wp), 1)
                    D[i,j] = d; D[j,i] = d
            D = (D - D.min()) / (D.max() - D.min() + 1e-9)
            cond = squareform(D, checks=False)
            Z = linkage(cond, method='ward')
            coph = cophenet(Z)
            cut = float(np.quantile(coph, Q_PROXY))
            lbl = fcluster(Z, t=cut, criterion='distance')
            n_types = len(set(lbl.tolist()))
            proxy[ix] = lbl + offset
            offset += n_types + 1
            per_em_stats.append({'bat': b, 'n': n, 'n_types': n_types})

        np.save(PROXY, proxy)
        pdf = pd.DataFrame(per_em_stats)
        print(f'  proxy types/bat: {pdf.n_types.mean():.1f} ± {pdf.n_types.std():.1f}'
              f'  [paper: 27 ± 2]')

    # Per-emitter ARI/NMI vs HDBSCAN-NCA labels
    rec = []
    for b in bats:
        m = (em_abs == b) & (proxy >= 0)
        if m.sum() < 30: continue
        ari = adjusted_rand_score(proxy[m], nca[m])
        nmi = normalized_mutual_info_score(proxy[m], nca[m])
        rec.append({'bat': b, 'n': int(m.sum()), 'ari': ari, 'nmi': nmi})
    p = pd.DataFrame(rec)
    print(f'\nPer-emitter ARI: {p.ari.mean():.3f} ± {p.ari.std():.3f}  [paper: 0.12 ± 0.01]')
    print(f'Per-emitter NMI: {p.nmi.mean():.3f} ± {p.nmi.std():.3f}  [paper: 0.30 ± 0.01]')
    print(f'N bats with proxy: {len(p)}')

    # Save proxy_label into seg_df + summary csv
    seg_df['proxy_label'] = proxy
    seg_df['syllable_id'] = nca
    st['seg_df'] = seg_df
    joblib.dump(st, STATE, compress=3)
    p.to_csv('docs/thesis/figures/per_emitter_ari_nmi_152k.csv', index=False)
    print(f'\nUpdated state with proxy_label + syllable_id columns')
    print(f'Saved per-emitter table: docs/thesis/figures/per_emitter_ari_nmi_152k.csv')


if __name__ == '__main__':
    main()
