"""UMAP + HDBSCAN sweep on full-corpus ablation state.

Runs:
    1) UMAP fit on (~150k, 192) tf_specs
    2) HDBSCAN sweep across (frac, min_samples, epsilon) grid
    3) Per-emitter ARI/NMI against proxy (reuses proxy_label column if present)
    4) Saves sweep CSV and embedding
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, joblib
import umap, hdbscan
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
STATE_PATH = CACHE / 'ablation_state_fullcorpus.joblib'
EMB_PATH = CACHE / 'umap_fullcorpus_nn30_md0.3.npy'
OUT_CSV = Path('docs/thesis/figures/hdbscan_repro_sweep_fullcorpus.csv')

FRACS = [0.003, 0.005, 0.007, 0.010, 0.015, 0.020, 0.025]
MIN_SAMPLES = [10, 20]
EPS = [0.05, 0.10]
METHOD = 'leaf'


def peremit_ari_nmi(lbl, proxy, em):
    aris, nmis = [], []
    for e in sorted(set(em)):
        mask = (em == e) & (proxy >= 0) & (lbl >= 0)
        if mask.sum() < 30: continue
        aris.append(adjusted_rand_score(proxy[mask], lbl[mask]))
        nmis.append(normalized_mutual_info_score(proxy[mask], lbl[mask]))
    if not aris: return np.nan, np.nan, np.nan, np.nan, 0
    return np.mean(aris), np.std(aris), np.mean(nmis), np.std(nmis), len(aris)


def peremit_ntypes(lbl, em):
    cs = []
    for e in sorted(set(em)):
        mask = (em == e) & (lbl >= 0)
        if mask.sum() < 30: continue
        cs.append(len(set(lbl[mask].tolist())))
    if not cs: return np.nan, np.nan, 0
    return np.mean(cs), np.std(cs), len(cs)


def knn_reassign(lbl, emb, k=30):
    if (lbl == -1).sum() == 0: return lbl
    nn = lbl >= 0
    if nn.sum() < k: return lbl
    knn = KNeighborsClassifier(n_neighbors=min(k, nn.sum()), weights='uniform',
                                n_jobs=-1).fit(emb[nn], lbl[nn])
    out = lbl.copy()
    out[~nn] = knn.predict(emb[~nn])
    return out


def main():
    print(f'Loading {STATE_PATH}...')
    st = joblib.load(STATE_PATH)
    seg_df, tf_specs = st['seg_df'], st['tf_specs']
    em = seg_df['emitter'].values
    # use abs(emitter) for per-emitter grouping (negative IDs = same physical bat)
    em_abs = np.abs(em)
    proxy = seg_df['proxy_label'].values if 'proxy_label' in seg_df.columns \
            else np.full(len(seg_df), -1)
    if (proxy < 0).all():
        print('  WARN: no proxy_label in fullcorpus state; ARI/NMI will be NaN.')
        print('  Proxy needs to be built separately (use scripts/build_proxy_fullcorpus.py)')

    X = tf_specs.reshape(len(tf_specs), -1)
    print(f'  tf_specs: {tf_specs.shape} → X: {X.shape}')
    print(f'  unique |emitters|: {len(set(em_abs))}')

    # UMAP
    if EMB_PATH.exists():
        print(f'Loading UMAP from {EMB_PATH}...')
        emb = np.load(EMB_PATH)
    else:
        print('Fitting UMAP (n_neighbors=30, min_dist=0.3)...')
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                             metric='euclidean', random_state=0)
        emb = reducer.fit_transform(X).astype(np.float32)
        np.save(EMB_PATH, emb)
        print(f'  saved: {EMB_PATH}')
    print(f'  UMAP: {emb.shape}')

    # HDBSCAN sweep
    N = len(emb)
    rows = []
    print(f'\nHDBSCAN sweep over {len(FRACS)}×{len(MIN_SAMPLES)}×{len(EPS)} = '
          f'{len(FRACS)*len(MIN_SAMPLES)*len(EPS)} configs...')
    print(f'{"frac":>6s} {"mcs":>6s} {"ms":>4s} {"eps":>6s}'
          f' {"n_cl":>5s} {"noise%":>7s} {"sil":>7s} {"ARI_pe":>7s} {"NMI_pe":>7s} {"t/em":>6s}')
    for frac in FRACS:
        mcs = max(10, int(N * frac))
        for ms in MIN_SAMPLES:
            for eps in EPS:
                hdb = hdbscan.HDBSCAN(
                    min_cluster_size=mcs, min_samples=ms,
                    cluster_selection_epsilon=eps,
                    cluster_selection_method=METHOD,
                    metric='euclidean', core_dist_n_jobs=-1,
                ).fit(emb)
                lbl = hdb.labels_
                n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
                noise = int((lbl == -1).sum())
                try:
                    nn = lbl >= 0
                    sil = float(silhouette_score(emb[nn][:10000], lbl[nn][:10000])) \
                          if nn.sum() > 100 and n_cl > 1 else np.nan
                except Exception:
                    sil = np.nan
                ari_m, ari_s, nmi_m, nmi_s, n_em = peremit_ari_nmi(lbl, proxy, em_abs)
                nt_m, nt_s, _ = peremit_ntypes(lbl, em_abs)
                lbl_r = knn_reassign(lbl, emb)
                ari_rm, ari_rs, nmi_rm, nmi_rs, _ = peremit_ari_nmi(lbl_r, proxy, em_abs)
                nt_rm, nt_rs, _ = peremit_ntypes(lbl_r, em_abs)
                rows.append({
                    'frac': frac, 'mcs': mcs, 'min_samples': ms, 'eps': eps,
                    'n_clusters': n_cl, 'noise_frac': noise/N, 'n_em_proxy': n_em,
                    'silhouette': sil,
                    'ARI_pe_noise': ari_m, 'ARI_pe_noise_sd': ari_s,
                    'NMI_pe_noise': nmi_m, 'NMI_pe_noise_sd': nmi_s,
                    'types_per_em_raw': nt_m, 'types_per_em_raw_sd': nt_s,
                    'ARI_pe_reassigned': ari_rm, 'ARI_pe_reassigned_sd': ari_rs,
                    'NMI_pe_reassigned': nmi_rm, 'NMI_pe_reassigned_sd': nmi_rs,
                    'types_per_em_reassigned': nt_rm, 'types_per_em_reassigned_sd': nt_rs,
                })
                print(f'{frac:>6.3f} {mcs:>6d} {ms:>4d} {eps:>6.2f}'
                      f' {n_cl:>5d} {noise/N:>7.1%} {sil if not np.isnan(sil) else 0:>7.3f}'
                      f' {ari_m if not np.isnan(ari_m) else 0:>7.3f}'
                      f' {nmi_m if not np.isnan(nmi_m) else 0:>7.3f}'
                      f' {nt_m if not np.isnan(nt_m) else 0:>6.1f}')

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')
    print(f'\n=== 7-cluster configs ===')
    print(df[df.n_clusters == 7][['frac','mcs','min_samples','eps','ARI_pe_noise',
                                    'NMI_pe_noise','types_per_em_raw']].to_string(index=False))
    print(f'\n=== Top 5 by ARI_pe_noise ===')
    print(df.sort_values('ARI_pe_noise', ascending=False).head()[
        ['frac','mcs','min_samples','eps','n_clusters','ARI_pe_noise','NMI_pe_noise',
         'types_per_em_raw']].to_string(index=False))


if __name__ == '__main__':
    main()
