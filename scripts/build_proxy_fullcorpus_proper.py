"""Rebuild DTW-MFCC proxy on full-corpus — CORRECT Assom methodology.

Algorithm (matches notebooks/_build_assom_paper_reproduction.py L920-955):
  1. Pairwise DTW distance matrix (per emitter)
  2. Min-max normalize to [0, 1]
  3. WARD linkage (not average!)
  4. Compute COPHENETIC distances from the linkage tree
  5. Threshold = quantile(cophenetic_distances, 0.05)  (NOT raw DTW quantile)
  6. fcluster with criterion='distance'

Target: paper's 27 ± 2 syllable types per emitter (old pipeline: 22.7 ± 3.6)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, joblib, librosa
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform
from tqdm.auto import tqdm

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
STATE = CACHE / 'ablation_state_fullcorpus.joblib'

N_PER_EM = 400
Q = 0.05
N_MFCC = 13
RNG_SEED = 0


def main():
    st = joblib.load(STATE)
    seg_df = st['seg_df']
    tf_specs = st['tf_specs']
    em_abs = np.abs(seg_df['emitter'].to_numpy())
    unique_em = sorted(set(em_abs.tolist()))
    print(f'Full corpus: {len(seg_df)} segs | {len(unique_em)} physical bats')

    rng = np.random.default_rng(RNG_SEED)
    proxy = np.full(len(seg_df), -1, dtype=np.int32)
    offset = 0
    per_em_stats = []

    for em in tqdm(unique_em, desc='emitters'):
        ix = np.where(em_abs == em)[0]
        if len(ix) < 5: continue
        if len(ix) > N_PER_EM:
            ix = rng.choice(ix, size=N_PER_EM, replace=False)
        ix = np.sort(ix)

        specs = tf_specs[ix]
        mfccs = [librosa.feature.mfcc(S=s, n_mfcc=N_MFCC) for s in specs]
        n = len(mfccs)

        # 1. Pairwise DTW
        D = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i+1, n):
                D_, wp = librosa.sequence.dtw(X=mfccs[i], Y=mfccs[j], metric='euclidean')
                d = float(D_[-1, -1]) / max(len(wp), 1)
                D[i, j] = d; D[j, i] = d

        # 2. Min-max normalize to [0, 1]
        D_norm = (D - D.min()) / (D.max() - D.min() + 1e-9)

        # 3. Ward linkage on condensed (upper-tri) distance matrix
        condensed = squareform(D_norm, checks=False)
        Z = linkage(condensed, method='ward')

        # 4. Cophenetic distances from the tree
        coph_dists = cophenet(Z)

        # 5. Threshold = quantile of cophenetic (NOT raw) at q
        cut = float(np.quantile(coph_dists, Q))

        # 6. fcluster at that threshold
        lbl = fcluster(Z, t=cut, criterion='distance')
        n_types = len(set(lbl.tolist()))

        proxy[ix] = lbl + offset
        offset += n_types + 1
        per_em_stats.append({
            'emitter_abs': em, 'n_points': n, 'n_types': n_types,
            'coph_cut': round(cut, 4),
        })

    pdf = pd.DataFrame(per_em_stats)
    print(f'\n=== Per-emitter proxy stats (fullcorpus, correct methodology) ===')
    print(f'  Emitters processed: {len(pdf)}')
    print(f'  Mean points/emitter: {pdf.n_points.mean():.0f}')
    print(f'  Mean TYPES/emitter:  {pdf.n_types.mean():.1f} ± {pdf.n_types.std():.1f}  [paper: 27 ± 2]')
    print(f'  min/max types: {pdf.n_types.min()}/{pdf.n_types.max()}')
    print(f'  Mean cophenetic cut: {pdf.coph_cut.mean():.4f}')

    seg_df['proxy_label'] = proxy
    st['seg_df'] = seg_df
    joblib.dump(st, STATE, compress=3)
    print(f'\nSaved updated proxy_label into: {STATE}')

    pdf.to_csv('docs/thesis/figures/proxy_fullcorpus_per_emitter.csv', index=False)
    print(f'Saved: docs/thesis/figures/proxy_fullcorpus_per_emitter.csv')

    n_cov = (proxy >= 0).sum()
    print(f'\nProxy coverage: {n_cov} / {len(proxy)} ({n_cov/len(proxy):.1%})')
    print(f'Total global proxy types: {offset}')


if __name__ == '__main__':
    main()
