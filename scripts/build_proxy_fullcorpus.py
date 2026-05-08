"""Build DTW-MFCC qt_ward proxy labels on full-corpus state.

Replicates paper's per-emitter DTW-on-MFCCs + agglomerative clustering with
quantile threshold q=0.05. Adds proxy_label column to seg_df in-place.

Input:  ablation_state_fullcorpus.joblib
Output: rewrites same file with proxy_label column added
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, joblib, librosa
from sklearn.cluster import AgglomerativeClustering
from tqdm.auto import tqdm

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
STATE_PATH = CACHE / 'ablation_state_fullcorpus.joblib'

N_PROXY_PER_EMITTER = 300     # paper uses all; we cap for DTW compute
Q = 0.05                       # quantile threshold (paper)
N_TOP_EMITTERS = 10            # compute proxy for top-10 emitters
RANDOM_STATE = 42


def compute_proxy(st):
    seg_df = st['seg_df']
    tf_specs = st['tf_specs']
    em_abs = seg_df['emitter'].abs().values
    emit_counts = {}
    for e in set(em_abs.tolist()):
        emit_counts[e] = int((em_abs == e).sum())
    top = sorted(emit_counts.items(), key=lambda x: -x[1])[:N_TOP_EMITTERS]
    print(f'Top {N_TOP_EMITTERS} emitters (by |ID|): {top}')

    rng = np.random.default_rng(RANDOM_STATE)
    proxy = np.full(len(seg_df), -1, dtype=int)
    offset = 0

    for em, n_total in top:
        ix = np.where(em_abs == em)[0]
        if len(ix) > N_PROXY_PER_EMITTER:
            ix = rng.choice(ix, size=N_PROXY_PER_EMITTER, replace=False)
        specs = tf_specs[ix]  # (n, 6, 32)
        # derive MFCC from log-mel via DCT
        mfccs = [librosa.feature.mfcc(S=s, n_mfcc=13) for s in specs]
        n = len(mfccs)
        if n < 3: continue
        print(f'  em={em}: n={n}, computing DTW distance matrix...')
        dist = np.zeros((n, n), dtype=np.float32)
        for i in tqdm(range(n), desc=f'em={em}', leave=False):
            for j in range(i+1, n):
                D, wp = librosa.sequence.dtw(X=mfccs[i], Y=mfccs[j], metric='euclidean')
                d = float(D[-1, -1]) / max(len(wp), 1)
                dist[i, j] = d; dist[j, i] = d
        tri = dist[np.triu_indices(n, 1)]
        thr = float(np.quantile(tri, Q))
        ac = AgglomerativeClustering(n_clusters=None, metric='precomputed',
                                      linkage='average', distance_threshold=thr)
        lbl = ac.fit_predict(dist).astype(int)
        n_types = int(lbl.max()) + 1
        print(f'    → {n_types} types (threshold={thr:.2f})')
        proxy[ix] = lbl + offset
        offset += n_types

    return proxy


def main():
    print(f'Loading {STATE_PATH}...')
    st = joblib.load(STATE_PATH)
    seg_df = st['seg_df']
    print(f'N = {len(seg_df)}')
    proxy = compute_proxy(st)
    assigned = (proxy >= 0).sum()
    print(f'\nProxy assigned to {assigned} points ({assigned/len(seg_df):.1%})')
    print(f'Total unique proxy types: {len(set(proxy[proxy>=0].tolist()))}')

    # types per emitter
    em_abs = seg_df['emitter'].abs().values
    types = []
    for e in set(em_abs[proxy >= 0].tolist()):
        mask = (em_abs == e) & (proxy >= 0)
        if mask.sum() < 30: continue
        types.append(len(set(proxy[mask].tolist())))
    print(f'Mean types per emitter: {np.mean(types):.1f} ± {np.std(types):.1f}'
          f' [paper: 27 ± 2]  |  n_emitters with ≥30 labeled = {len(types)}')

    seg_df['proxy_label'] = proxy
    st['seg_df'] = seg_df
    joblib.dump(st, STATE_PATH, compress=3)
    print(f'Saved proxy_label into {STATE_PATH.name}')


if __name__ == '__main__':
    main()
