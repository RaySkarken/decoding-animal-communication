"""Acoustic ground-truth proxy labels (Assom / 0.3 Syllables dictionaries).

DTW on MFCC features + Ward linkage + cophenetic-distance quantile cut
(``syllableID_qt_ward``-style). Used for RQ1-style ARI/NMI vs HDBSCAN, not
behavioral context.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


# MFCC / STFT — decodingNonHumanCommunication/0.3 Syllables dictionaries.ipynb
MFCC_N_FFT = 2048
MFCC_WIN_LENGTH = 1024
MFCC_HOP_LENGTH = 256
MFCC_N_MELS = 64
MFCC_FMAX = 120_000


def _mean_norm_mfcc(mfcc: np.ndarray) -> np.ndarray:
    mfcc_std = np.std(mfcc, axis=1, keepdims=True)
    mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
    if np.all(mfcc_std != 0):
        return (mfcc - mfcc_mean) / mfcc_std
    return mfcc - mfcc_mean


def _dtw_distance_matrix(mfcc_list: list) -> np.ndarray:
    import dtaidistance.dtw_ndim as dm
    from dtaidistance.exceptions import CythonException

    series_new = [_mean_norm_mfcc(x) for x in mfcc_list]
    seqs = [s.T.astype(np.float64) for s in series_new]
    try:
        return np.asarray(dm.distance_matrix_fast(seqs, parallel=False))
    except CythonException:
        return np.asarray(dm.distance_matrix(seqs, use_c=False, parallel=False))


def ward_qt_cluster_labels(ds: np.ndarray, cophene_distance_quantile: float = 0.05) -> np.ndarray:
    from scipy.cluster import hierarchy
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    ds = np.asarray(ds, dtype=np.float64)
    ds = (ds - np.min(ds)) / (np.max(ds) - np.min(ds) + 1e-12)
    condensed = squareform(ds, checks=False)
    z = linkage(condensed, method="ward")
    coph_out = hierarchy.cophenet(z)
    # scipy ≥1.6: cophenet(Z) → condensed distances only; two-arg call → (corr, condensed).
    if isinstance(coph_out, tuple):
        coph_dists = np.asarray(coph_out[1], dtype=np.float64).ravel()
    else:
        coph_dists = np.asarray(coph_out, dtype=np.float64).ravel()
    t = np.quantile(coph_dists, cophene_distance_quantile)
    return fcluster(z, t, criterion="distance").astype(np.int32)


def segment_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    y = np.asarray(y, dtype=np.float32).ravel()
    if len(y) < MFCC_WIN_LENGTH:
        y = np.pad(y, (0, MFCC_WIN_LENGTH - len(y)))
    return librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_fft=MFCC_N_FFT,
        win_length=MFCC_WIN_LENGTH,
        hop_length=MFCC_HOP_LENGTH,
        n_mels=MFCC_N_MELS,
        fmax=min(MFCC_FMAX, sr // 2 - 1),
    )


def compute_acoustic_proxy_labels(
    seg_df: pd.DataFrame,
    *,
    max_segments_per_emitter: Optional[int] = 400,
    cophene_quantile: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-emitter DTW–MFCC–Ward proxy labels (Assom-style).

    Parameters
    ----------
    seg_df
        Must contain ``audio``, ``sr``, ``emitter``. Rows are processed in
        emitter groups; within each emitter at most *max_segments_per_emitter*
        segments are used (first rows in sorted DataFrame index order).

    Returns
    -------
    proxy_labels : ndarray (n_rows,)
        Ward cluster id (1..K) where computed; ``-1`` where skipped.
    mask_computed : ndarray (n_rows,) bool
        True where *proxy_labels* is valid.
    """
    if "audio" not in seg_df.columns or "sr" not in seg_df.columns:
        raise ValueError("seg_df must contain 'audio' and 'sr' columns")

    n = len(seg_df)
    proxy_labels = np.full(n, -1, dtype=np.int32)
    mask = np.zeros(n, dtype=bool)

    for emitter in np.unique(seg_df["emitter"].values):
        ix = np.where(seg_df["emitter"].values == emitter)[0]
        ix = np.sort(ix)
        if max_segments_per_emitter is not None and len(ix) > max_segments_per_emitter:
            ix = ix[:max_segments_per_emitter]
        if len(ix) < 3:
            continue

        mfcc_list = []
        for i in ix:
            row = seg_df.iloc[int(i)]
            mfcc_list.append(segment_mfcc(row["audio"], int(row["sr"])))

        ds = _dtw_distance_matrix(mfcc_list)
        local = ward_qt_cluster_labels(ds, cophene_distance_quantile=cophene_quantile)
        proxy_labels[ix] = local
        mask[ix] = True

    return proxy_labels, mask
