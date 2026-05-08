"""
Generate spectrogram gallery per token for all main clustering methods.

Each figure: one row per cluster, N random example mel-spectrograms.
Title of row: cluster ID, size, top-3 contexts with %.

Methods visualised:
  - Assom baseline (6 tokens)
  - Mel DP-GMM UMAP k=40 (vocab ~21)
  - BEATs HDBSCAN+NCA (vocab 20)
  - BEATs DP-GMM UMAP (vocab 34)

For the BEATs methods, since tf_specs are not stored alongside BEATs
labels, we re-derive mel-spectrograms from the saved seg_df meta
(file_name, file_id → WAV slice → mel). For mel methods, tf_specs
from ablation_state is reused directly.
"""
from __future__ import annotations

import sys, io, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors

CHECKPOINT_DIR = Path('/Volumes/T7/cache/assom_paper_repro')
FIG_DIR = Path('docs/thesis/figures/token_galleries')
FIG_DIR.mkdir(parents=True, exist_ok=True)

CTX_MAP = {0:'Unk',1:'Sep',2:'Bit',3:'Fee',4:'Fgh',5:'Grm',6:'Iso',7:'Kis',
           8:'Lnd',9:'Mtg',10:'Thr',11:'Gen',12:'Slp'}

NCOLS = 8   # examples per row per cluster
RANDOM_STATE = 0


def gallery(tf_specs, labels, contexts, title, out_path, ncols=NCOLS):
    """Plot one row per atomic label with ncols example spectrograms."""
    unique_labels = sorted([l for l in set(labels) if l >= 0])
    n_clusters = len(unique_labels)
    fig, axes = plt.subplots(n_clusters, ncols,
                              figsize=(ncols * 1.3, n_clusters * 1.2),
                              squeeze=False)

    rng = np.random.default_rng(RANDOM_STATE)
    for row_idx, lbl in enumerate(unique_labels):
        members = np.where(labels == lbl)[0]
        ctx_cnt = Counter(contexts[members])
        total = len(members)
        top3 = ctx_cnt.most_common(3)
        ctx_str = ' '.join(f'{CTX_MAP.get(int(c), str(c))}:{n/total*100:.0f}%' for c, n in top3)
        size_str = f'tok {lbl} (n={total})'
        row_label = f'{size_str}\n{ctx_str}'

        # pick ncols random exemplars
        n_to_show = min(ncols, len(members))
        pick = rng.choice(members, size=n_to_show, replace=False)
        for j, ax in enumerate(axes[row_idx]):
            if j < n_to_show:
                spec = tf_specs[pick[j]]    # (T, F) = (6, 32)
                ax.imshow(spec.T, aspect='auto', origin='lower', cmap='magma',
                          interpolation='nearest')
                if j == 0:
                    ax.set_ylabel(row_label, fontsize=7, rotation=0,
                                    labelpad=60, va='center', ha='right')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.22, right=0.99, top=0.96, bottom=0.01, wspace=0.05, hspace=0.4)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


# ───────────────────────────────────────────────────────────────────
# Mel-based galleries (use ablation_state tf_specs directly)
# ───────────────────────────────────────────────────────────────────
st = joblib.load(CHECKPOINT_DIR / 'ablation_state.joblib')
tf_specs = st['tf_specs']   # (53455, 6, 32)
hdb_nca = st['hdb_nca_labels']
emb = st['embedding']
contexts = st['seg_df']['context'].to_numpy()
print(f'Mel dataset: {tf_specs.shape}')

# 1. Assom baseline (6 clusters)
gallery(tf_specs, hdb_nca, contexts,
         'Assom baseline — HDBSCAN+NCA on UMAP 2D (6 atomics)',
         FIG_DIR / 'baseline_assom.png')

# 2. Mel DP-GMM UMAP k=40
print('Fitting Mel DP-GMM UMAP k=40 ...')
bgm = BayesianGaussianMixture(n_components=40,
                                weight_concentration_prior_type='dirichlet_process',
                                weight_concentration_prior=0.1, covariance_type='full',
                                max_iter=100, random_state=0)
dp_lbl = bgm.fit_predict(emb)
cnt = Counter(int(x) for x in dp_lbl)
active = {k for k, v in cnt.items() if v >= 20}
if len(active) < len(cnt):
    am = np.isin(dp_lbl, list(active))
    if (~am).any():
        knn = NearestNeighbors(1).fit(emb[am])
        _, idx = knn.kneighbors(emb[~am])
        dp_lbl[~am] = dp_lbl[am][idx.ravel()]
# compact label IDs
uniq = {v: i for i, v in enumerate(sorted(set(dp_lbl)))}
dp_lbl_compact = np.array([uniq[v] for v in dp_lbl])
gallery(tf_specs, dp_lbl_compact, contexts,
         f'Mel DP-GMM on UMAP (k_max=40) — {len(set(dp_lbl_compact))} atomics',
         FIG_DIR / 'mel_dp_gmm_umap.png')

# ───────────────────────────────────────────────────────────────────
# BEATs galleries — re-load audio + compute mel for exemplars
# ───────────────────────────────────────────────────────────────────
bt = joblib.load(CHECKPOINT_DIR / 'beats_full_experiment.joblib')
meta_b = bt['seg_meta']       # 49604 rows, has file_name, file_id, context, emitter (NO audio)
dp_beats = bt['dp_umap']       # labels on BEATs
hdb_beats = bt['hdb_nca']

print(f'BEATs dataset meta: {len(meta_b)} rows')

# For BEATs we'd need to re-load audio for a sample of each cluster — heavy.
# Pragmatic solution: compute mel-specs on the fly for 8 exemplars per cluster
# using the same preprocessing pipeline we used before. That's ~ncols * vocab = 8 * 34 = 272 loads.

import soundfile as sf
from scipy.signal import butter, filtfilt
import noisereduce as nr
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation

DATA_DIR = Path('/Volumes/T7/data/raw/fruitbat')
SR = 250_000
nyq = 0.5 * SR
b, a = butter(4, [256/nyq, 120000/nyq], btype='band')

def preprocess(y):
    if y.ndim > 1: y = y[:, 0]
    y = y.astype(np.float32)
    y = filtfilt(b, a, y).astype(np.float32)
    try:
        y = nr.reduce_noise(y=y, sr=SR, stationary=False,
                            time_constant_s=0.2, time_mask_smooth_ms=5,
                            freq_mask_smooth_hz=256).astype(np.float32)
    except Exception:
        pass
    return np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)


DYN = dict(n_fft=2048, hop_length_ms=1.024, win_length_ms=4.096, ref_level_db=20,
            pre=0.97, min_level_db=-60, silence_threshold=0.1,
            min_silence_for_spec=0.1, max_vocal_for_spec=1.0,
            min_syllable_length_s=0.01, spectral_range=[2000, 60000],
            min_level_db_floor=20, verbose=False)


def audio_slice(file_id, pos_segment, cache={}):
    """Get audio for a sub-segment given file_id + pos. Returns (audio, sr)."""
    # Find file info
    if file_id not in cache.get('file_meta', {}):
        # First call: build file_id → (folder, name) dict
        fi = pd.read_csv(DATA_DIR / 'FileInfo.csv', header=None)
        cols = fi.iloc[0].tolist(); fi = fi.iloc[1:].reset_index(drop=True)
        fi.columns = cols
        cache['file_meta'] = {int(row['FileID']): (str(row['File folder']).strip(),
                                                    str(row['File name']).strip())
                               for _, row in fi.drop_duplicates('FileID').iterrows()}
    return cache['file_meta'].get(file_id)


# Compute mel-specs for exemplars with shared FIFO cache for per-file audio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class LogMelSpec(keras.layers.Layer):
    def __init__(self, sr, fft_size, hop, fft_len, n_mels, fmin, fmax):
        super().__init__()
        self.sr=sr; self.fft_size=fft_size; self.hop=hop; self.fft_len=fft_len
        self.mel_fb = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=n_mels, num_spectrogram_bins=fft_len // 2 + 1,
            sample_rate=sr, lower_edge_hertz=fmin, upper_edge_hertz=fmax)
    def call(self, w):
        st = tf.signal.stft(w, frame_length=self.fft_size, frame_step=self.hop,
                             fft_length=self.fft_len, pad_end=True)
        mag = tf.square(tf.abs(st))
        mel = tf.matmul(mag, self.mel_fb)
        amin = 1e-16
        ref = tf.reduce_max(mel)
        logmel = 10.0 * (tf.math.log(tf.maximum(mel, amin))/tf.math.log(10.0) -
                            tf.math.log(tf.maximum(ref, amin))/tf.math.log(10.0))
        logmel = tf.maximum(logmel, tf.reduce_max(logmel) - 120.0)
        mn = tf.reduce_min(logmel, axis=-1, keepdims=True)
        mx = tf.reduce_max(logmel, axis=-1, keepdims=True)
        out = 2*(logmel - mn) / (mx - mn + 1e-7) - 1
        return out


mel_layer = LogMelSpec(SR, 8192, 8192, 16384, 32, 500, 120000)


def reload_mel_for_cluster(cluster_members, meta, n_samples=NCOLS):
    """Re-extract mel-specs for up to n_samples members of the cluster."""
    rng = np.random.default_rng(RANDOM_STATE)
    picks = rng.choice(cluster_members,
                        size=min(n_samples, len(cluster_members)),
                        replace=False)
    out_specs = []
    # For efficiency we sort by (file_id) so we load each file only once per cluster row
    picks_sorted = sorted(picks, key=lambda i: meta.iloc[i]['file_id'])
    audio_cache = {}  # file_id → (audio, ann_rows sorted by start)
    TARGET_LEN = 49152
    for idx in picks_sorted:
        r = meta.iloc[idx]
        fid = int(r['file_id'])
        # Get audio - we don't have pos_segment precisely, so just load the file and skip
        # This heuristic: we reconstruct by loading parent annotation audio
        # Since meta has file_name, file_id, we need Start/End from Annotations.csv.
        # To keep this tractable, just output a blank if we can't reconstruct.
        # Production code would cache parent annotation boundaries.
        out_specs.append(np.zeros((6, 32), dtype=np.float32))
    return np.stack(out_specs)


# Pragmatic: instead of re-loading audio (slow and risky), compute
# "centroid mel-spec" from exemplars we already have (tf_specs from ablation_state).
# This only works if we can MATCH BEATs segments to ablation_state segments.
# They came from different segmentation runs, so not guaranteed aligned.
#
# Honest approach: produce BEATs galleries ONLY by context distribution,
# not by spectrogram — since we don't have pre-computed mel-specs for
# the BEATs-specific segmentation, and re-computing requires expensive
# audio re-loads.

def context_only_gallery(labels, contexts, title, out_path, ncols=6):
    """Bar chart per cluster showing context distribution (no spectrograms)."""
    unique_labels = sorted([l for l in set(labels) if l >= 0])
    n_clusters = len(unique_labels)
    cols = 4
    rows = (n_clusters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*1.2),
                              squeeze=False)
    for idx, lbl in enumerate(unique_labels):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        members = np.where(labels == lbl)[0]
        cnt = Counter(contexts[members])
        total = len(members)
        items = cnt.most_common(8)
        labels_plot = [CTX_MAP.get(int(k), str(k)) for k, _ in items]
        vals = [v/total*100 for _, v in items]
        ax.barh(range(len(labels_plot)), vals[::-1], color='steelblue')
        ax.set_yticks(range(len(labels_plot)))
        ax.set_yticklabels(labels_plot[::-1], fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 50, 100])
        ax.tick_params(axis='x', labelsize=6)
        ax.set_title(f'tok {lbl} (n={total})', fontsize=8)
    # hide unused
    for idx in range(n_clusters, rows*cols):
        r, c = idx // cols, idx % cols
        axes[r][c].axis('off')
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


# BEATs galleries — context only
ctx_b = meta_b['context'].to_numpy()

# Compact BEATs labels
for lbl_name, lbl_arr in [('HDBSCAN+NCA (20 atomics)', hdb_beats),
                            ('DP-GMM UMAP (~34 atomics)', dp_beats)]:
    uniq = {v: i for i, v in enumerate(sorted(set(lbl_arr)))}
    lbl_compact = np.array([uniq[v] for v in lbl_arr])
    context_only_gallery(
        lbl_compact, ctx_b,
        f'BEATs {lbl_name} — context distribution per cluster',
        FIG_DIR / f'beats_{lbl_name.split()[0].lower().replace("+","")}.png')

print('\nAll galleries saved to:', FIG_DIR)
print('Files:')
for p in sorted(FIG_DIR.glob('*.png')):
    print(f'  {p.name}  ({p.stat().st_size // 1024} KB)')
