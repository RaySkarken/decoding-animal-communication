"""Build ablation_state with FULL corpus — STREAMING version (memory-safe).

Processes one WAV file at a time: load → annotate → preprocess → dynseg →
mel → discard audio. Peak memory stays below ~2 GB regardless of corpus size.

Key change from batch version: drops audio from working set immediately after
mel computation. Suitable for full 82-emitter / ~22k-file run on a laptop.

Output: /Volumes/T7/cache/assom_paper_repro/ablation_state_fullcorpus.joblib
"""
from __future__ import annotations
import io, sys, warnings, zipfile, gc, random
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
from tqdm.auto import tqdm
import soundfile as sf
from scipy import signal as scipy_signal
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    import noisereduce as nr
    HAVE_NR = True
except ImportError:
    HAVE_NR = False
    print('noisereduce not available; skipping NR step')

try:
    from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation
except ImportError:
    print('FATAL: vocalseg not available'); sys.exit(1)


# ─────────────────────────── CONFIG (from builder) ───────────────────────────
CHECKPOINT_DIR = Path('/Volumes/T7/cache/assom_paper_repro')
DATA_DIR = Path('/Volumes/T7/data/raw/fruitbat')
OUT_PATH = CHECKPOINT_DIR / 'ablation_state_fullcorpus.joblib'
PROGRESS_DIR = CHECKPOINT_DIR / 'fullcorpus_progress'
PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

SR = 250_000
CONTEXT_DICT = {0: 'Unknown', 1: 'Separation', 2: 'Biting', 3: 'Feeding',
                4: 'Fighting', 5: 'Grooming', 6: 'Isolation', 7: 'Kissing',
                8: 'Landing', 9: 'Mating protest', 10: 'Threat-like',
                11: 'General', 12: 'Sleeping'}
EXCLUDED_CONTEXTS = [0, 11, 12]

BP_LOW, BP_HIGH = 256, 120_000
NR_TIME_CONSTANT_S = 0.2
NR_TIME_MASK_SMOOTH_MS = 5
NR_FREQ_MASK_SMOOTH_HZ = 256
NR_STATIONARY = False
PRE_EMPHASIS = 0.97

DYN_SEG_PARAMS = dict(
    n_fft=2048, hop_length_ms=1000*256/SR, win_length_ms=1000*1024/SR,
    ref_level_db=20, pre=PRE_EMPHASIS, min_level_db=-60,
    silence_threshold=0.1, min_silence_for_spec=0.1, max_vocal_for_spec=1.0,
    min_syllable_length_s=0.01, spectral_range=[2000, 60000],
    min_level_db_floor=20, verbose=False,
)

TF_FFT_SIZE, TF_HOP_SIZE, TF_FFT_LENGTH = 8192, 8192, 16384
TF_N_MELS, TF_FMIN, TF_FMAX = 32, 500, 120_000
TF_NORMALIZE = 'tanh'
SPEC_TIME, SPEC_FREQ = 6, 32
TARGET_AUDIO_LEN = SPEC_TIME * TF_HOP_SIZE

NORM_ADAPT_N_FILES = 500    # ~= 3000 segments for normalization adapt
RANDOM_STATE = 0

# Checkpoint flushing — periodically save accumulated results
FLUSH_EVERY_FILES = 2000


# ─────────────────────────── LogMel layer ────────────────────────────────────
class LogMelSpectrogram(keras.layers.Layer):
    def __init__(self, sample_rate, fft_size, hop_size, fft_length, window_fn,
                 n_mels, f_min=0.0, f_max=None, normalize=None, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate; self.fft_size = fft_size
        self.hop_size = hop_size; self.fft_length = fft_length
        self.window_fn = window_fn; self.n_mels = n_mels
        self.f_min = f_min; self.f_max = f_max if f_max else sample_rate / 2
        self.normalize = normalize
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.fft_length // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min, upper_edge_hertz=self.f_max,
        )

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super().build(input_shape)

    def call(self, waveforms):
        def _tf_log10(x):
            return tf.math.log(x) / tf.math.log(tf.constant(10, dtype=x.dtype))

        def _norm(log_mel):
            min_v = tf.math.reduce_min(log_mel, axis=3, keepdims=True)
            max_v = tf.math.reduce_max(log_mel, axis=3, keepdims=True)
            if self.normalize == 'tanh':
                out = 2.0 * (log_mel - min_v) / (max_v - min_v + 1e-7) - 1.0
                nan_val = -1.0
            elif self.normalize == 'sigmoid':
                out = (log_mel - min_v) / (max_v - min_v + 1e-7)
                nan_val = 0.0
            else:
                return log_mel
            idx = tf.where(tf.math.is_nan(out))
            return tf.tensor_scatter_nd_update(
                out, idx, tf.ones(tf.shape(idx)[0], dtype=out.dtype) * nan_val)

        def _p2db(m, amin=1e-16, top_db=120.0):
            ref = tf.reduce_max(m)
            ls = 10.0 * _tf_log10(tf.maximum(amin, m))
            ls -= 10.0 * _tf_log10(tf.maximum(amin, ref))
            return tf.maximum(ls, tf.reduce_max(ls) - top_db)

        spec = tf.signal.stft(waveforms, frame_length=self.fft_size,
                              frame_step=self.hop_size, fft_length=self.fft_length,
                              pad_end=True)
        mag = tf.abs(spec)
        mel = tf.matmul(tf.square(mag), self.mel_filterbank)
        log_mel = _p2db(mel)
        log_mel = _norm(log_mel)
        sh = tf.shape(log_mel)
        return tf.reshape(log_mel, [-1, sh[2], sh[3]])


def _pad_or_crop(y, target_len):
    y = np.asarray(y, dtype=np.float32)
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        return y[start:start + target_len]
    pad_val = float(np.mean(y)) if len(y) else 0.0
    left = (target_len - len(y)) // 2
    right = target_len - len(y) - left
    return np.concatenate([np.full(left, pad_val, dtype=np.float32), y,
                           np.full(right, pad_val, dtype=np.float32)])


def butter_bp(lo, hi, fs, order=4):
    nyq = 0.5 * fs
    b, a = scipy_signal.butter(order, [lo/nyq, hi/nyq], btype='band')
    return b, a


_BP_B, _BP_A = butter_bp(BP_LOW, BP_HIGH, SR, order=4)


def preprocess_audio(y, sr):
    y = np.asarray(y, dtype=np.float32)
    if sr != SR:
        y = scipy_signal.resample(y, int(len(y) * SR / sr)).astype(np.float32)
    y = scipy_signal.filtfilt(_BP_B, _BP_A, y).astype(np.float32)
    if HAVE_NR and len(y) >= int(SR * NR_TIME_CONSTANT_S):
        try:
            y = nr.reduce_noise(
                y=y, sr=SR, stationary=NR_STATIONARY,
                time_constant_s=NR_TIME_CONSTANT_S,
                time_mask_smooth_ms=NR_TIME_MASK_SMOOTH_MS,
                freq_mask_smooth_hz=NR_FREQ_MASK_SMOOTH_HZ,
            ).astype(np.float32)
        except Exception:
            pass
    if PRE_EMPHASIS:
        y = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1]).astype(np.float32)
    return y


def extract_subsegs(audio):
    """Return list of sub-segments after preprocessing + dynamic seg."""
    a = preprocess_audio(audio, SR)
    try:
        r = dynamic_threshold_segmentation(a, SR, **DYN_SEG_PARAMS)
    except Exception:
        r = None
    subs = []
    if r is not None and len(r.get('onsets', [])) > 0:
        for onset_s, offset_s in zip(r['onsets'], r['offsets']):
            si, ei = int(onset_s * SR), int(offset_s * SR)
            sub = a[si:ei]
            if len(sub) >= 50:
                subs.append((sub, si, ei))
    if not subs:
        subs.append((a, 0, len(a)))
    return subs


# ─────────────────────────── MAIN ────────────────────────────────────────────

def load_annotations():
    ann = pd.read_csv(DATA_DIR / 'Annotations.csv', low_memory=False)
    with open(DATA_DIR / 'FileInfo.csv') as f:
        max_cols = max(len(l.split(',')) for l in f)
    fi = pd.read_csv(DATA_DIR / 'FileInfo.csv', header=None,
                     names=[f'c{i}' for i in range(max_cols)], low_memory=False)
    fi.columns = fi.iloc[0].values
    fi = fi.iloc[1:].reset_index(drop=True)
    fi['FileID'] = fi['FileID'].astype(int)
    fi = fi[['FileID', 'File name', 'File folder']].drop_duplicates('FileID')
    ann['FileID'] = ann['FileID'].astype(int)
    df = ann.merge(fi, on='FileID', how='inner')
    df['Emitter'] = pd.to_numeric(df['Emitter'], errors='coerce')
    df['Addressee'] = pd.to_numeric(df['Addressee'], errors='coerce')
    df['Context'] = pd.to_numeric(df['Context'], errors='coerce')
    df = df.dropna(subset=['Emitter', 'Context', 'Start sample', 'End sample'])
    df['Emitter'] = df['Emitter'].astype(int)
    df['Context'] = df['Context'].astype(int)
    df['Start sample'] = df['Start sample'].astype(int)
    df['End sample'] = df['End sample'].astype(int)
    df = df[df['Emitter'] != 0]
    df = df[~df['Context'].isin(EXCLUDED_CONTEXTS)]
    df['Context_name'] = df['Context'].map(CONTEXT_DICT)
    return df


def stream_subsegs(df):
    """Yield (seg_meta_dict, mel_raw) one sub-segment at a time.

    Opens each WAV file exactly once. After extracting all annotated
    sub-segments from that file and returning their raw log-mel, audio is
    freed. `mel_raw` is raw (tanh-normalized inside LogMelSpectrogram but
    NOT yet z-scored by the external Normalization layer)."""

    log_mel_layer = LogMelSpectrogram(
        sample_rate=SR, fft_size=TF_FFT_SIZE, hop_size=TF_HOP_SIZE,
        fft_length=TF_FFT_LENGTH, window_fn=tf.signal.hamming_window,
        n_mels=TF_N_MELS, f_min=TF_FMIN, f_max=TF_FMAX,
        normalize=TF_NORMALIZE, name='LogMel')

    reshape = layers.Reshape((1, -1))
    zip_cache = {}
    def get_zip(folder):
        if folder not in zip_cache:
            zp = DATA_DIR / f'{folder}.zip'
            zip_cache[folder] = zipfile.ZipFile(zp, 'r') if zp.exists() else None
        return zip_cache[folder]

    skipped = 0
    try:
        grouped = df.groupby('FileID')
        for file_id, group in tqdm(grouped, total=grouped.ngroups, desc='files'):
            row0 = group.iloc[0]
            folder = str(row0['File folder']).strip()
            fname = str(row0['File name']).strip()
            zf = get_zip(folder)
            if zf is None:
                skipped += len(group); continue
            try:
                wav_bytes = zf.read(fname)
                audio_full, file_sr = sf.read(io.BytesIO(wav_bytes), dtype='float32')
            except Exception:
                skipped += len(group); continue

            # collect sub-segments from this file
            sub_audios = []
            sub_metas = []
            for _, r in group.iterrows():
                s, e = int(r['Start sample']), int(r['End sample'])
                if e > len(audio_full) or s >= e:
                    skipped += 1; continue
                seg = audio_full[s:e].astype(np.float32)
                if len(seg) < 100:
                    skipped += 1; continue
                if file_sr != SR:
                    seg = scipy_signal.resample(seg,
                          int(len(seg) * SR / file_sr)).astype(np.float32)
                for sub, si, ei in extract_subsegs(seg):
                    if len(sub) < 50: continue
                    sub_audios.append(_pad_or_crop(sub, TARGET_AUDIO_LEN))
                    sub_metas.append({
                        'context': int(r['Context']),
                        'context_name': r['Context_name'],
                        'emitter': int(r['Emitter']),
                        'addressee': int(r['Addressee']) if pd.notna(r['Addressee']) else -1,
                        'file_name': fname, 'file_id': int(file_id),
                        'duration_s': len(sub) / SR,
                        'parent_start': int(s) + si, 'parent_end': int(s) + ei,
                    })
            if not sub_audios:
                continue

            # batch mel for this file (usually 1-20 subsegs)
            batch = np.stack(sub_audios, axis=0).astype(np.float32)
            with tf.device('/CPU:0'):
                mel_batch = log_mel_layer(reshape(tf.constant(batch))).numpy()
            # shape: (B, ~6, 32); crop/pad to (6, 32)
            for i in range(len(mel_batch)):
                m = mel_batch[i]
                if m.shape[0] == SPEC_TIME:
                    out = m
                elif m.shape[0] > SPEC_TIME:
                    out = m[:SPEC_TIME]
                else:
                    out = np.zeros((SPEC_TIME, SPEC_FREQ), dtype=np.float32)
                    out[:m.shape[0]] = m
                yield sub_metas[i], out.astype(np.float32)

            # free memory
            del audio_full, sub_audios, sub_metas, batch, mel_batch

    finally:
        for z in zip_cache.values():
            if z is not None:
                z.close()

    print(f'Skipped (missing/invalid): {skipped}')


def main():
    print('[1/3] Loading annotations...')
    df = load_annotations()
    print(f'Filtered annotations: {len(df)} | emitters: {df.Emitter.nunique()}'
          f' | files: {df.FileID.nunique()}')
    print(df.Context_name.value_counts().to_string())

    # Streaming pass — accumulate (seg_meta, mel_raw) with periodic flush
    print('\n[2/3] Streaming WAV → subseg → mel (tanh-normalized)...')
    metas = []
    mels = []     # list of (6, 32) float32 arrays
    total = 0
    last_flush = 0
    for meta, mel in stream_subsegs(df):
        metas.append(meta)
        mels.append(mel)
        total += 1
        if total - last_flush >= 100_000:
            flush_path = PROGRESS_DIR / f'flush_{total}.npz'
            np.savez_compressed(flush_path,
                mels=np.stack(mels[last_flush:], axis=0),
                metas=pd.DataFrame(metas[last_flush:]).to_records(index=False))
            last_flush = total
            print(f'  flushed → {flush_path.name} | total={total}')
            gc.collect()
    print(f'Total sub-segments: {total}')

    print('\n[3/3] Building seg_df and tf_specs, adapting Normalization...')
    seg_df = pd.DataFrame(metas).reset_index(drop=True)
    seg_df['pos_segment'] = seg_df.groupby('file_name').cumcount()
    tf_specs_raw = np.stack(mels, axis=0).astype(np.float32)
    del metas, mels; gc.collect()
    print(f'tf_specs_raw: {tf_specs_raw.shape}, mem≈{tf_specs_raw.nbytes/1e6:.0f} MB')

    # adapt Normalization on a random subsample, then apply to all
    rng = np.random.default_rng(RANDOM_STATE)
    n_adapt = min(50_000, len(tf_specs_raw))
    adapt_idx = rng.choice(len(tf_specs_raw), n_adapt, replace=False)
    norm = layers.Normalization(axis=-1)
    norm.adapt(tf_specs_raw[adapt_idx])
    tf_specs = norm(tf_specs_raw).numpy().astype(np.float32)
    del tf_specs_raw; gc.collect()
    print(f'tf_specs (normalized): {tf_specs.shape}')

    print(f'\nSaving to {OUT_PATH}...')
    joblib.dump({
        'seg_df': seg_df, 'tf_specs': tf_specs,
        'HDBSCAN_MIN_SAMPLES': 20, 'HDBSCAN_EPSILON': 0.1,
        'HDBSCAN_METHOD': 'leaf',
        'UMAP_N_NEIGHBORS': 30, 'UMAP_MIN_DIST': 0.3,
        'UMAP_METRIC': 'euclidean', 'UMAP_SEED': 0,
        'RANDOM_STATE': RANDOM_STATE,
    }, OUT_PATH, compress=3)
    print(f'Saved. {OUT_PATH.stat().st_size/1e6:.1f} MB | N={len(seg_df)}')


if __name__ == '__main__':
    main()
