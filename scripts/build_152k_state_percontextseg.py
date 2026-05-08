"""Build 152k state with PER-CONTEXT segmentation parameters.

Hypothesis: different vocalization contexts have different acoustic profiles
(Isolation = long mom-pup calls; conflict = staccato; cooperative = brief).
Adapting dynamic_threshold params per context might extract more meaningful
sub-segments and shift the per-context tokenization advantage.

Per-context overrides:
  - Isolation (6): longer min_syllable_length, slower onset detection
  - Mating, Fighting, Threat-like (9, 4, 10): standard params
  - Feeding, Grooming, Kissing (3, 5, 7): shorter min length
  - Biting (2): standard
"""
from __future__ import annotations
import sys, importlib.util, gc, io, zipfile
from pathlib import Path
import numpy as np, pandas as pd
import soundfile as sf
from scipy import signal as scipy_signal
from tqdm.auto import tqdm

SCRIPT_127K = Path(__file__).parent / 'build_full_corpus_state.py'
spec = importlib.util.spec_from_file_location('b127k', SCRIPT_127K)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Override mel params (21,32)
mod.TF_FFT_SIZE = 2048; mod.TF_HOP_SIZE = 2048; mod.TF_FFT_LENGTH = 8192
mod.TF_N_MELS = 32; mod.SPEC_TIME = 21; mod.SPEC_FREQ = 32
mod.TARGET_AUDIO_LEN = 21 * 2048

mod.OUT_PATH = mod.CHECKPOINT_DIR / 'ablation_state_152k_21x32_pcseg.joblib'
mod.PROGRESS_DIR = mod.CHECKPOINT_DIR / 'progress_152k_21x32_pcseg'
mod.PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


# Per-context segmentation params
DEFAULT_DYN = mod.DYN_SEG_PARAMS.copy()
PER_CONTEXT_OVERRIDES = {
    # Isolation: long mom-pup calls, lenient
    6:  {'min_syllable_length_s': 0.03, 'silence_threshold': 0.07,
         'spectral_range': [2000, 60000]},
    # Conflict (Mating, Fighting, Threat-like): high-energy, broad spectrum
    4:  {'min_syllable_length_s': 0.015, 'silence_threshold': 0.10,
         'spectral_range': [3000, 80000]},
    9:  {'min_syllable_length_s': 0.015, 'silence_threshold': 0.10,
         'spectral_range': [3000, 80000]},
    10: {'min_syllable_length_s': 0.015, 'silence_threshold': 0.10,
         'spectral_range': [3000, 80000]},
    # Cooperative (Feeding, Grooming, Kissing): brief, finer granularity
    3:  {'min_syllable_length_s': 0.008, 'silence_threshold': 0.10,
         'spectral_range': [2000, 60000]},
    5:  {'min_syllable_length_s': 0.008, 'silence_threshold': 0.10,
         'spectral_range': [2000, 60000]},
    7:  {'min_syllable_length_s': 0.008, 'silence_threshold': 0.10,
         'spectral_range': [2000, 60000]},
    2:  {'min_syllable_length_s': 0.010, 'silence_threshold': 0.10,
         'spectral_range': [2000, 60000]},
}


def get_dyn_params(context_id):
    p = DEFAULT_DYN.copy()
    if context_id in PER_CONTEXT_OVERRIDES:
        p.update(PER_CONTEXT_OVERRIDES[context_id])
    return p


# Per-context-segmentation overrides extract_subsegs to know context
def extract_subsegs_pc(audio, context):
    a = mod.preprocess_audio(audio, mod.SR)
    params = get_dyn_params(context)
    try:
        from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation
        r = dynamic_threshold_segmentation(a, mod.SR, **params)
    except Exception:
        r = None
    subs = []
    if r is not None and len(r.get('onsets', [])) > 0:
        for onset_s, offset_s in zip(r['onsets'], r['offsets']):
            si, ei = int(onset_s * mod.SR), int(offset_s * mod.SR)
            sub = a[si:ei]
            if len(sub) >= 50:
                subs.append((sub, si, ei))
    if not subs:
        subs.append((a, 0, len(a)))
    return subs


# Override the streaming function with per-context dispatch
def stream_subsegs_pc(df):
    log_mel_layer = mod.LogMelSpectrogram(
        sample_rate=mod.SR, fft_size=mod.TF_FFT_SIZE, hop_size=mod.TF_HOP_SIZE,
        fft_length=mod.TF_FFT_LENGTH,
        window_fn=__import__('tensorflow').signal.hamming_window,
        n_mels=mod.TF_N_MELS, f_min=mod.TF_FMIN, f_max=mod.TF_FMAX,
        normalize=mod.TF_NORMALIZE, name='LogMel')
    import tensorflow as tf
    from tensorflow.keras import layers
    reshape = layers.Reshape((1, -1))

    zip_cache = {}
    def get_zip(folder):
        if folder not in zip_cache:
            zp = mod.DATA_DIR / f'{folder}.zip'
            zip_cache[folder] = zipfile.ZipFile(zp, 'r') if zp.exists() else None
        return zip_cache[folder]

    skipped = 0
    grouped = df.groupby('FileID')
    try:
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

            sub_audios, sub_metas = [], []
            for _, r in group.iterrows():
                s, e = int(r['Start sample']), int(r['End sample'])
                if e > len(audio_full) or s >= e:
                    skipped += 1; continue
                seg = audio_full[s:e].astype(np.float32)
                if len(seg) < 100:
                    skipped += 1; continue
                if file_sr != mod.SR:
                    seg = scipy_signal.resample(seg, int(len(seg)*mod.SR/file_sr)).astype(np.float32)
                ctx_id = int(r['Context'])
                for sub, si, ei in extract_subsegs_pc(seg, ctx_id):
                    if len(sub) < 50: continue
                    sub_audios.append(mod._pad_or_crop(sub, mod.TARGET_AUDIO_LEN))
                    sub_metas.append({
                        'context': ctx_id,
                        'context_name': r['Context_name'],
                        'emitter': int(r['Emitter']),
                        'addressee': int(r['Addressee']) if pd.notna(r['Addressee']) else -1,
                        'file_name': fname, 'file_id': int(file_id),
                        'duration_s': len(sub) / mod.SR,
                        'parent_start': int(s)+si, 'parent_end': int(s)+ei,
                    })
            if not sub_audios: continue

            batch = np.stack(sub_audios, axis=0).astype(np.float32)
            with tf.device('/CPU:0'):
                mel_batch = log_mel_layer(reshape(tf.constant(batch))).numpy()
            for i in range(len(mel_batch)):
                m = mel_batch[i]
                if m.shape[0] == mod.SPEC_TIME:
                    out = m
                elif m.shape[0] > mod.SPEC_TIME:
                    out = m[:mod.SPEC_TIME]
                else:
                    out = np.zeros((mod.SPEC_TIME, mod.SPEC_FREQ), dtype=np.float32)
                    out[:m.shape[0]] = m
                yield sub_metas[i], out.astype(np.float32)
            del audio_full, sub_audios, sub_metas, batch, mel_batch
    finally:
        for z in zip_cache.values():
            if z is not None: z.close()
    print(f'Skipped: {skipped}')


def load_annotations_paper_exact():
    ann = pd.read_csv(mod.DATA_DIR / 'Annotations.csv', low_memory=False)
    with open(mod.DATA_DIR / 'FileInfo.csv') as f:
        max_cols = max(len(l.split(',')) for l in f)
    fi = pd.read_csv(mod.DATA_DIR / 'FileInfo.csv', header=None,
                     names=[f'c{i}' for i in range(max_cols)], low_memory=False)
    fi.columns = fi.iloc[0].values
    fi = fi.iloc[1:].reset_index(drop=True)
    fi['FileID'] = fi['FileID'].astype(int)
    fi = fi[['FileID', 'File name', 'File folder']].drop_duplicates('FileID')
    ann['FileID'] = ann['FileID'].astype(int)
    df = ann.merge(fi, on='FileID', how='inner')
    df['Emitter']   = pd.to_numeric(df['Emitter'], errors='coerce')
    df['Addressee'] = pd.to_numeric(df['Addressee'], errors='coerce')
    df['Context']   = pd.to_numeric(df['Context'], errors='coerce')
    df = df.dropna(subset=['Emitter', 'Context', 'Start sample', 'End sample'])
    df['Emitter'] = df['Emitter'].astype(int)
    df['Context'] = df['Context'].astype(int)
    df['Start sample'] = df['Start sample'].astype(int)
    df['End sample']   = df['End sample'].astype(int)
    df = df[~df['Context'].isin(mod.EXCLUDED_CONTEXTS)]
    df['Context_name'] = df['Context'].map(mod.CONTEXT_DICT)
    return df


def main_pc():
    print('=== Per-context segmentation build ===')
    print(f'  Output: {mod.OUT_PATH}')
    df = load_annotations_paper_exact()
    print(f'Annotations: {len(df)}, files: {df.FileID.nunique()}')
    print(df.Context_name.value_counts().to_string())

    metas, mels = [], []
    total = 0
    for meta, mel in stream_subsegs_pc(df):
        metas.append(meta); mels.append(mel); total += 1
    print(f'Total: {total}')

    seg_df = pd.DataFrame(metas).reset_index(drop=True)
    seg_df['pos_segment'] = seg_df.groupby('file_name').cumcount()
    tf_specs_raw = np.stack(mels, axis=0).astype(np.float32)
    del metas, mels; gc.collect()

    rng = np.random.default_rng(0)
    n_adapt = min(50_000, len(tf_specs_raw))
    adapt_idx = rng.choice(len(tf_specs_raw), n_adapt, replace=False)
    import tensorflow as tf
    from tensorflow.keras import layers
    norm = layers.Normalization(axis=-1)
    norm.adapt(tf_specs_raw[adapt_idx])
    tf_specs = norm(tf_specs_raw).numpy().astype(np.float32)
    del tf_specs_raw; gc.collect()

    import joblib
    joblib.dump({'seg_df': seg_df, 'tf_specs': tf_specs,
                 'PER_CONTEXT_OVERRIDES': PER_CONTEXT_OVERRIDES},
                mod.OUT_PATH, compress=3)
    print(f'Saved: {mod.OUT_PATH}  ({mod.OUT_PATH.stat().st_size/1e6:.0f} MB)')

    # Per-context segment count breakdown
    by_ctx = seg_df.groupby('context_name').size()
    print('\nSegment count per context (per-context-seg pipeline):')
    print(by_ctx.to_string())


if __name__ == '__main__':
    main_pc()
