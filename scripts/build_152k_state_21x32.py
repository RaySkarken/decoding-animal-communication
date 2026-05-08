"""Build 152k corpus with (21, 32) mel — matching Assom's TF_AE.ipynb actual code.

Paper Fig 2 caption says (6, 32) but their pickle 'DB_Isolated_segs__allSegments_MEL_21x32.pkl'
and code at L1254-1262 of TF_AE.ipynb use (21, 32). This produces 672-D feature
vectors instead of 192-D, giving better visual separation in UMAP.

Diff from build_152k_state.py:
    fft_size      8192  →  2048
    hop_size      8192  →  2048
    fft_length    16384 →  8192
    SPEC_TIME     6     →  21
    TARGET_AUDIO_LEN  49152  →  43008
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

SCRIPT_127K = Path(__file__).parent / 'build_full_corpus_state.py'
spec = importlib.util.spec_from_file_location('b127k', SCRIPT_127K)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Override mel params to match Assom's TF_AE.ipynb final visualization setup
mod.TF_FFT_SIZE   = 2048
mod.TF_HOP_SIZE   = 2048
mod.TF_FFT_LENGTH = 8192
mod.TF_N_MELS     = 32
mod.SPEC_TIME     = 21
mod.SPEC_FREQ     = 32
mod.TARGET_AUDIO_LEN = mod.SPEC_TIME * mod.TF_HOP_SIZE   # 21 * 2048 = 43008

mod.OUT_PATH      = mod.CHECKPOINT_DIR / 'ablation_state_152k_21x32.joblib'
mod.PROGRESS_DIR  = mod.CHECKPOINT_DIR / 'progress_152k_21x32'
mod.PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


# Paper-exact filter: only context filter (matches paper's 152.5k count)
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


mod.load_annotations = load_annotations_paper_exact


if __name__ == '__main__':
    print(f'Building 152k state with mel ({mod.SPEC_TIME}, {mod.SPEC_FREQ})')
    print(f'  fft={mod.TF_FFT_SIZE}, hop={mod.TF_HOP_SIZE}, fft_length={mod.TF_FFT_LENGTH}')
    print(f'  target_audio_len={mod.TARGET_AUDIO_LEN}')
    print(f'  output: {mod.OUT_PATH}')
    mod.main()
