"""Build 152k corpus state — paper-exact filter (only contexts excluded, NO emitter filter).

Streaming version that fits into < 3 GB RAM on a laptop. Output is read by
notebooks/assom_full_reproduction_152k.ipynb.

This is just `build_full_corpus_state.py` with the line
    df = df[df['Emitter'] != 0]
removed, and a new output path. Reuses all existing helper functions.
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path

# Reuse helpers from the 127k builder
SCRIPT_127K = Path(__file__).with_name('build_full_corpus_state.py')
spec = importlib.util.spec_from_file_location('b127k', SCRIPT_127K)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Override constants for 152k build
mod.OUT_PATH = mod.CHECKPOINT_DIR / 'ablation_state_152k.joblib'
mod.PROGRESS_DIR = mod.CHECKPOINT_DIR / 'progress_152k'
mod.PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

# Override load_annotations to NOT filter on Emitter — paper-exact
import pandas as pd, numpy as np
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
    # ★ PAPER-EXACT: only context filter, no emitter filter (id=0 included)
    df = df[~df['Context'].isin(mod.EXCLUDED_CONTEXTS)]
    df['Context_name'] = df['Context'].map(mod.CONTEXT_DICT)
    return df

mod.load_annotations = load_annotations_paper_exact

if __name__ == '__main__':
    mod.main()
