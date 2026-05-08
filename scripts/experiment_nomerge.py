"""
A1 WITHOUT merge operation — keep all baseline clusters, only split/add/prune.

Rationale: earlier runs showed acoustic-driven merge destroys
context-specific clusters (Isolation 92% purity got absorbed into a
large mixed mega-cluster in Mel space). Disabling merge preserves the
useful baseline structure while still allowing refinement within
clusters.

Writes docs/thesis/figures/nomerge_sweep.csv.
"""
from __future__ import annotations

import sys, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import silhouette_score

from src.adaptive_tokenizer import (
    AcousticTokenizer, AcousticTokenizerConfig,
    BPEMerger, PMIMerger, SequenceMergerConfig,
    TokenizerState, Token,
    full_evaluation,
)

CHECKPOINT_DIR = Path('/Volumes/T7/cache/assom_paper_repro')
state_in = joblib.load(CHECKPOINT_DIR / 'ablation_state.joblib')

seg_df       = state_in['seg_df']
tf_specs     = state_in['tf_specs']
embedding    = state_in['embedding']
hdbnca_lbl   = state_in['hdb_nca_labels']
RANDOM_STATE = state_in['RANDOM_STATE']

mel_flat = tf_specs.reshape(len(tf_specs), -1).astype(np.float32)

sequences_per_file, context_per_sequence = [], []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if len(seg_ids) < 2: continue
    sequences_per_file.append(seg_ids)
    context_per_sequence.append(int(np.bincount(g['context'].to_numpy()).argmax()))
emitters_per_segment = seg_df['emitter'].to_numpy()
contexts_per_segment = seg_df['context'].to_numpy()
proxy_labels = seg_df['proxy_label'].to_numpy() if 'proxy_label' in seg_df.columns else None


def state_from_labels(labels, X, sequences_per_file):
    tokens = {}
    for c in sorted(set(labels)):
        if c < 0: continue
        mids = np.where(labels == c)[0]
        tokens[int(c)] = Token(id=int(c), centroid=X[mids].mean(axis=0), member_ids=mids)
    sequences = [[int(labels[i]) for i in seg_ids if labels[i] >= 0]
                  for seg_ids in sequences_per_file]
    return TokenizerState(tokens=tokens, labels=np.asarray(labels, dtype=int), sequences=sequences)


def eval_state(state, label, fit_space):
    mask = state.labels >= 0
    n = min(8000, int(mask.sum()))
    sil_u = float(silhouette_score(embedding[mask][:n], state.labels[mask][:n], random_state=0))
    sil_m = float(silhouette_score(mel_flat[mask][:n], state.labels[mask][:n], random_state=0))
    native = mel_flat if fit_space == 'mel' else embedding
    res = full_evaluation(
        state, embedding=native,
        contexts_per_segment=contexts_per_segment,
        contexts_per_sequence=context_per_sequence,
        proxy_labels=proxy_labels, emitters=emitters_per_segment,
        run_hp1=True, hp1_feature_bundles=('bos', 'inv'),
        random_state=RANDOM_STATE, show_progress=False,
    )
    from collections import Counter
    cnt = Counter(int(x) for x in state.labels if x >= 0)
    sizes = sorted(cnt.values(), reverse=True)
    top_share = sizes[0] / sum(sizes)
    return {
        'method': label,
        'fit_space': fit_space,
        'vocab': state.vocab_size,
        'atomic': state.atomic_vocab_size,
        'comp': state.composite_vocab_size,
        'top_share': round(top_share, 3),
        'silh_UMAP': round(sil_u, 3),
        'silh_Mel':  round(sil_m, 3),
        'ari_proxy': round(res.metrics.get('ari_proxy', np.nan), 3),
        'nmi_proxy': round(res.metrics.get('nmi_proxy', np.nan), 3),
        'F1_bos': round(res.metrics.get('hp1_f1_original_bos', np.nan), 3),
        'F1_inv': round(res.metrics.get('hp1_f1_original_inv', np.nan), 3),
        'F1_bos_Δ': round((res.metrics.get('hp1_f1_original_bos', 0) -
                             res.metrics.get('hp1_f1_permuted_bos', 0)), 3),
        'F1_inv_Δ': round((res.metrics.get('hp1_f1_original_inv', 0) -
                             res.metrics.get('hp1_f1_permuted_inv', 0)), 3),
    }


COMMON_NOMERGE = dict(
    seed_min_cluster_frac=0.01, seed_min_samples=20,
    enable_merge=False,                      # <<< KEY CHANGE
    enable_add=True, enable_prune=True, enable_split=True,
    split_silhouette_threshold=0.25,
    add_min_size=50, add_outlier_quantile=0.95,
    prune_min_size=50,
    max_iterations=3,
    random_state=RANDOM_STATE, show_progress=False,
    silhouette_sample_size=5000,
)

rows = []

print('[1/5] Assom baseline')
state_b = state_from_labels(hdbnca_lbl, embedding, sequences_per_file)
rows.append(eval_state(state_b, 'assom', 'umap'))

print('[2/5] A1-nomerge on UMAP')
t0 = time.time()
tok = AcousticTokenizer(AcousticTokenizerConfig(**COMMON_NOMERGE))
state = tok.fit(embedding, seed_labels=hdbnca_lbl, sequences_per_file=sequences_per_file)
rows.append(eval_state(state, 'a1-nomerge-umap', 'umap'))
print(f'     took {time.time()-t0:.1f}s, vocab={state.vocab_size}')

print('[3/5] A1-nomerge on Mel-192D')
t0 = time.time()
tok = AcousticTokenizer(AcousticTokenizerConfig(**COMMON_NOMERGE))
state_nm_mel = tok.fit(mel_flat, seed_labels=hdbnca_lbl, sequences_per_file=sequences_per_file)
rows.append(eval_state(state_nm_mel, 'a1-nomerge-mel', 'mel'))
print(f'     took {time.time()-t0:.1f}s, vocab={state_nm_mel.vocab_size}')

print('[4/5] A1-nomerge-mel + BPE')
t0 = time.time()
state_bpe = copy.deepcopy(state_nm_mel)
BPEMerger(SequenceMergerConfig(
    max_merges=30, min_bigram_count=30, min_sequences_containing=10, show_progress=False,
)).fit(state_bpe)
rows.append(eval_state(state_bpe, 'a1-nomerge-mel+bpe', 'mel'))
print(f'     took {time.time()-t0:.1f}s, vocab={state_bpe.vocab_size}')

print('[5/5] A1-nomerge-mel + PMI')
t0 = time.time()
state_pmi = copy.deepcopy(state_nm_mel)
PMIMerger(SequenceMergerConfig(
    max_merges=30, min_bigram_count=30, min_sequences_containing=10,
    pmi_threshold=0.3, show_progress=False,
)).fit(state_pmi)
rows.append(eval_state(state_pmi, 'a1-nomerge-mel+pmi', 'mel'))
print(f'     took {time.time()-t0:.1f}s, vocab={state_pmi.vocab_size}')

df = pd.DataFrame(rows)
print('\n=== NO-MERGE SWEEP TABLE ===')
print(df.to_string(index=False))

Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/nomerge_sweep.csv', index=False)
joblib.dump({'rows': rows, 'states': {
    'assom': state_b,
    'a1-nomerge-mel': state_nm_mel,
    'a1-nomerge-mel+bpe': state_bpe,
    'a1-nomerge-mel+pmi': state_pmi,
}}, CHECKPOINT_DIR / 'nomerge_experiment.joblib', compress=3)
print('\nSaved: docs/thesis/figures/nomerge_sweep.csv')
