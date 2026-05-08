"""
Deeper experiment on Mel-192D space:
1. A1 with different max_iterations → track vocab growth
2. A1 + BPE/PMI → do mergers still hurt in Mel space?
3. Compare against Assom baseline on BOTH spaces consistently.

Writes docs/thesis/figures/mel_space_sweep.csv (for thesis table).
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

sequences_per_file = []
context_per_sequence = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if len(seg_ids) < 2:
        continue
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


def eval_both_spaces(state, label, method_space):
    # Silhouette on both
    mask = state.labels >= 0
    n = min(8000, int(mask.sum()))
    sil_u = float(silhouette_score(embedding[mask][:n], state.labels[mask][:n], random_state=0))
    sil_m = float(silhouette_score(mel_flat[mask][:n], state.labels[mask][:n], random_state=0))

    # Use the NATIVE space for eval embedding arg
    native = mel_flat if method_space == 'mel' else embedding
    res = full_evaluation(
        state, embedding=native,
        contexts_per_segment=contexts_per_segment,
        contexts_per_sequence=context_per_sequence,
        proxy_labels=proxy_labels, emitters=emitters_per_segment,
        run_hp1=True, hp1_feature_bundles=('bos', 'inv'),
        random_state=RANDOM_STATE, show_progress=False,
    )
    return {
        'method': label,
        'fit_space': method_space,
        'vocab': state.vocab_size,
        'atomic': state.atomic_vocab_size,
        'comp': state.composite_vocab_size,
        'silh_UMAP': round(sil_u, 3),
        'silh_Mel': round(sil_m, 3),
        'ari_proxy': round(res.metrics.get('ari_proxy', np.nan), 3),
        'nmi_proxy': round(res.metrics.get('nmi_proxy', np.nan), 3),
        'F1_bos_orig': round(res.metrics.get('hp1_f1_original_bos', np.nan), 3),
        'F1_inv_orig': round(res.metrics.get('hp1_f1_original_inv', np.nan), 3),
        'F1_bos_perm': round(res.metrics.get('hp1_f1_permuted_bos', np.nan), 3),
        'F1_inv_perm': round(res.metrics.get('hp1_f1_permuted_inv', np.nan), 3),
    }


rows = []

# 1. Assom baseline (in both spaces for reference)
print('[1/6] Assom baseline')
t0 = time.time()
state_b = state_from_labels(hdbnca_lbl, embedding, sequences_per_file)
rows.append(eval_both_spaces(state_b, 'assom', 'umap'))
print(f'     took {time.time()-t0:.1f}s')

# 2. A1 on UMAP (as in the earlier sweep)
print('[2/6] A1 on UMAP')
t0 = time.time()
tok = AcousticTokenizer(AcousticTokenizerConfig(
    seed_min_cluster_frac=0.01, seed_min_samples=20,
    split_silhouette_threshold=0.25,
    add_min_size=50, prune_min_size=50,
    max_iterations=3,
    random_state=RANDOM_STATE, show_progress=False,
    silhouette_sample_size=5000,
))
state_au = tok.fit(embedding, seed_labels=hdbnca_lbl, sequences_per_file=sequences_per_file)
rows.append(eval_both_spaces(state_au, 'a1-on-umap', 'umap'))
print(f'     took {time.time()-t0:.1f}s, vocab={state_au.vocab_size}')

# 3. A1 on Mel-192D
print('[3/6] A1 on Mel-192D')
t0 = time.time()
tok = AcousticTokenizer(AcousticTokenizerConfig(
    seed_min_cluster_frac=0.01, seed_min_samples=20,
    split_silhouette_threshold=0.25,
    add_min_size=50, prune_min_size=50,
    max_iterations=3,
    random_state=RANDOM_STATE, show_progress=False,
    silhouette_sample_size=5000,
))
state_am = tok.fit(mel_flat, seed_labels=hdbnca_lbl, sequences_per_file=sequences_per_file)
rows.append(eval_both_spaces(state_am, 'a1-on-mel', 'mel'))
print(f'     took {time.time()-t0:.1f}s, vocab={state_am.vocab_size}')

# 4. A1 on Mel + BPE
print('[4/6] A1 on Mel + BPE')
t0 = time.time()
state_ambpe = copy.deepcopy(state_am)
bpe = BPEMerger(SequenceMergerConfig(
    max_merges=30, min_bigram_count=30, min_sequences_containing=10,
    show_progress=False,
))
bpe.fit(state_ambpe)
rows.append(eval_both_spaces(state_ambpe, 'a1-on-mel+bpe', 'mel'))
print(f'     took {time.time()-t0:.1f}s, vocab={state_ambpe.vocab_size}')

# 5. A1 on Mel + PMI
print('[5/6] A1 on Mel + PMI')
t0 = time.time()
state_ampmi = copy.deepcopy(state_am)
pmi = PMIMerger(SequenceMergerConfig(
    max_merges=30, min_bigram_count=30, min_sequences_containing=10,
    pmi_threshold=0.3, show_progress=False,
))
pmi.fit(state_ampmi)
rows.append(eval_both_spaces(state_ampmi, 'a1-on-mel+pmi', 'mel'))
print(f'     took {time.time()-t0:.1f}s, vocab={state_ampmi.vocab_size}')

# 6. A1 on Mel with more aggressive add/split (force vocab growth)
print('[6/6] A1 on Mel aggressive (more iters, smaller min_size)')
t0 = time.time()
tok = AcousticTokenizer(AcousticTokenizerConfig(
    seed_min_cluster_frac=0.005, seed_min_samples=15,    # smaller mcs
    split_silhouette_threshold=0.35,                      # higher → more candidates
    split_max_per_pass=10,
    add_min_size=25, add_outlier_quantile=0.85,          # more outliers
    add_min_silhouette=0.15,                              # lower bar
    prune_min_size=25,
    max_iterations=5,
    random_state=RANDOM_STATE, show_progress=False,
    silhouette_sample_size=5000,
))
state_am_agg = tok.fit(mel_flat, seed_labels=hdbnca_lbl, sequences_per_file=sequences_per_file)
rows.append(eval_both_spaces(state_am_agg, 'a1-on-mel-aggressive', 'mel'))
print(f'     took {time.time()-t0:.1f}s, vocab={state_am_agg.vocab_size}')

df = pd.DataFrame(rows)
print('\n=== MEL-SPACE EXPERIMENT TABLE ===')
print(df.to_string(index=False))

Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/mel_space_sweep.csv', index=False)
print('\nSaved: docs/thesis/figures/mel_space_sweep.csv')

# Save states for potential reuse
joblib.dump({'rows': rows, 'states': {
    'assom': state_b, 'a1-umap': state_au, 'a1-mel': state_am,
    'a1-mel+bpe': state_ambpe, 'a1-mel+pmi': state_ampmi,
    'a1-mel-aggressive': state_am_agg,
}}, CHECKPOINT_DIR / 'mel_space_experiment.joblib', compress=3)
