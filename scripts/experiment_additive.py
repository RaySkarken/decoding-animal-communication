"""
PURELY ADDITIVE A1 — only split + add, no merge + no prune.

Rationale: previous experiments showed both `merge` and `prune` can
destroy context-specific baseline clusters (Isolation 92%, Separation 86%
purity). Purely additive refinement preserves the baseline and only adds
new structure on top.

Hypotheses to test:
1. Purely additive should keep baseline's 6 clusters AND add splits of
   the messy shared cluster (5).
2. Silhouette may only marginally improve but context-purity should stay
   high or improve.
3. F1 bos / inv should improve because we ADD discriminating atoms
   without removing anything useful.
"""
from __future__ import annotations

import sys, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.metrics import silhouette_score

from src.adaptive_tokenizer import (
    AcousticTokenizer, AcousticTokenizerConfig,
    BPEMerger, PMIMerger, SequenceMergerConfig,
    TokenizerState, Token,
    full_evaluation,
)

CHECKPOINT_DIR = Path('/Volumes/T7/cache/assom_paper_repro')
state_in = joblib.load(CHECKPOINT_DIR / 'ablation_state.joblib')

seg_df = state_in['seg_df']
tf_specs = state_in['tf_specs']
embedding = state_in['embedding']
hdbnca_lbl = state_in['hdb_nca_labels']
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


def ctx_metrics(labels, contexts):
    Hs = []
    pure_seg = 0
    total = 0
    for tid in sorted(set(labels)):
        if tid < 0: continue
        members = np.where(labels == tid)[0]
        if len(members) == 0: continue
        ctx = contexts[members]
        counts = np.array(list(Counter(ctx).values()))
        probs = counts / counts.sum(); probs = probs[probs > 0]
        H = -np.sum(probs * np.log2(probs))
        Hs.append(H)
        top = counts.max() / counts.sum()
        if top >= 0.5:
            pure_seg += len(members)
        total += len(members)
    return dict(mean_H=float(np.mean(Hs)), context_purity=pure_seg/total if total else 0)


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
    cm = ctx_metrics(state.labels, contexts_per_segment)
    return {
        'method': label,
        'fit_space': fit_space,
        'vocab': state.vocab_size,
        'atomic': state.atomic_vocab_size,
        'comp': state.composite_vocab_size,
        'silh_Mel': round(sil_m, 3),
        'silh_UMAP': round(sil_u, 3),
        'ari_proxy': round(res.metrics.get('ari_proxy', np.nan), 3),
        'nmi_proxy': round(res.metrics.get('nmi_proxy', np.nan), 3),
        'ctx_H': round(cm['mean_H'], 2),
        'ctx_purity': round(cm['context_purity'], 3),
        'F1_bos': round(res.metrics.get('hp1_f1_original_bos', np.nan), 3),
        'F1_inv': round(res.metrics.get('hp1_f1_original_inv', np.nan), 3),
    }


ADDITIVE_COMMON = dict(
    seed_min_cluster_frac=0.01, seed_min_samples=20,
    enable_split=True, enable_add=True,
    enable_merge=False, enable_prune=False,   # <<< PURELY ADDITIVE
    split_silhouette_threshold=0.25,
    split_max_per_pass=10,
    add_min_size=50, add_outlier_quantile=0.95, add_min_silhouette=0.2,
    max_iterations=3,
    random_state=RANDOM_STATE, show_progress=False,
    silhouette_sample_size=5000,
)

rows = []

print('[1/6] Assom baseline (with context metrics)')
state_b = state_from_labels(hdbnca_lbl, embedding, sequences_per_file)
rows.append(eval_state(state_b, 'assom', 'umap'))

print('[2/6] A1-additive on UMAP')
t0 = time.time()
tok = AcousticTokenizer(AcousticTokenizerConfig(**ADDITIVE_COMMON))
state_au = tok.fit(embedding, seed_labels=hdbnca_lbl, sequences_per_file=sequences_per_file)
rows.append(eval_state(state_au, 'a1-add-umap', 'umap'))
print(f'     took {time.time()-t0:.1f}s, vocab={state_au.vocab_size}')

print('[3/6] A1-additive on Mel')
t0 = time.time()
tok = AcousticTokenizer(AcousticTokenizerConfig(**ADDITIVE_COMMON))
state_am = tok.fit(mel_flat, seed_labels=hdbnca_lbl, sequences_per_file=sequences_per_file)
rows.append(eval_state(state_am, 'a1-add-mel', 'mel'))
print(f'     took {time.time()-t0:.1f}s, vocab={state_am.vocab_size}')

print('[4/6] A1-additive-mel + BPE')
state_bpe = copy.deepcopy(state_am)
BPEMerger(SequenceMergerConfig(max_merges=30, min_bigram_count=30,
                                  min_sequences_containing=10, show_progress=False)).fit(state_bpe)
rows.append(eval_state(state_bpe, 'a1-add-mel+bpe', 'mel'))

print('[5/6] A1-additive-mel + PMI')
state_pmi = copy.deepcopy(state_am)
PMIMerger(SequenceMergerConfig(max_merges=30, min_bigram_count=30,
                                  min_sequences_containing=10, pmi_threshold=0.3, show_progress=False)).fit(state_pmi)
rows.append(eval_state(state_pmi, 'a1-add-mel+pmi', 'mel'))

print('[6/6] Aggressive additive: lower split_threshold, smaller min_size')
cfg_agg = dict(ADDITIVE_COMMON)
cfg_agg.update(dict(
    split_silhouette_threshold=0.40,    # more clusters qualify for split
    split_max_per_pass=15,
    add_min_size=25, add_outlier_quantile=0.85, add_min_silhouette=0.15,
    max_iterations=5,
))
t0 = time.time()
state_agg = AcousticTokenizer(AcousticTokenizerConfig(**cfg_agg)).fit(
    mel_flat, seed_labels=hdbnca_lbl, sequences_per_file=sequences_per_file)
rows.append(eval_state(state_agg, 'a1-add-mel-aggressive', 'mel'))
print(f'     took {time.time()-t0:.1f}s, vocab={state_agg.vocab_size}')

df = pd.DataFrame(rows)
print('\n=== ADDITIVE EXPERIMENT TABLE ===')
print(df.to_string(index=False))
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/additive_sweep.csv', index=False)
joblib.dump({'rows': rows, 'states': {
    'assom': state_b, 'a1-add-umap': state_au, 'a1-add-mel': state_am,
    'a1-add-mel+bpe': state_bpe, 'a1-add-mel+pmi': state_pmi,
    'a1-add-aggressive': state_agg,
}}, CHECKPOINT_DIR / 'additive_experiment.joblib', compress=3)
print('\nSaved: docs/thesis/figures/additive_sweep.csv')
