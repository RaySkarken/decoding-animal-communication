"""
Last outstanding experiment: BPE / PMI on top of DP-UMAP (vocab ~20).

Hypothesis: previously BPE hurt HP1 F1 inv because it compressed sequences
too aggressively on a 10-atomic vocabulary, destroying ensemble stats.
With DP-UMAP giving ~20 atomics, each composite replaces a smaller fraction
of sequence tokens — BPE might not hurt as much, and might even help
context alignment.

If this experiment improves over both (a) DP-UMAP alone and (b) adds no
new destruction, we have a genuine candidate for best method.
"""
from __future__ import annotations

import sys, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score

from src.adaptive_tokenizer import (
    TokenizerState, Token, full_evaluation,
    BPEMerger, PMIMerger, SequenceMergerConfig,
)

CHECKPOINT_DIR = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CHECKPOINT_DIR / 'ablation_state.joblib')
mel = st['tf_specs'].reshape(-1, 192).astype(np.float32)
embedding = st['embedding']
hdbnca = st['hdb_nca_labels']
seg_df = st['seg_df']
RANDOM_STATE = st['RANDOM_STATE']

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
    Hs = []; pure_seg = 0; total = 0
    for tid in sorted(set(labels)):
        if tid < 0: continue
        members = np.where(labels == tid)[0]
        if len(members) == 0: continue
        ctx_cnt = Counter(contexts[members])
        probs = np.array(list(ctx_cnt.values())); probs = probs / probs.sum(); probs = probs[probs > 0]
        Hs.append(-np.sum(probs * np.log2(probs)))
        if max(ctx_cnt.values()) / len(members) >= 0.5:
            pure_seg += len(members)
        total += len(members)
    return dict(mean_H=float(np.mean(Hs)), context_purity=pure_seg/total if total else 0)


def eval_state(state, label, fit_space):
    mask = state.labels >= 0
    n = min(8000, int(mask.sum()))
    sil_u = float(silhouette_score(embedding[mask][:n], state.labels[mask][:n], random_state=0))
    sil_m = float(silhouette_score(mel[mask][:n], state.labels[mask][:n], random_state=0))
    native = mel if fit_space == 'mel' else embedding
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


# Get DP-UMAP state with k=40 (the best context-alignment result)
print('Fitting DP-UMAP k=40 base state...')
bgm = BayesianGaussianMixture(
    n_components=40,
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=0.1,
    covariance_type='full',
    max_iter=100,
    random_state=RANDOM_STATE,
)
dp_labels = bgm.fit_predict(embedding)
# prune tiny components
cnt = Counter(int(x) for x in dp_labels)
active = {k for k, v in cnt.items() if v >= 20}
if len(active) < len(cnt):
    active_mask = np.isin(dp_labels, list(active))
    tiny_mask = ~active_mask
    if tiny_mask.sum() > 0:
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=1).fit(embedding[active_mask])
        _, idx = knn.kneighbors(embedding[tiny_mask])
        active_lbl = dp_labels[active_mask]
        dp_labels[tiny_mask] = active_lbl[idx.ravel()]

state_dp = state_from_labels(dp_labels, embedding, sequences_per_file)
print(f'DP-UMAP base: vocab = {state_dp.vocab_size}')

rows = [eval_state(state_dp, 'DP-UMAP (base)', 'umap')]

for max_merges in [10, 30, 60]:
    for merger_name, merger_cls in [('BPE', BPEMerger), ('PMI', PMIMerger)]:
        print(f'{merger_name} max_merges={max_merges}...')
        s = copy.deepcopy(state_dp)
        cfg_args = dict(max_merges=max_merges, min_bigram_count=30,
                         min_sequences_containing=10, show_progress=False)
        if merger_name == 'PMI':
            cfg_args['pmi_threshold'] = 0.3
        merger_cls(SequenceMergerConfig(**cfg_args)).fit(s)
        rows.append(eval_state(s, f'DP-UMAP+{merger_name}-m{max_merges}', 'umap'))

df = pd.DataFrame(rows)
print('\n=== DP-UMAP + BPE/PMI ===')
print(df.to_string(index=False))

Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/dp_plus_bpe.csv', index=False)
print('\nSaved.')
