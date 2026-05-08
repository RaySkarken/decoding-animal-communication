"""
DP-HDP (Dirichlet Process Mixture Model) alternative to HDBSCAN-based
adaptive tokenization.

Implementation: sklearn's BayesianGaussianMixture with
weight_concentration_prior_type='dirichlet_process' — this is exactly
DP-GMM, the single-level version of the Hierarchical Dirichlet Process.
Vocabulary size emerges from data via the concentration prior.

Tests multiple configurations:
1. Different max_components (upper bound on # clusters)
2. Different concentration priors (higher = more clusters expected)
3. On UMAP 2D AND Mel 192D
"""
from __future__ import annotations

import sys, time
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


def eval_dp(labels, label_name, fit_space):
    native = mel if fit_space == 'mel' else embedding
    state = state_from_labels(labels, native, sequences_per_file)
    mask = state.labels >= 0
    n = min(8000, int(mask.sum()))
    sil_u = float(silhouette_score(embedding[mask][:n], state.labels[mask][:n], random_state=0))
    sil_m = float(silhouette_score(mel[mask][:n], state.labels[mask][:n], random_state=0))
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
        'method': label_name,
        'fit_space': fit_space,
        'vocab': len(set(state.labels)) - (1 if -1 in state.labels else 0),
        'silh_Mel': round(sil_m, 3),
        'silh_UMAP': round(sil_u, 3),
        'ari_proxy': round(res.metrics.get('ari_proxy', np.nan), 3),
        'nmi_proxy': round(res.metrics.get('nmi_proxy', np.nan), 3),
        'ctx_H': round(cm['mean_H'], 2),
        'ctx_purity': round(cm['context_purity'], 3),
        'F1_bos': round(res.metrics.get('hp1_f1_original_bos', np.nan), 3),
        'F1_inv': round(res.metrics.get('hp1_f1_original_inv', np.nan), 3),
    }


def run_dp(X, max_components, concentration, seed=0, max_iter=100):
    bgm = BayesianGaussianMixture(
        n_components=max_components,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=concentration,
        covariance_type='full',
        max_iter=max_iter,
        random_state=seed,
        warm_start=False,
    )
    labels = bgm.fit_predict(X)
    # drop components with < 20 members (numerically inactive)
    cnt = Counter(int(x) for x in labels)
    active = {k for k, v in cnt.items() if v >= 20}
    # remap: keep active as-is, assign tiny to nearest active centroid
    if len(active) < len(cnt):
        active_ids = sorted(active)
        tiny_ids = [k for k in cnt if k not in active]
        if tiny_ids and active_ids:
            from sklearn.neighbors import NearestNeighbors
            active_mask = np.isin(labels, list(active))
            knn = NearestNeighbors(n_neighbors=1).fit(X[active_mask])
            tiny_mask = ~active_mask
            if tiny_mask.sum() > 0:
                _, idx = knn.kneighbors(X[tiny_mask])
                # Get the label of the nearest active point
                active_label = labels[active_mask]
                assigned = active_label[idx.ravel()]
                labels[tiny_mask] = assigned
    return labels


rows = []
rows.append(eval_dp(hdbnca, 'assom (baseline)', 'umap'))

for space_name, X, fit_space in [('UMAP-2D', embedding, 'umap'),
                                    ('Mel-192D', mel, 'mel')]:
    for max_k, conc in [(20, 0.1), (20, 1.0), (20, 5.0),
                         (40, 0.1), (40, 1.0)]:
        t0 = time.time()
        try:
            labels = run_dp(X, max_components=max_k, concentration=conc, seed=RANDOM_STATE)
            lbl_name = f'DP-{space_name}-k{max_k}-c{conc}'
            rows.append(eval_dp(labels, lbl_name, fit_space))
            vocab = rows[-1]['vocab']
            print(f'{lbl_name}: vocab={vocab}, took {time.time()-t0:.1f}s')
        except Exception as e:
            print(f'DP-{space_name}-k{max_k}-c{conc}: FAILED ({e})')

df = pd.DataFrame(rows)
print('\n=== DP-GMM SWEEP ===')
print(df.to_string(index=False))
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/dp_hdp_sweep.csv', index=False)
print('\nSaved: docs/thesis/figures/dp_hdp_sweep.csv')
