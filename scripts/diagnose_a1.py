"""
Diagnose why A1-only's silhouette drops below the Assom baseline, and
sweep configurations to find one that genuinely improves.

Produces:
  - Per-cluster silhouette on both the 2D UMAP embedding AND the 192-D
    mel features, for the baseline (6 clusters) and a1-only (10 clusters).
    This tells us whether the drop is real or an artefact of UMAP distance
    distortion.
  - Sweep of A1 over:
        * embedding choice: UMAP 2D vs raw 192-D mel
        * split_silhouette_threshold ∈ {0.25, 0.35, 0.40, 0.50}
    reporting silhouette, ARI vs proxy, HP1 F1 inv at each point.

Uses checkpointed state from ablation_state.joblib.
"""
from __future__ import annotations

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib

from src.adaptive_tokenizer import (
    AcousticTokenizer, AcousticTokenizerConfig,
    TokenizerState, Token,
    full_evaluation,
)

CHECKPOINT_DIR = Path('/Volumes/T7/cache/assom_paper_repro')
state_in = joblib.load(CHECKPOINT_DIR / 'ablation_state.joblib')

seg_df       = state_in['seg_df']
tf_specs     = state_in['tf_specs']            # (N, 6, 32)
embedding    = state_in['embedding']            # (N, 2) UMAP
hdbnca_lbl   = state_in['hdb_nca_labels']       # (N,)
SEQUENCES    = state_in['SEQUENCES']
RANDOM_STATE = state_in['RANDOM_STATE']

mel_flat = tf_specs.reshape(len(tf_specs), -1).astype(np.float32)  # (N, 192)

# Per-file sequences / contexts
sequences_per_file = []
context_per_sequence = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if len(seg_ids) < 2:
        continue
    sequences_per_file.append(seg_ids)
    ctx_vals = g['context'].to_numpy()
    context_per_sequence.append(int(np.bincount(ctx_vals).argmax()))
emitters_per_segment = seg_df['emitter'].to_numpy()
contexts_per_segment = seg_df['context'].to_numpy()
proxy_labels = seg_df['proxy_label'].to_numpy() if 'proxy_label' in seg_df.columns else None


def state_from_labels(labels, X, sequences_per_file):
    from src.adaptive_tokenizer.types import Token
    tokens = {}
    for c in sorted(set(labels)):
        if c < 0: continue
        mids = np.where(labels == c)[0]
        tokens[int(c)] = Token(id=int(c), centroid=X[mids].mean(axis=0), member_ids=mids)
    sequences = []
    for seg_ids in sequences_per_file:
        tok_seq = [int(labels[i]) for i in seg_ids if labels[i] >= 0]
        sequences.append(tok_seq)
    return TokenizerState(tokens=tokens, labels=np.asarray(labels, dtype=int), sequences=sequences)


# ───────────────────────────────────────────────────────────
# DIAGNOSIS 1 — per-cluster silhouette on both spaces for baseline
# ───────────────────────────────────────────────────────────
from sklearn.metrics import silhouette_samples, silhouette_score

def per_cluster_silh(X, labels, sample_size=8000, rs=0):
    mask = labels >= 0
    Xg, yg = X[mask], labels[mask]
    if sample_size < len(Xg):
        rng = np.random.default_rng(rs)
        idx = rng.choice(len(Xg), size=sample_size, replace=False)
        Xg, yg = Xg[idx], yg[idx]
    s_samp = silhouette_samples(Xg, yg)
    out = {}
    for c in np.unique(yg):
        out[int(c)] = float(s_samp[yg == c].mean())
    return out


print('='*70)
print('DIAGNOSIS 1 — per-cluster silhouette, baseline (6 clusters)')
print('='*70)
for space_name, X_in in [('UMAP 2D', embedding), ('Mel 192D', mel_flat)]:
    per = per_cluster_silh(X_in, hdbnca_lbl)
    mean_ = float(np.mean(list(per.values())))
    print(f'\\n{space_name}:  mean silh = {mean_:.3f}')
    for cid, s in sorted(per.items()):
        N_c = int((hdbnca_lbl == cid).sum())
        print(f'  cluster {cid}: silh = {s:+.3f}   N = {N_c}')


# ───────────────────────────────────────────────────────────
# EXPERIMENT 1 — A1 on UMAP vs on Mel, across split thresholds
# ───────────────────────────────────────────────────────────
print('\n' + '='*70)
print('EXPERIMENT 1 — sweep split threshold × embedding space')
print('='*70)

rows = []
for space_name, X_in in [('UMAP 2D', embedding), ('Mel 192D', mel_flat)]:
    for thresh in [0.25, 0.35, 0.40, 0.50]:
        t0 = time.time()
        cfg = AcousticTokenizerConfig(
            seed_min_cluster_frac=0.01, seed_min_samples=20,
            split_silhouette_threshold=thresh,
            add_min_size=50, prune_min_size=50,
            max_iterations=2,
            random_state=RANDOM_STATE,
            show_progress=False,
            silhouette_sample_size=5000,
        )
        tok = AcousticTokenizer(cfg)
        state = tok.fit(X_in, seed_labels=hdbnca_lbl, sequences_per_file=sequences_per_file)
        dt = time.time() - t0

        # Evaluate: silhouette on BOTH spaces
        silh_u = float(silhouette_score(embedding[state.labels >= 0][:8000],
                                          state.labels[state.labels >= 0][:8000], random_state=0))
        silh_m = float(silhouette_score(mel_flat[state.labels >= 0][:8000],
                                          state.labels[state.labels >= 0][:8000], random_state=0))

        # HP1 F1 inv via full_evaluation — quick
        res = full_evaluation(
            state, embedding=X_in,
            contexts_per_sequence=context_per_sequence,
            proxy_labels=proxy_labels, emitters=emitters_per_segment,
            run_hp1=True, hp1_feature_bundles=('inv',),
            random_state=RANDOM_STATE, show_progress=False,
        )

        rows.append({
            'space': space_name,
            'split_thresh': thresh,
            'vocab': state.vocab_size,
            'silh_UMAP': round(silh_u, 3),
            'silh_Mel': round(silh_m, 3),
            'ari_proxy': round(res.metrics.get('ari_proxy', np.nan), 3),
            'nmi_proxy': round(res.metrics.get('nmi_proxy', np.nan), 3),
            'hp1_F1_inv': round(res.metrics.get('hp1_f1_original_inv', np.nan), 3),
            'time_s': round(dt, 1),
        })
        print(f'{space_name:10s} thr={thresh:.2f}  vocab={state.vocab_size:3d}  '
              f'silh_U={silh_u:.3f}  silh_M={silh_m:.3f}  '
              f'ARI={res.metrics.get("ari_proxy", 0):.3f}  '
              f'F1inv={res.metrics.get("hp1_f1_original_inv", 0):.3f}  ({dt:.0f}s)')

# add baseline row
state_b = state_from_labels(hdbnca_lbl, embedding, sequences_per_file)
silh_u_b = float(silhouette_score(embedding[hdbnca_lbl >= 0][:8000],
                                     hdbnca_lbl[hdbnca_lbl >= 0][:8000], random_state=0))
silh_m_b = float(silhouette_score(mel_flat[hdbnca_lbl >= 0][:8000],
                                     hdbnca_lbl[hdbnca_lbl >= 0][:8000], random_state=0))
res_b = full_evaluation(
    state_b, embedding=embedding,
    contexts_per_sequence=context_per_sequence,
    proxy_labels=proxy_labels, emitters=emitters_per_segment,
    run_hp1=True, hp1_feature_bundles=('inv',),
    random_state=RANDOM_STATE, show_progress=False,
)
rows.append({
    'space': 'BASELINE',
    'split_thresh': None,
    'vocab': 6,
    'silh_UMAP': round(silh_u_b, 3),
    'silh_Mel': round(silh_m_b, 3),
    'ari_proxy': round(res_b.metrics.get('ari_proxy', np.nan), 3),
    'nmi_proxy': round(res_b.metrics.get('nmi_proxy', np.nan), 3),
    'hp1_F1_inv': round(res_b.metrics.get('hp1_f1_original_inv', np.nan), 3),
    'time_s': 0,
})

print('\n=== SWEEP TABLE ===')
df = pd.DataFrame(rows)
print(df.to_string(index=False))

out_path = CHECKPOINT_DIR / 'a1_sweep_results.csv'
df.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')
