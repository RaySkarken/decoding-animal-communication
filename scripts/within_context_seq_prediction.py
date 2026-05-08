"""Within-context sequence prediction (downstream task: language modelling of bat syllable sequences).

Hypothesis: per-context tokenizer should model syllable sequences within its own
context BETTER than global tokenizer because tokens are specifically tuned for
that context's acoustic structure.

Experiment:
  For each context c with ≥100 vocalizations:
    - 5 random splits 80/20 by file
    - For each split:
      - Train bigram Markov model on train (Laplace smoothing)
      - Compute test cross-entropy (bits/token)
    - Compare two vocabularies: GLOBAL vs PER-CONTEXT (token IDs from same files)

Reported:
  - bits per token (lower = better compression)
  - perplexity = 2^(bits/token)
  - LL_total per vocalization (higher = better)
  - Compression ratio = bits_per_token / log2(V) (cardinality-fair)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib
from pathlib import Path
from collections import Counter, defaultdict
from itertools import pairwise
from sklearn.mixture import BayesianGaussianMixture

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
hdb_nca = np.load(CACHE / 'hdb_nca_labels_152k_21x32.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
HP1_CTX = [2,3,4,5,6,7,9,10]
CTX = {2:'Biting',3:'Feeding',4:'Fighting',5:'Grooming',6:'Isolation',7:'Kissing',9:'Mating',10:'Threat'}

# Per-context labels
labels_pc = np.full(len(seg_df), -1, dtype=np.int32)
offset = 0
for c in HP1_CTX:
    mc = ctx == c
    if mc.sum() < 30: continue
    bgm = BayesianGaussianMixture(n_components=15,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1, covariance_type='full',
        max_iter=200, random_state=0).fit(emb[mc])
    labels_pc[mc] = bgm.predict(emb[mc]) + offset
    offset += bgm.n_components

# Build sequences per file (only HP1 contexts, identified bats)
print('Building sequences...')
def build_seqs(label_arr):
    seqs = []
    for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
        seg_ids = g.index.to_list()
        if not seg_ids: continue
        dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
        if dom_em == 0: continue
        dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
        if dom_ctx not in HP1_CTX: continue
        seq = [int(label_arr[i]) for i in seg_ids if label_arr[i] >= 0]
        if len(seq) < 2: continue
        seqs.append({'file': fname, 'context': dom_ctx, 'seq': seq})
    return pd.DataFrame(seqs)

seqs_global = build_seqs(hdb_nca)
seqs_pc = build_seqs(labels_pc)
print(f'  global: {len(seqs_global)} seqs, vocab={len(set(s for seq in seqs_global.seq for s in seq))}')
print(f'  per-context: {len(seqs_pc)} seqs, vocab={len(set(s for seq in seqs_pc.seq for s in seq))}')


def fit_bigram(train_seqs, vocab_set, alpha=1.0):
    """Fit bigram Markov model with Laplace smoothing."""
    V = len(vocab_set)
    vocab_list = sorted(vocab_set)
    v_to_i = {v: i for i, v in enumerate(vocab_list)}
    # transition counts
    N = np.zeros((V, V), dtype=np.float64) + alpha   # Laplace
    # initial counts
    N0 = np.zeros(V, dtype=np.float64) + alpha
    for seq in train_seqs:
        if not seq: continue
        N0[v_to_i[seq[0]]] += 1
        for a, b in pairwise(seq):
            if a in v_to_i and b in v_to_i:
                N[v_to_i[a], v_to_i[b]] += 1
    # normalize
    P0 = N0 / N0.sum()
    P = N / N.sum(axis=1, keepdims=True)
    return {'V':V, 'vocab_list':vocab_list, 'v_to_i':v_to_i, 'P0':P0, 'P':P}


def eval_bigram(model, test_seqs):
    """Return total bits, total tokens, per-token cross-entropy."""
    total_bits = 0.0
    total_tokens = 0
    v_to_i = model['v_to_i']
    P0 = model['P0']; P = model['P']
    for seq in test_seqs:
        if not seq: continue
        # OOV handling: use uniform 1/V probability for unknown tokens
        if seq[0] not in v_to_i:
            total_bits += np.log2(model['V'])
        else:
            total_bits += -np.log2(max(P0[v_to_i[seq[0]]], 1e-12))
        total_tokens += 1
        for a, b in pairwise(seq):
            if a in v_to_i and b in v_to_i:
                total_bits += -np.log2(max(P[v_to_i[a], v_to_i[b]], 1e-12))
            else:
                total_bits += np.log2(model['V'])
            total_tokens += 1
    return total_bits, total_tokens, total_bits / max(total_tokens, 1)


# === Per-context within-context sequence prediction ===
print('\n══════ WITHIN-CONTEXT SEQUENCE PREDICTION ══════')
print(f'{"context":12s} {"n_vocs":>7s} {"|V|_g":>6s} {"|V|_pc":>7s} {"bits/tok GLOBAL":>16s} {"bits/tok PC":>13s} {"Δ":>7s} {"better":>8s}')
print('-'*100)

results = []
N_SPLITS = 5
rng_master = np.random.default_rng(0)

for c in HP1_CTX:
    sg = seqs_global[seqs_global.context == c]
    sp = seqs_pc[seqs_pc.context == c]
    # Same files in both — match on filename (should be already)
    common_files = set(sg.file) & set(sp.file)
    sg_dict = {row['file']: row['seq'] for _, row in sg.iterrows() if row['file'] in common_files}
    sp_dict = {row['file']: row['seq'] for _, row in sp.iterrows() if row['file'] in common_files}
    files = sorted(common_files)
    if len(files) < 50:
        print(f'{CTX[c]:12s} {len(files):>7d}  -- skip, too few vocs')
        continue

    bits_g_splits, bits_p_splits = [], []
    Vg_split, Vp_split = [], []
    for split in range(N_SPLITS):
        rng = np.random.default_rng(split)
        files_perm = list(files); rng.shuffle(files_perm)
        n_train = int(len(files_perm) * 0.8)
        train_files = files_perm[:n_train]
        test_files = files_perm[n_train:]

        # Build train/test syllable sequences for both vocabularies
        train_g = [sg_dict[f] for f in train_files]
        test_g  = [sg_dict[f] for f in test_files]
        train_p = [sp_dict[f] for f in train_files]
        test_p  = [sp_dict[f] for f in test_files]

        # Vocab on train only (no test leakage)
        vocab_g_train = set(s for seq in train_g for s in seq)
        vocab_p_train = set(s for seq in train_p for s in seq)
        if not vocab_g_train or not vocab_p_train: continue

        m_g = fit_bigram(train_g, vocab_g_train)
        m_p = fit_bigram(train_p, vocab_p_train)

        bits_total_g, n_tok_g, bpt_g = eval_bigram(m_g, test_g)
        bits_total_p, n_tok_p, bpt_p = eval_bigram(m_p, test_p)
        bits_g_splits.append(bpt_g); bits_p_splits.append(bpt_p)
        Vg_split.append(m_g['V']); Vp_split.append(m_p['V'])

    if not bits_g_splits: continue
    bg, sg_std = np.mean(bits_g_splits), np.std(bits_g_splits)
    bp, sp_std = np.mean(bits_p_splits), np.std(bits_p_splits)
    Vg_mean = np.mean(Vg_split); Vp_mean = np.mean(Vp_split)
    delta = bp - bg
    better = 'PC' if bp < bg else 'GLOBAL'
    print(f'{CTX[c]:12s} {len(files):>7d} {Vg_mean:>6.1f} {Vp_mean:>7.1f} {bg:.3f}±{sg_std:.3f}      {bp:.3f}±{sp_std:.3f}  {delta:+.3f} {better:>8s}')

    results.append({
        'context': CTX[c], 'n_vocs': len(files),
        'V_global': Vg_mean, 'V_pc': Vp_mean,
        'bits_token_global': bg, 'bits_token_global_std': sg_std,
        'bits_token_pc': bp, 'bits_token_pc_std': sp_std,
        'delta': delta,
        # cardinality-fair: compression ratio
        'compress_global': bg / np.log2(Vg_mean),
        'compress_pc': bp / np.log2(Vp_mean),
    })

df = pd.DataFrame(results)
print('\n══════ COMPRESSION RATIO (cardinality-fair: bits/token / log2(V)) ══════')
print(f'{"context":12s} {"compr_global":>14s} {"compr_pc":>10s} {"Δ":>9s}')
for _, r in df.iterrows():
    delta = r['compress_pc'] - r['compress_global']
    print(f'{r["context"]:12s} {r["compress_global"]:>14.3f} {r["compress_pc"]:>10.3f} {delta:>+9.3f}')

print(f'\n=== AGGREGATE ===')
print(f'Mean bits/token:    GLOBAL {df.bits_token_global.mean():.3f} ± {df.bits_token_global.std():.3f}')
print(f'                    PC     {df.bits_token_pc.mean():.3f} ± {df.bits_token_pc.std():.3f}')
print(f'                    delta  {df.delta.mean():+.3f}')
print(f'Mean compression:   GLOBAL {df.compress_global.mean():.3f}')
print(f'                    PC     {df.compress_pc.mean():.3f}')
print(f'\nWins by bits/token:  PC wins {(df.delta < 0).sum()} / {len(df)} contexts')
print(f'Wins by compression: PC wins {(df.compress_pc < df.compress_global).sum()} / {len(df)} contexts')
df.to_csv('docs/thesis/figures/within_context_seq_prediction.csv', index=False)
