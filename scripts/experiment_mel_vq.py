"""
VQ on SAME mel features as Assom baseline.

This is the apples-to-apples comparison: HDBSCAN-on-UMAP vs VQ-on-UMAP
vs VQ-on-mel, using identical segments and features. Answers: "is VQ
a better tokenization method than HDBSCAN for THIS data?"

Previous BEATs-VQ experiment showed VQ on BEATs ≠ win. But that could
be BEATs feature issue, not VQ issue. Here we isolate VQ vs HDBSCAN.

Also add: BPE on VQ tokens (truly adaptive layer).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from src.adaptive_tokenizer import (
    TokenizerState, Token, BPEMerger, SequenceMergerConfig,
)

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CKPT / 'ablation_state.joblib')
mel_flat = st['tf_specs'].reshape(-1, 192).astype(np.float32)
emb = st['embedding']            # UMAP 2D
umap_16 = np.load(CKPT / 'umap_16d.npy')   # UMAP 16D from dim sweep
seg_df = st['seg_df']
ctx = seg_df['context'].to_numpy()
emitters = seg_df['emitter'].to_numpy()
proxy = seg_df['proxy_label'].to_numpy() if 'proxy_label' in seg_df.columns else None
hdbnca = st['hdb_nca_labels']

# Mel-PCA50 for cleaner silhouette / clustering
mel_pca = PCA(n_components=50, random_state=0).fit_transform(mel_flat)

# Per-file sequences
sequences_per_file, contexts_per_seq = [], []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if len(seg_ids) < 2: continue
    sequences_per_file.append(seg_ids)
    contexts_per_seq.append(int(np.bincount(g['context'].to_numpy()).argmax()))
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]


def ctx_purity(labels, ctx):
    p, t = 0, 0
    for tid in sorted(set(labels)):
        if tid < 0: continue
        m = np.where(labels == tid)[0]
        if not len(m): continue
        if max(Counter(ctx[m]).values()) / len(m) >= 0.5: p += len(m)
        t += len(m)
    return p / t if t else 0

def ami(a, b):
    mask = (a >= 0) & (b >= 0)
    if mask.sum() < 2 or len(set(a[mask])) < 2 or len(set(b[mask])) < 2: return np.nan
    return float(adjusted_mutual_info_score(a[mask], b[mask]))

def per_emitter_ami(labels, proxy, em):
    if proxy is None: return np.nan
    vals = []
    for e in set(em):
        mask = (em == e) & (proxy >= 0) & (labels >= 0)
        if mask.sum() < 20: continue
        vals.append(adjusted_mutual_info_score(labels[mask], proxy[mask]))
    return float(np.mean(vals)) if vals else np.nan

def silh(X, labels, n=6000):
    mask = labels >= 0
    n = min(n, int(mask.sum()))
    if n < 100: return np.nan
    lbls = labels[mask][:n]
    if len(set(lbls)) < 2: return np.nan
    return float(silhouette_score(X[mask][:n], lbls, random_state=0))

def levenshtein(a, b):
    if len(a) < len(b): a, b = b, a
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0]*len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(cur[j-1]+1, prev[j]+1, prev[j-1]+(ca!=cb))
        prev = cur
    return prev[-1]

def knn_lev_f1(sequences, contexts, subsample=1500, permute=False, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(contexts)
    if len(sequences) > subsample:
        idx = rng.choice(len(sequences), size=subsample, replace=False)
        sequences = [sequences[i] for i in idx]
        y = y[idx]
    if permute:
        sequences = [list(rng.permutation(s)) for s in sequences]
    n = len(sequences)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = levenshtein(sequences[i], sequences[j])
            D[i, j] = d; D[j, i] = d
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    f1s = []
    for tr, te in cv.split(range(n), y):
        preds = []
        for i in te:
            nn = np.argsort(D[i, tr])[:5]
            preds.append(Counter(y[tr][nn]).most_common(1)[0][0])
        f1s.append(f1_score(y[te], preds, average='weighted'))
    return float(np.mean(f1s)), float(np.std(f1s))

def build_seqs(labels):
    return [[int(labels[i]) for i in seg_ids if labels[i] >= 0]
             for seg_ids in sequences_per_file]

def eval_method(labels, space_X, name, run_hp1=True):
    d = {
        'method': name, 'vocab': len(set(labels)) - (1 if -1 in labels else 0),
        'silh_native': round(silh(space_X, labels), 3),
        'ctx_purity': round(ctx_purity(labels, ctx), 3),
        'AMI_ctx': round(ami(labels, ctx), 3),
        'AMI_proxy': round(per_emitter_ami(labels, proxy, emitters), 3),
    }
    if run_hp1:
        seqs = build_seqs(labels)
        pairs = [(s, c) for s, c in zip(seqs, contexts_per_seq) if len(s) >= 2 and c in HP1_CTX]
        seqs_hp1, ctx_hp1 = [s for s,c in pairs], [c for s,c in pairs]
        f1_o, _ = knn_lev_f1(seqs_hp1, ctx_hp1, permute=False)
        f1_p, _ = knn_lev_f1(seqs_hp1, ctx_hp1, permute=True)
        d['F1_lev'] = round(f1_o, 3); d['Delta_HP1'] = round(f1_o - f1_p, 3)
    return d


rows = []
# Reference baseline
rows.append(eval_method(hdbnca, mel_pca, 'Mel HDBSCAN baseline (Assom)'))

# VQ on Mel-PCA50
print('Running VQ on Mel-PCA50 ...')
for K in [16, 32, 64, 128]:
    t0 = time.time()
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(mel_pca)
    r = eval_method(km.labels_.astype(int), mel_pca, f'VQ on Mel-PCA50 K={K}')
    rows.append(r)
    print(f'  K={K} took {time.time()-t0:.0f}s AMI={r["AMI_ctx"]} F1_lev={r.get("F1_lev","-")}')

# VQ on UMAP-2D
print('Running VQ on UMAP-2D ...')
for K in [16, 32, 64]:
    t0 = time.time()
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(emb)
    r = eval_method(km.labels_.astype(int), mel_pca, f'VQ on UMAP-2D K={K}')
    rows.append(r)
    print(f'  K={K} AMI={r["AMI_ctx"]}')

# VQ on UMAP-16D
print('Running VQ on UMAP-16D ...')
for K in [16, 32, 64]:
    t0 = time.time()
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(umap_16)
    r = eval_method(km.labels_.astype(int), mel_pca, f'VQ on UMAP-16D K={K}')
    rows.append(r)
    print(f'  K={K} AMI={r["AMI_ctx"]}')

# Now: BPE on top of BEST VQ variant (adaptive layer 2)
# Pick best by AMI_ctx
print('\nBPE on best VQ tokens ...')
best_idx = int(np.argmax([r['AMI_ctx'] if r['AMI_ctx'] == r['AMI_ctx'] else -1 for r in rows[1:]])) + 1
best = rows[best_idx]
print(f'Best VQ: {best["method"]} AMI={best["AMI_ctx"]}')

# Regenerate its labels
parts = best['method'].split('K=')
K = int(parts[1])
space = 'mel_pca' if 'Mel-PCA50' in best['method'] else ('emb' if 'UMAP-2D' in best['method'] else 'umap_16')
X_space = {'mel_pca': mel_pca, 'emb': emb, 'umap_16': umap_16}[space]
km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(X_space)
vq_labels = km.labels_.astype(int)

# Build TokenizerState + BPE
tokens = {int(c): Token(id=int(c),
                           centroid=X_space[vq_labels == c].mean(axis=0),
                           member_ids=np.where(vq_labels == c)[0])
          for c in set(vq_labels)}
seqs_for_bpe = build_seqs(vq_labels)
state = TokenizerState(tokens=tokens, labels=vq_labels, sequences=seqs_for_bpe)

for n_merges in [10, 30, 60]:
    s = TokenizerState(tokens=dict(state.tokens), labels=state.labels.copy(),
                        sequences=[list(x) for x in state.sequences])
    BPEMerger(SequenceMergerConfig(max_merges=n_merges, min_bigram_count=30,
                                      min_sequences_containing=10, show_progress=False)).fit(s)
    # eval from sequences directly (labels unchanged — BPE is seq-level)
    seqs_hp1 = [seq for seq, c in zip(s.sequences, contexts_per_seq) if len(seq) >= 2 and c in HP1_CTX]
    ctx_hp1 = [c for seq, c in zip(s.sequences, contexts_per_seq) if len(seq) >= 2 and c in HP1_CTX]
    f1_o, _ = knn_lev_f1(seqs_hp1, ctx_hp1, permute=False)
    f1_p, _ = knn_lev_f1(seqs_hp1, ctx_hp1, permute=True)
    rows.append({
        'method': f'{best["method"]} + BPE {n_merges} merges',
        'vocab': s.vocab_size, 'silh_native': best['silh_native'],
        'ctx_purity': best['ctx_purity'], 'AMI_ctx': best['AMI_ctx'],
        'AMI_proxy': best['AMI_proxy'],
        'F1_lev': round(f1_o, 3), 'Delta_HP1': round(f1_o - f1_p, 3),
    })
    print(f'  +BPE {n_merges}: vocab={s.vocab_size}, F1_lev={f1_o:.3f}, ΔHP1={f1_o-f1_p:+.3f}')

df = pd.DataFrame(rows)
print('\n\n=== MEL VQ + BPE SUMMARY ===')
print(df.to_string(index=False))
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/mel_vq_sweep.csv', index=False)
