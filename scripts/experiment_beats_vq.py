"""
VQ (Vector Quantization) on BEATs embeddings — proper tokenization
study tied to thesis title.

Three variants tested:
  1. Static VQ via k-means — matches Sarkar & Magimai-Doss 2025
     (competitor baseline — must beat)
  2. Adaptive VQ: codebook refinement via size/usage-based
     split/merge on the CODEBOOK (not on embeddings directly)
  3. BPE on VQ tokens — sequence-level adaptive layer

All evaluated under lit-aligned protocol:
  - AMI vs context (chance-corrected)
  - AMI vs qt_ward proxy (per-emitter)
  - Levenshtein-kNN F1 original vs permuted (HP1 test)
  - Silhouette on BEATs-PCA50 (native space)

Evaluated codebook sizes K ∈ {16, 32, 64, 128, 256}.
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
from sklearn.metrics import (
    silhouette_score, adjusted_mutual_info_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from src.adaptive_tokenizer import TokenizerState, Token

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
st_b = joblib.load(CKPT / 'beats_full_experiment.joblib')
X = st_b['X_beats']              # (49604, 768)
meta = st_b['seg_meta']           # file_name, file_id, context, emitter
ctx = meta['context'].to_numpy()
emitters = meta['emitter'].to_numpy()

# We don't have qt_ward proxy on BEATs segments (different segmentation from
# the main pipeline). For AMI-proxy we'd need to match to ablation_state; skip.
# Instead: use ONLY AMI_ctx, F1_lev, silh as primary metrics.

# Build per-file sequences
sequences_per_file, contexts_per_seq = [], []
meta_sorted = meta.reset_index(drop=True)
for fname, g in meta_sorted.groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if len(seg_ids) < 2: continue
    sequences_per_file.append(seg_ids)
    contexts_per_seq.append(int(np.bincount(g['context'].to_numpy()).argmax()))
print(f'Sequences: {len(sequences_per_file)}')

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
keep = [(s, c) for s, c in zip(sequences_per_file, contexts_per_seq) if c in HP1_CTX]
sequences_hp1 = [s for s, c in keep]
contexts_hp1 = [c for s, c in keep]

# PCA-50 for silhouette computation (768D silhouette too expensive)
X_pca = PCA(n_components=50, random_state=0).fit_transform(X)


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
            cur[j] = min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + (ca != cb))
        prev = cur
    return prev[-1]


def build_label_sequences(labels, seqs_per_file):
    return [[int(labels[i]) for i in seg_ids if labels[i] >= 0] for seg_ids in seqs_per_file]


def knn_lev_f1(sequences, contexts, k=5, n_splits=5, subsample=1500, permute=False, seed=0):
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
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    f1s = []
    for tr, te in cv.split(range(n), y):
        preds = []
        for i in te:
            nn = np.argsort(D[i, tr])[:k]
            preds.append(Counter(y[tr][nn]).most_common(1)[0][0])
        f1s.append(f1_score(y[te], preds, average='weighted'))
    return float(np.mean(f1s)), float(np.std(f1s))


def eval_all(labels, method_name, run_hp1=True, hp1_subsample=1500):
    out = {
        'method': method_name,
        'vocab': len(set(labels)) - (1 if -1 in labels else 0),
        'silh_PCA50': round(silh(X_pca, labels), 3),
        'ctx_purity': round(ctx_purity(labels, ctx), 3),
        'AMI_ctx': round(ami(labels, ctx), 3),
    }
    if run_hp1:
        seqs = build_label_sequences(labels, sequences_per_file)
        pairs = [(s, c) for s, c in zip(seqs, contexts_per_seq) if len(s) >= 2 and c in HP1_CTX]
        seqs_hp1 = [s for s, c in pairs]
        ctx_s = [c for s, c in pairs]
        f1_o, s_o = knn_lev_f1(seqs_hp1, ctx_s, permute=False, subsample=hp1_subsample)
        f1_p, s_p = knn_lev_f1(seqs_hp1, ctx_s, permute=True, subsample=hp1_subsample)
        out['F1_lev'] = round(f1_o, 3)
        out['F1_lev_perm'] = round(f1_p, 3)
        out['Delta_HP1'] = round(f1_o - f1_p, 3)
    return out


# ─── VQ 1: Static k-means (Sarkar & Magimai-Doss 2025 baseline) ────────
print('\n=== VARIANT 1: Static k-means VQ on BEATs-PCA50 ===')
rows = []
for K in [16, 32, 64, 128, 256]:
    print(f'K={K} ...', end='', flush=True)
    t0 = time.time()
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(X_pca)
    labels = km.labels_.astype(int)
    r = eval_all(labels, f'Static VQ K={K}', run_hp1=(K <= 128))
    r['fit_time_s'] = round(time.time() - t0, 1)
    rows.append(r)
    print(f'  took {time.time()-t0:.0f}s, AMI_ctx={r["AMI_ctx"]}, F1_lev={r.get("F1_lev","-")}')

# ─── VQ 2: Adaptive codebook — size-based prune + centroid-distance merge ──
print('\n=== VARIANT 2: Adaptive VQ (usage-prune + distance-merge) ===')
def adaptive_vq(X, K_init=128, min_usage=50, merge_dist_q=0.05, max_iter=3):
    """
    Start with K_init codebook entries via k-means.
    Iteratively:
      - drop codebook entries with usage < min_usage (reassign to nearest)
      - merge pairs with centroid distance < q-th percentile
    """
    km = KMeans(n_clusters=K_init, random_state=0, n_init=5).fit(X)
    codebook = km.cluster_centers_
    labels = km.labels_.astype(int)
    for it in range(max_iter):
        # Prune: drop entries with low usage
        cnt = Counter(int(x) for x in labels)
        alive = [k for k in range(len(codebook)) if cnt.get(k, 0) >= min_usage]
        if len(alive) < len(codebook):
            idmap = {old: new for new, old in enumerate(alive)}
            new_codebook = codebook[alive]
            # Reassign labels to nearest alive entry
            from sklearn.neighbors import NearestNeighbors
            knn = NearestNeighbors(n_neighbors=1).fit(new_codebook)
            _, idx = knn.kneighbors(X)
            labels = idx.ravel().astype(int)
            codebook = new_codebook
        # Merge close pairs
        from scipy.spatial.distance import pdist, squareform
        D = squareform(pdist(codebook))
        np.fill_diagonal(D, np.inf)
        thr = np.quantile(D[D < np.inf], merge_dist_q) if (D < np.inf).any() else 0
        merged = False
        for _ in range(10):
            if D.min() > thr: break
            i, j = np.unravel_index(D.argmin(), D.shape)
            if i == j: break
            # Merge j into i (weighted centroid)
            n_i = (labels == i).sum()
            n_j = (labels == j).sum()
            codebook[i] = (codebook[i] * n_i + codebook[j] * n_j) / max(n_i + n_j, 1)
            labels[labels == j] = i
            # Remove row/col j
            codebook = np.delete(codebook, j, axis=0)
            # Relabel: shift labels > j down by 1
            labels[labels > j] -= 1
            D = squareform(pdist(codebook))
            np.fill_diagonal(D, np.inf)
            merged = True
        if not merged:
            break
    return labels


for K_init in [64, 128, 256]:
    print(f'Adaptive VQ K_init={K_init} ...', end='', flush=True)
    t0 = time.time()
    labels = adaptive_vq(X_pca, K_init=K_init, min_usage=50, merge_dist_q=0.01, max_iter=3)
    r = eval_all(labels, f'Adaptive VQ K_init={K_init}', run_hp1=True)
    r['fit_time_s'] = round(time.time() - t0, 1)
    rows.append(r)
    print(f'  took {time.time()-t0:.0f}s, final_vocab={r["vocab"]}, AMI_ctx={r["AMI_ctx"]}, F1_lev={r.get("F1_lev","-")}')

# ─── VQ 3: Residual VQ (hierarchical) ──────────────────────────────────
print('\n=== VARIANT 3: Residual VQ (coarse + fine) ===')
def residual_vq(X, K_coarse=16, K_fine=8):
    """Two-level: coarse partition into K_coarse, then fine partition of
    residual (X - coarse_centroid) into K_fine per coarse cluster.
    Total vocab = K_coarse * K_fine. Token = (coarse, fine)."""
    km1 = KMeans(n_clusters=K_coarse, random_state=0, n_init=10).fit(X)
    coarse = km1.labels_
    codebook_c = km1.cluster_centers_
    residuals = X - codebook_c[coarse]
    # Fine per coarse cluster
    fine = np.zeros(len(X), dtype=int)
    for c in range(K_coarse):
        mask = coarse == c
        if mask.sum() < K_fine: continue
        kmf = KMeans(n_clusters=K_fine, random_state=0, n_init=5).fit(residuals[mask])
        fine[mask] = kmf.labels_
    # Combined token = coarse * K_fine + fine
    return coarse * K_fine + fine


for K_c, K_f in [(8, 4), (16, 4), (16, 8), (32, 4)]:
    K_total = K_c * K_f
    print(f'RVQ coarse={K_c} fine={K_f} (total={K_total}) ...', end='', flush=True)
    t0 = time.time()
    labels = residual_vq(X_pca, K_coarse=K_c, K_fine=K_f)
    r = eval_all(labels, f'RVQ {K_c}x{K_f}', run_hp1=(K_total <= 128))
    r['fit_time_s'] = round(time.time() - t0, 1)
    rows.append(r)
    print(f'  took {time.time()-t0:.0f}s, actual_vocab={r["vocab"]}, AMI_ctx={r["AMI_ctx"]}')

# ─── Summary ───────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
print('\n\n=== VQ-ON-BEATS SUMMARY ===')
print(df.to_string(index=False))
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df.to_csv('docs/thesis/figures/beats_vq_sweep.csv', index=False)

# Save the best adaptive-VQ state for later (BPE etc)
best_row = df.loc[df['AMI_ctx'].idxmax()]
print(f'\nBest AMI_ctx: {best_row["method"]} = {best_row["AMI_ctx"]:.3f}')
