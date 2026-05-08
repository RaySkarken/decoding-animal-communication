"""BEATs(frozen) + FC fine-tune → латенты → кластеризация → RF на токенах.

Pipeline на 5 cross-bat сидов:
  1. BEATs frozen эмбеддинги (уже в кэше, 49604 × 768).
  2. FC head 768 → 256 → 256 → 8 контекстов; обучается на train эмиттерах.
  3. Латенты = вход последнего слоя (256D), для всех сегментов.
  4. Кластеризация латентов на train эмиттерах разными способами:
       kmeans_15, kmeans_30 (на 256D напрямую),
       umap8d_kmeans_15, umap8d_kmeans_30 (UMAP-8D + k-means),
       umap2d_hdbscan (UMAP-2D + HDBSCAN с дефолтными параметрами).
  5. Test эмиттеры получают токены через nearest centroid (для k-means)
     или KNN-1 предсказание (для HDBSCAN-разметки).
  6. RF на агрегатах последовательности токенов (нормированные частоты + Zhang-18).
     Train на train_em вокализациях, eval на test_em.

Контрольный baseline: end-to-end FC без дискретизации, majority vote per voc.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, time
from pathlib import Path
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans, HDBSCAN as sk_HDBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import umap

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}')
CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX2IDX = {c: i for i, c in enumerate(HP1_CTX)}
N_SEEDS = 5

print('Loading BEATs cache...')
st = joblib.load(CACHE / 'beats_full_experiment.joblib')
seg_meta = st['seg_meta'].reset_index(drop=True)
X_beats = st['X_beats']
print(f'  X_beats: {X_beats.shape}, dtype={X_beats.dtype}')

mask_hp1 = seg_meta['context'].isin(HP1_CTX) & (seg_meta['emitter'] != 0)
seg_meta = seg_meta[mask_hp1].reset_index(drop=True)
X_beats = X_beats[mask_hp1.values]
print(f'  after HP1 filter: {len(seg_meta)} segments, {seg_meta["file_name"].nunique()} vocs, '
      f'{seg_meta["emitter"].nunique()} emitters')

ctx_arr = seg_meta['context'].map(CTX2IDX).to_numpy()  # 0..7
em_arr = seg_meta['emitter'].to_numpy()
file_arr = seg_meta['file_name'].to_numpy()


# --- FC head ---
class FCHead(nn.Module):
    def __init__(self, in_dim=768, hidden=256, n_cls=8, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.cls = nn.Linear(hidden, n_cls)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)

    def forward(self, x, return_latent=False):
        h = self.drop(self.act(self.bn1(self.fc1(x))))
        z = self.act(self.bn2(self.fc2(h)))  # latent (no dropout here)
        out = self.cls(self.drop(z))
        if return_latent:
            return out, z
        return out


def split_emitters(seed, all_em):
    rng = np.random.default_rng(seed)
    em = np.array(sorted(list(all_em))); rng.shuffle(em)
    return set(em[:11].tolist())


def voc_aggregate(tokens, files, ctx, vocab_size):
    """Per-vocalization features: token freq + length + entropy + max + n_unique.
       + per-context cross-vocalization stats. Returns X, y_ctx, voc_files.
    """
    df = pd.DataFrame({'idx': np.arange(len(tokens)), 'file': files, 'tok': tokens, 'ctx': ctx})
    df = df[df['tok'] >= 0]
    X, y, vocs = [], [], []
    for fname, g in df.groupby('file'):
        toks = g['tok'].to_numpy()
        if len(toks) == 0: continue
        freq = np.bincount(toks, minlength=vocab_size) / len(toks)
        ent = -np.sum(freq * np.log(freq + 1e-12))
        feat = np.concatenate([freq, [len(toks), ent, freq.max(),
                                       float(np.count_nonzero(freq))]])
        X.append(feat.astype(np.float32))
        y.append(int(g['ctx'].mode()[0]))
        vocs.append(fname)
    return np.array(X), np.array(y), np.array(vocs)


def majority_per_voc(preds, files):
    """Per-voc majority vote (для end-to-end baseline)."""
    df = pd.DataFrame({'pred': preds, 'file': files})
    out_y = []
    for fname, g in df.groupby('file', sort=False):
        out_y.append(int(g['pred'].mode()[0]))
    return np.array(out_y)


def voc_true_labels(ctx, files):
    df = pd.DataFrame({'ctx': ctx, 'file': files})
    return np.array([int(g['ctx'].mode()[0]) for _, g in df.groupby('file', sort=False)])


# --- Main loop over seeds ---
results = []
for seed in range(N_SEEDS):
    torch.manual_seed(seed); np.random.seed(seed)
    test_em = split_emitters(seed, set(em_arr.tolist()))
    train_idx = np.where(~np.isin(em_arr, list(test_em)))[0]
    test_idx = np.where(np.isin(em_arr, list(test_em)))[0]
    print(f'\n--- seed {seed}: train={len(train_idx)}, test={len(test_idx)} ---', flush=True)

    # Step 1: Fine-tune FC head
    model = FCHead().to(DEVICE)
    Xtr = torch.tensor(X_beats[train_idx], dtype=torch.float32).to(DEVICE)
    ytr = torch.tensor(ctx_arr[train_idx], dtype=torch.long).to(DEVICE)

    counts = np.bincount(ctx_arr[train_idx], minlength=len(HP1_CTX))
    cw = torch.tensor((counts.sum() / (len(HP1_CTX) * counts.clip(min=1))).astype(np.float32)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bs = 256
    n_ep = 80
    t0 = time.time()
    for ep in range(n_ep):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot, n = 0., 0
        for i in range(0, len(Xtr), bs):
            ii = perm[i:i+bs]
            x = Xtr[ii]; y = ytr[ii]
            out = model(x)
            loss = F.cross_entropy(out, y, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        if (ep + 1) % 20 == 0:
            print(f'  fine-tune ep{ep+1}: loss={tot/n:.4f}', flush=True)
    print(f'  fine-tune done in {time.time()-t0:.1f}s', flush=True)

    # Step 2: Extract latents for ALL segments
    model.eval()
    Z = np.zeros((len(X_beats), 256), dtype=np.float32)
    preds_endtoend = np.zeros(len(X_beats), dtype=np.int32)
    Xall = torch.tensor(X_beats, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X_beats), bs):
            x = Xall[i:i+bs].to(DEVICE)
            out, z = model(x, return_latent=True)
            Z[i:i+bs] = z.cpu().numpy()
            preds_endtoend[i:i+bs] = out.argmax(-1).cpu().numpy()

    # End-to-end baseline: majority vote per voc on test
    test_files_arr = file_arr[test_idx]
    test_preds = preds_endtoend[test_idx]
    pred_voc = majority_per_voc(test_preds, test_files_arr)
    true_voc = voc_true_labels(ctx_arr[test_idx], test_files_arr)
    f1w_e2e = f1_score(true_voc, pred_voc, average='weighted', zero_division=0)
    f1m_e2e = f1_score(true_voc, pred_voc, average='macro', zero_division=0)
    print(f'  >>> end-to-end FC majority vote: f1w={f1w_e2e:.3f}, f1m={f1m_e2e:.3f}',
          flush=True)
    results.append({'seed': seed, 'method': 'BEATs+FC e2e (majority vote)',
                    'f1_w': f1w_e2e, 'f1_m': f1m_e2e})

    # Step 3: Cluster latents on TRAIN
    Z_train = Z[train_idx]
    print(f'  clustering {len(Z_train)} train latents...', flush=True)

    # UMAP-8D и UMAP-2D от train
    print('  fitting UMAP-8D...', flush=True)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Z_train), min(20000, len(Z_train)), replace=False)
    umap8 = umap.UMAP(n_components=8, n_neighbors=30, min_dist=0.0,
                       random_state=seed, n_jobs=1).fit(Z_train[sub])
    print('  fitting UMAP-2D...', flush=True)
    umap2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                       random_state=seed, n_jobs=1).fit(Z_train[sub])
    Z_train_8d = umap8.transform(Z_train)
    Z_train_2d = umap2.transform(Z_train)
    Z_test_8d = umap8.transform(Z[test_idx])
    Z_test_2d = umap2.transform(Z[test_idx])

    cluster_methods = {}

    # k-means 15 on raw 256D
    km15 = KMeans(n_clusters=15, n_init=10, random_state=seed).fit(Z_train)
    cluster_methods['km15_256d'] = (km15, Z_train, Z[test_idx])

    # k-means 30 on raw 256D
    km30 = KMeans(n_clusters=30, n_init=10, random_state=seed).fit(Z_train)
    cluster_methods['km30_256d'] = (km30, Z_train, Z[test_idx])

    # k-means 15 on UMAP-8D
    km15_u8 = KMeans(n_clusters=15, n_init=10, random_state=seed).fit(Z_train_8d)
    cluster_methods['km15_umap8d'] = (km15_u8, Z_train_8d, Z_test_8d)

    # k-means 30 on UMAP-8D
    km30_u8 = KMeans(n_clusters=30, n_init=10, random_state=seed).fit(Z_train_8d)
    cluster_methods['km30_umap8d'] = (km30_u8, Z_train_8d, Z_test_8d)

    # HDBSCAN on UMAP-2D, with NCA-like reassign
    hdb = sk_HDBSCAN(min_cluster_size=int(0.02 * len(Z_train_2d)), min_samples=20,
                     cluster_selection_epsilon=0.1).fit(Z_train_2d)
    L = hdb.labels_
    nn_mask = L >= 0
    if nn_mask.sum() > 50 and len(set(L[nn_mask].tolist())) > 1:
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(Z_train_2d[nn_mask], L[nn_mask])
        L_train = L.copy()
        if (~nn_mask).sum() > 0:
            L_train[~nn_mask] = knn.predict(Z_train_2d[~nn_mask])
        L_test = knn.predict(Z_test_2d)
    else:
        L_train = np.zeros(len(Z_train_2d), dtype=int); L_test = np.zeros(len(Z_test_2d), dtype=int)
    n_hdb_clusters = len(set(L_train.tolist()))
    print(f'  HDBSCAN-2D: {n_hdb_clusters} clusters', flush=True)
    cluster_methods['hdb_umap2d'] = (None, None, None)  # special handling
    # store labels directly
    direct_labels = {'hdb_umap2d': (L_train, L_test, n_hdb_clusters)}

    # Step 4-5: Build sequences, RF train, eval
    for cm_name, item in cluster_methods.items():
        if cm_name in direct_labels:
            tokens_train_seg, tokens_test_seg, vocab_size = direct_labels[cm_name]
        else:
            cm, Ztr_proj, Zte_proj = item
            tokens_train_seg = cm.predict(Ztr_proj)
            tokens_test_seg = cm.predict(Zte_proj)
            vocab_size = cm.n_clusters

        Xtr_voc, ytr_voc, _ = voc_aggregate(tokens_train_seg, file_arr[train_idx],
                                              ctx_arr[train_idx], vocab_size)
        Xte_voc, yte_voc, _ = voc_aggregate(tokens_test_seg, file_arr[test_idx],
                                              ctx_arr[test_idx], vocab_size)
        if len(Xtr_voc) < 10 or len(Xte_voc) < 10:
            print(f'  {cm_name}: too few vocs (train={len(Xtr_voc)}, test={len(Xte_voc)})')
            continue
        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                     random_state=seed, n_jobs=-1).fit(Xtr_voc, ytr_voc)
        pred = rf.predict(Xte_voc)
        f1w = f1_score(yte_voc, pred, average='weighted', zero_division=0)
        f1m = f1_score(yte_voc, pred, average='macro', zero_division=0)
        print(f'  >>> {cm_name} (|V|={vocab_size}): f1w={f1w:.3f}, f1m={f1m:.3f}',
              flush=True)
        results.append({'seed': seed, 'method': f'BEATs+FC latent → {cm_name}',
                        'vocab': vocab_size, 'f1_w': f1w, 'f1_m': f1m})


df = pd.DataFrame(results)
df.to_csv('docs/thesis/figures/beats_finetune_cluster_results.csv', index=False)
print('\n=== Summary (mean ± std over seeds) ===')
agg = df.groupby('method').agg(f1w_mean=('f1_w', 'mean'), f1w_std=('f1_w', 'std'),
                                f1m_mean=('f1_m', 'mean'), f1m_std=('f1_m', 'std'),
                                n=('seed', 'count')).reset_index()
agg = agg.sort_values('f1w_mean', ascending=False)
print(agg.to_string(index=False))
agg.to_csv('docs/thesis/figures/beats_finetune_cluster_summary.csv', index=False)

print('\nFor reference (на других экспериментах):')
print('  PC-share + BERT с предтренингом: 0.582 ± 0.080')
print('  SensorSCAN + linear probe:        0.538 ± 0.041')
print('  Two FC на raw mel:                0.537 ± 0.034')
print('  RF на raw mel:                    0.529 ± 0.058')
print('  Per-context k-means + Bayes:      0.448 ± 0.078')
print('  BEATs + DP-GMM (без файн-тюна):  0.307')
print('  Assom + RF baseline:              0.346 ± 0.075')
