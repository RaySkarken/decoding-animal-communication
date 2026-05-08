"""Attentive probing на замороженном BEATs (NLM, pitch_shifted) — Schwinger 2025 style.

Корпус: cache 49604 BEATs эмбеддингов (768D, pitch_shift) → 8750 вокализаций.
HP1_CTX = [2,3,4,5,6,7,9,10] (8 рабочих контекстов), эмиттер известен.

Сравниваем 5 вариантов pooling × probe head:
  v1: mean pool сегментов + linear classifier (~ Schwinger linear probing)
  v2: mean+std pool + linear classifier
  v3: mean+std pool + 2-layer MLP (текущий FC baseline)
  v4: attentive pool (1 CLS query + 8-head MHA) + linear classifier
  v5: attentive pool + 2-layer MLP

Для Schwinger 2025: attentive probing > linear probing на BEATs/AudioSet-моделях.
Гипотеза: v4/v5 > v1/v2/v3 на macro F1.

Cross-bat protocol: 30 train / 11 test эмиттеров, 5 сидов.
Метрика: macro F1 (ведущая), weighted F1 (в CSV).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import os, time
import numpy as np, pandas as pd
import joblib
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {DEVICE}', flush=True)

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
N_CTX = len(HP1_CTX)
ctx2y = {c: i for i, c in enumerate(HP1_CTX)}

MAX_K = 32  # truncate vocs with more segments (q95=17, max=177)
N_EPOCHS = 60
BATCH = 64
LR = 1e-3
WD = 1e-4

print('Loading BEATs cache...', flush=True)
st = joblib.load(CACHE / 'beats_full_experiment.joblib')
seg_meta = st['seg_meta'].reset_index(drop=True)
X = st['X_beats'].astype(np.float32)
print(f'  segments: {X.shape}, vocs: {seg_meta["file_name"].nunique()}', flush=True)

mask = seg_meta['context'].isin(HP1_CTX) & (seg_meta['emitter'] != 0)
seg_meta = seg_meta[mask].reset_index(drop=True)
X = X[mask.values]
print(f'  after HP1+identified: segments={len(X)}, vocs={seg_meta["file_name"].nunique()}, emitters={seg_meta["emitter"].nunique()}', flush=True)


def build_voc_data():
    """Group segments by voc → list of (K, 768) tensors + labels."""
    voc_segs = []
    voc_y = []
    voc_em = []
    for fname, g in seg_meta.groupby('file_name', sort=False):
        idxs = g.index.values
        ctx = int(g['context'].mode().iloc[0])
        em = int(g['emitter'].mode().iloc[0])
        if len(idxs) > MAX_K:
            idxs = np.random.RandomState(hash(fname) & 0xffffffff).choice(idxs, MAX_K, replace=False)
        voc_segs.append(X[idxs])
        voc_y.append(ctx2y[ctx])
        voc_em.append(em)
    return voc_segs, np.array(voc_y, dtype=np.int64), np.array(voc_em, dtype=np.int64)


print('Building voc-level data...', flush=True)
voc_segs, voc_y, voc_em = build_voc_data()
print(f'  vocs: {len(voc_segs)}', flush=True)


class VocDataset(Dataset):
    def __init__(self, indices, segs, y):
        self.idx = indices
        self.segs = segs
        self.y = y
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        ii = self.idx[i]
        return self.segs[ii], self.y[ii]


def collate(batch):
    segs = [b[0] for b in batch]
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    Ks = [s.shape[0] for s in segs]
    Kmax = max(Ks)
    B = len(segs)
    x = torch.zeros(B, Kmax, 768, dtype=torch.float32)
    m = torch.zeros(B, Kmax, dtype=torch.bool)
    for i, s in enumerate(segs):
        k = s.shape[0]
        x[i, :k] = torch.from_numpy(s)
        m[i, :k] = True
    return x, m, ys


# ─────────────── pooling heads ───────────────
class MeanPool(nn.Module):
    def forward(self, x, mask):  # x: (B, K, D), mask: (B, K)
        m = mask.unsqueeze(-1).float()
        return (x * m).sum(1) / m.sum(1).clamp(min=1)


class MeanStdPool(nn.Module):
    def forward(self, x, mask):
        m = mask.unsqueeze(-1).float()
        cnt = m.sum(1).clamp(min=1)
        mean = (x * m).sum(1) / cnt
        var = ((x - mean.unsqueeze(1)) ** 2 * m).sum(1) / cnt
        std = (var + 1e-6).sqrt()
        return torch.cat([mean, std], dim=-1)  # (B, 2D)


class AttentivePool(nn.Module):
    def __init__(self, dim=768, n_heads=8):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.mha = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, mask):  # mask True=valid
        B = x.shape[0]
        cls = self.cls.expand(B, -1, -1)
        kpm = ~mask
        out, _ = self.mha(cls, x, x, key_padding_mask=kpm)
        return self.norm(out.squeeze(1))


class Probe(nn.Module):
    def __init__(self, pool: nn.Module, in_dim: int, hidden: int | None, n_cls: int = N_CTX):
        super().__init__()
        self.pool = pool
        if hidden is None:
            self.head = nn.Linear(in_dim, n_cls)
        else:
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden, n_cls),
            )
    def forward(self, x, mask):
        z = self.pool(x, mask)
        return self.head(z)


def make_probe(variant: str) -> Probe:
    if variant == 'mean_lin':
        return Probe(MeanPool(), 768, None)
    if variant == 'meanstd_lin':
        return Probe(MeanStdPool(), 1536, None)
    if variant == 'meanstd_mlp':
        return Probe(MeanStdPool(), 1536, 256)
    if variant == 'attn_lin':
        return Probe(AttentivePool(768, 8), 768, None)
    if variant == 'attn_mlp':
        return Probe(AttentivePool(768, 8), 768, 256)
    raise ValueError(variant)


VARIANTS = ['mean_lin', 'meanstd_lin', 'meanstd_mlp', 'attn_lin', 'attn_mlp']


# ─────────────── train/eval ───────────────
def train_eval(variant: str, tr_idx: np.ndarray, te_idx: np.ndarray, seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = VocDataset(tr_idx, voc_segs, voc_y)
    test_ds = VocDataset(te_idx, voc_segs, voc_y)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, collate_fn=collate)

    cls_counts = np.bincount(voc_y[tr_idx], minlength=N_CTX).astype(np.float32)
    cw = (cls_counts.sum() / (N_CTX * np.clip(cls_counts, 1, None)))
    cw_t = torch.from_numpy(cw).to(DEVICE)

    model = make_probe(variant).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    crit = nn.CrossEntropyLoss(weight=cw_t)

    for ep in range(N_EPOCHS):
        model.train()
        ep_loss = 0.0; n = 0
        for x, m, y in train_dl:
            x = x.to(DEVICE); m = m.to(DEVICE); y = y.to(DEVICE)
            logits = model(x, m)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * y.size(0); n += y.size(0)
        sched.step()

    model.eval()
    preds = []; ys = []
    with torch.no_grad():
        for x, m, y in test_dl:
            x = x.to(DEVICE); m = m.to(DEVICE)
            logits = model(x, m)
            preds.append(logits.argmax(-1).cpu().numpy())
            ys.append(y.numpy())
    pred = np.concatenate(preds); yt = np.concatenate(ys)
    f1m = f1_score(yt, pred, average='macro', zero_division=0)
    f1w = f1_score(yt, pred, average='weighted', zero_division=0)
    return {'variant': variant, 'seed': seed, 'f1_macro': f1m, 'f1_weighted': f1w, 'n_test': len(yt)}


def split_emitters(seed):
    rng = np.random.default_rng(seed)
    em = np.array(sorted(set(voc_em.tolist())))
    rng.shuffle(em)
    test_em = set(em[:11].tolist())
    return test_em


# ─────────────── run ───────────────
results = []
for seed in range(5):
    test_em = split_emitters(seed)
    tr_idx = np.array([i for i, em in enumerate(voc_em) if em not in test_em])
    te_idx = np.array([i for i, em in enumerate(voc_em) if em in test_em])
    print(f'\n--- seed {seed}: train={len(tr_idx)}, test={len(te_idx)} ---', flush=True)
    for v in VARIANTS:
        t0 = time.time()
        r = train_eval(v, tr_idx, te_idx, seed)
        elapsed = time.time() - t0
        print(f'  >>> {v:14s}: f1m={r["f1_macro"]:.3f}, f1w={r["f1_weighted"]:.3f} ({elapsed:.0f}s)', flush=True)
        results.append(r)

dfr = pd.DataFrame(results)
out_csv = Path('docs/thesis/figures/beats_attentive_probe_results.csv')
dfr.to_csv(out_csv, index=False)

print('\n=== Summary (mean ± std, sorted by macro F1) ===', flush=True)
agg = dfr.groupby('variant').agg(
    f1m_mean=('f1_macro', 'mean'), f1m_std=('f1_macro', 'std'),
    f1w_mean=('f1_weighted', 'mean'), f1w_std=('f1_weighted', 'std'),
    n=('seed', 'count'),
).sort_values('f1m_mean', ascending=False)
print(agg.to_string(), flush=True)
agg.to_csv(Path('docs/thesis/figures/beats_attentive_probe_summary.csv'))
print(f'\nSaved: {out_csv}', flush=True)
