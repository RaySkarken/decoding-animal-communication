"""Two FC layers поверх frozen эмбеддингов: классификация контекста.

Архитектура: aggregate(features per voc) -> Linear -> ReLU -> Dropout -> Linear -> 8 contexts.
Aggregate = concat(mean, std) по сегментам вокализации.

Сравнения:
  raw mel (192-D)         -> 2FC      vs   raw mel -> RF (§3.5)
  UMAP-8D                 -> 2FC      vs   UMAP-8D  -> RF (§3.5)
  BEATs-768D (если есть)  -> 2FC

Без токенизации; классификатор обучаем в конец-к-концу.
5 сидов, разбиение по особям.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, time
from pathlib import Path
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}')
CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX2IDX = {c: i for i, c in enumerate(HP1_CTX)}

print('Loading state...')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
mel = st['tf_specs'].reshape(len(seg_df), -1).astype(np.float32)  # (N, 21*32) = (N, 672)
emb_8d = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy()
pos_arr = seg_df['pos_segment'].to_numpy()

print(f'  mel shape: {mel.shape}, emb_8d shape: {emb_8d.shape}')


def aggregate_voc(features, idx_arr):
    """concat(mean, std) per vocalization."""
    f = features[idx_arr]
    return np.concatenate([f.mean(0), f.std(0)])


def build_dataset(features):
    """Group по file_name, возвращает (X, y_ctx, y_em) per voc."""
    df = pd.DataFrame({'idx': np.arange(len(seg_df)), 'file': file_arr, 'pos': pos_arr,
                       'ctx': ctx, 'em': em_arr})
    df = df[df['em'] != 0]
    df = df[df['ctx'].isin(HP1_CTX)]
    X, y_ctx, y_em = [], [], []
    for fname, g in df.sort_values('pos').groupby('file', sort=False):
        idxs = g['idx'].to_numpy()
        if len(idxs) < 1: continue
        X.append(aggregate_voc(features, idxs))
        y_ctx.append(CTX2IDX[int(np.bincount(g['ctx'].to_numpy()).argmax())])
        y_em.append(int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0]))
    return np.array(X, dtype=np.float32), np.array(y_ctx), np.array(y_em)


print('Building datasets...')
X_mel, y_ctx, y_em = build_dataset(mel)
X_8d, _, _ = build_dataset(emb_8d)
print(f'  vocs: {len(X_mel)}, mel-agg dim: {X_mel.shape[1]}, 8d-agg dim: {X_8d.shape[1]}')


class TwoFC(nn.Module):
    def __init__(self, in_dim, hidden, n_cls, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_cls)
        )
    def forward(self, x): return self.net(x)


class VocDS(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32), int(self.y[i])


def split_by_emitter_idx(y_em, seed):
    rng = np.random.default_rng(seed)
    em = np.array(sorted(set(y_em.tolist()))); rng.shuffle(em)
    test_em = set(em[:11].tolist())
    tr_idx = np.where(~np.isin(y_em, list(test_em)))[0]
    te_idx = np.where(np.isin(y_em, list(test_em)))[0]
    return tr_idx, te_idx


def evaluate(model, dl):
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(DEVICE); y = y.to(DEVICE)
            pred.append(model(x).argmax(-1).cpu().numpy())
            true.append(y.cpu().numpy())
    pred = np.concatenate(pred); true = np.concatenate(true)
    return f1_score(true, pred, average='weighted'), f1_score(true, pred, average='macro')


def run(name, X, seed, hidden=128, lr=1e-3, n_ep=80, batch=64):
    in_dim = X.shape[1]
    tr_idx, te_idx = split_by_emitter_idx(y_em, seed)
    # Z-normalize по обучающей
    mu = X[tr_idx].mean(0); sd = X[tr_idx].std(0) + 1e-6
    Xn = (X - mu) / sd
    ds_tr = VocDS(Xn[tr_idx], y_ctx[tr_idx])
    ds_te = VocDS(Xn[te_idx], y_ctx[te_idx])
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False)

    model = TwoFC(in_dim, hidden, len(HP1_CTX)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # class weights inverse freq
    counts = np.bincount(y_ctx[tr_idx], minlength=len(HP1_CTX))
    w = (counts.sum() / (len(HP1_CTX) * counts.clip(min=1))).astype(np.float32)
    cw = torch.tensor(w).to(DEVICE)

    best_f1w = 0
    for ep in range(n_ep):
        model.train(); tot, n = 0., 0
        for x, y in dl_tr:
            x = x.to(DEVICE); y = y.to(DEVICE)
            loss = F.cross_entropy(model(x), y, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        if (ep + 1) % 20 == 0:
            f1w, f1m = evaluate(model, dl_te)
            best_f1w = max(best_f1w, f1w)

    f1w, f1m = evaluate(model, dl_te)
    return f1w, f1m, best_f1w


print('\n=== Two FC fine-tune, 5 seeds ===')
results = []
configs = {'mel-agg': X_mel, 'umap8d-agg': X_8d}
for seed in range(5):
    print(f'\n--- Seed {seed} ---', flush=True)
    for name, X in configs.items():
        t0 = time.time()
        f1w, f1m, best = run(name, X, seed)
        elapsed = time.time() - t0
        print(f"  >>> {name}/seed={seed}: f1w={f1w:.3f} (best {best:.3f}), f1m={f1m:.3f}  ({elapsed:.1f}s)",
              flush=True)
        results.append({'name': name, 'seed': seed, 'f1_w': f1w, 'f1_m': f1m,
                        'f1_w_best': best, 'time_s': elapsed})

df = pd.DataFrame(results)
df.to_csv('docs/thesis/figures/two_fc_results.csv', index=False)
agg = df.groupby('name').agg(f1w_mean=('f1_w', 'mean'), f1w_std=('f1_w', 'std'),
                              f1m_mean=('f1_m', 'mean'), f1m_std=('f1_m', 'std')).reset_index()
agg.to_csv('docs/thesis/figures/two_fc_summary.csv', index=False)
print('\n=== Summary ===')
print(agg.to_string(index=False))
print('\nSaved: docs/thesis/figures/two_fc_*.csv')
