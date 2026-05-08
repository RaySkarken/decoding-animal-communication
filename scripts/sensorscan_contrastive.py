"""SensorSCAN-style contrastive learning на мел-сегментах.

Идея:
  - Encoder: 2-layer MLP на мел-векторе (672D → 128D representation).
  - NT-Xent loss: positive pair = два сегмента из одной вокализации;
    negative = сегменты из разных вокализаций.
  - Контекст не используется при обучении encoder'а (self-supervised).
  - После предтренинга: linear probe на представлениях для классификации контекста.

Отличие от обычного SimCLR: positive pair определяется через "часть одной вокализации",
а не через аугментацию того же примера. Это работает потому, что внутри вокализации
сегменты тематически связаны (одинаковый контекст → акустически близки в среднем).

Это "weak supervision through structure" — близко к SensorSCAN, где временная
близость использовалась как сигнал того что это "одна и та же ситуация".

Сравнение с baseline:
  RF на raw mel agg = 0.529 (наш контрольный)
  Two FC на raw mel = 0.558 (предыдущий эксп.)

Если SensorSCAN representation + linear даст близкое или лучше — это интересно.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, time
from pathlib import Path
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

import os
DEVICE = torch.device('cpu') if os.environ.get('FORCE_CPU') else (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
print(f'Device: {DEVICE}')
CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX2IDX = {c: i for i, c in enumerate(HP1_CTX)}

print('Loading state...')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
mel = st['tf_specs'].reshape(len(seg_df), -1).astype(np.float32)
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy()


class Encoder(nn.Module):
    def __init__(self, in_dim=672, hidden=256, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, out),
        )
    def forward(self, x): return self.net(x)


class Projection(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    def forward(self, h): return self.head(h)


def nt_xent_loss(z1, z2, temperature=0.5):
    """Symmetric NT-Xent."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, d)
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature  # (2B, 2B)
    # mask self
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)
    # positives: i ↔ i+B
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, targets)


# Build positive pairs: (seg_a, seg_b) from same vocalization
print('Building positive pairs from vocalizations...')
file_to_idxs = {}
for i, f in enumerate(file_arr):
    if em_arr[i] == 0 or ctx[i] not in HP1_CTX: continue
    file_to_idxs.setdefault(f, []).append(i)
file_to_idxs = {k: v for k, v in file_to_idxs.items() if len(v) >= 2}
files_list = list(file_to_idxs.keys())
print(f'  vocalizations with ≥2 segments: {len(files_list)}')


class PairDS(Dataset):
    def __init__(self, files, file_to_idxs, mel, train_files):
        # train_files = set of file names from train emitters
        self.files = [f for f in files if f in train_files]
        self.f2i = file_to_idxs
        self.mel = mel
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        idxs = self.f2i[self.files[i]]
        a, b = np.random.choice(idxs, 2, replace=False)
        return torch.tensor(self.mel[a], dtype=torch.float32), torch.tensor(self.mel[b], dtype=torch.float32)


# Aggregate representations для linear probe (по вокализациям)
def voc_repr(encoder, mel_arr, file_arr, mask_keep):
    encoder.eval()
    df = pd.DataFrame({'idx': np.arange(len(file_arr)), 'file': file_arr})
    df = df[mask_keep]
    repr_per_voc, y_ctx, y_em = [], [], []
    files_, idxs_per_voc = [], []
    with torch.no_grad():
        for fname, g in df.groupby('file'):
            idxs = g['idx'].to_numpy()
            x = torch.tensor(mel_arr[idxs], dtype=torch.float32).to(DEVICE)
            h = encoder(x).cpu().numpy()  # (n, 128)
            repr_per_voc.append(np.concatenate([h.mean(0), h.std(0)]))
            y_ctx.append(CTX2IDX[int(np.bincount(ctx[idxs]).argmax())])
            y_em.append(int(Counter(em_arr[idxs].tolist()).most_common(1)[0][0]))
            files_.append(fname)
    return np.array(repr_per_voc, dtype=np.float32), np.array(y_ctx), np.array(y_em)


def split_by_emitter(seed):
    rng = np.random.default_rng(seed)
    em = sorted(set(em_arr[em_arr != 0].tolist()))
    em = np.array(em); rng.shuffle(em)
    return set(em[:11].tolist())


# Z-normalize mel
mu = mel.mean(0); sd = mel.std(0) + 1e-6
mel_n = (mel - mu) / sd

print('\n=== SensorSCAN contrastive pretraining + linear probe, 5 seeds ===')
results = []
for seed in range(5):
    test_em = split_by_emitter(seed)
    train_files = set([f for f in file_to_idxs
                        if em_arr[file_to_idxs[f][0]] not in test_em])
    print(f'\n--- seed {seed}: train_files={len(train_files)}, test_em={len(test_em)} ---', flush=True)

    encoder = Encoder().to(DEVICE)
    projhead = Projection(128).to(DEVICE)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(projhead.parameters()),
                              lr=1e-3, weight_decay=1e-4)
    ds = PairDS(files_list, file_to_idxs, mel_n, train_files)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    n_ep = 20
    t0 = time.time()
    for ep in range(n_ep):
        encoder.train(); projhead.train(); tot, n = 0., 0
        for xa, xb in dl:
            xa = xa.to(DEVICE); xb = xb.to(DEVICE)
            za = projhead(encoder(xa)); zb = projhead(encoder(xb))
            loss = nt_xent_loss(za, zb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xa.size(0); n += xa.size(0)
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f'  pre ep{ep+1}: nt-xent={tot/n:.4f}', flush=True)

    # Linear probe on context
    mask_train = np.array([em_arr[i] not in test_em and em_arr[i] != 0 and ctx[i] in HP1_CTX
                            for i in range(len(em_arr))])
    mask_test = np.array([em_arr[i] in test_em and ctx[i] in HP1_CTX
                           for i in range(len(em_arr))])
    Xtr, ytr_ctx, _ = voc_repr(encoder, mel_n, file_arr, mask_train)
    Xte, yte_ctx, _ = voc_repr(encoder, mel_n, file_arr, mask_test)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xtr, ytr_ctx)
    pred = clf.predict(Xte)
    f1w = f1_score(yte_ctx, pred, average='weighted')
    f1m = f1_score(yte_ctx, pred, average='macro')
    elapsed = time.time() - t0
    print(f'  >>> seed={seed}: f1w={f1w:.3f}, f1m={f1m:.3f}  ({elapsed:.1f}s)', flush=True)
    results.append({'seed': seed, 'f1_w': f1w, 'f1_m': f1m, 'time_s': elapsed})


df = pd.DataFrame(results)
df.to_csv('docs/thesis/figures/sensorscan_results.csv', index=False)
print('\n=== Summary ===')
print(f'  weighted F1: {df.f1_w.mean():.3f} ± {df.f1_w.std():.3f}')
print(f'  macro    F1: {df.f1_m.mean():.3f} ± {df.f1_m.std():.3f}')
print(f'\nFor reference:')
print(f'  RF on raw mel agg (no tokens): 0.529 ± 0.058')
print(f'  Two FC on raw mel:             ~0.55')
print(f'  Per-context Bayes:             0.448 ± 0.078')
print(f'  Assom + RF baseline:           0.346 ± 0.075')
print(f'\nSaved: docs/thesis/figures/sensorscan_results.csv')
