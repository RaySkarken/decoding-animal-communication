"""Shared sequence-modeling utilities for the bat-token conference experiments.

Label-agnostic TinyBERT (MLM pretrain + CLS head), within-vocalization shuffle
control, bag-of-tokens LR baseline, and helpers. Imported by the per-task scripts
(context, caller-ID, next-token) so the model/eval is identical across tasks.
No module-level data loading — pass arrays in.
"""
from __future__ import annotations
from collections import Counter
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]


class TinyBERT(nn.Module):
    def __init__(self, vocab, n_cls, d=64, nl=2, nh=4, max_len=64, drop=0.1):
        super().__init__()
        self.PAD, self.CLS, self.MASK, self.off = 0, 1, 2, 3
        self.full = vocab + 3
        self.te = nn.Embedding(self.full, d, padding_idx=0)
        self.pe = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, nh, d * 4, drop, batch_first=True,
                                           activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl)
        self.norm = nn.LayerNorm(d)
        self.mlm = nn.Linear(d, self.full)
        self.cls = nn.Linear(d, n_cls)
        self.max_len = max_len

    def forward(self, x, mlm=False, cls=False):
        L = x.size(1)
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.norm(self.enc(self.te(x) + self.pe(pos), src_key_padding_mask=(x == 0)))
        o = {}
        if mlm: o['mlm'] = self.mlm(h)
        if cls: o['cls'] = self.cls(h[:, 0])
        return o


def enc_seq(seq, max_len, m, srng=None):
    seq = list(seq[:max_len - 1])
    if srng is not None:
        srng.shuffle(seq)
    ids = [m.CLS] + [t + m.off for t in seq]
    return ids + [m.PAD] * (max_len - len(ids))


class DS(Dataset):
    def __init__(self, vocs, m, max_len, y2i, label_key, mlm=False, shuf=False, seed=0):
        self.v, self.m, self.max_len = vocs, m, max_len
        self.y2i, self.lk, self.mlm, self.shuf, self.seed = y2i, label_key, mlm, shuf, seed

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        sr = np.random.default_rng(self.seed * 1000003 + i) if self.shuf else None
        ids = torch.tensor(enc_seq(self.v[i]['seq'], self.max_len, self.m, sr), dtype=torch.long)
        y = self.y2i[self.v[i][self.lk]]
        if not self.mlm:
            return ids, y
        tgt = ids.clone()
        mp = (ids >= self.m.off) & (torch.rand_like(ids.float()) < 0.15)
        ids[mp] = self.m.MASK
        tgt[~mp] = -100
        return ids, tgt, y


def train_eval_bert(tr, te, vocab, y2i, label_key, seed, shuf, pre=8, ft=15, max_len_cap=64):
    torch.manual_seed(seed); np.random.seed(seed)
    n_cls = len(y2i)
    max_len = min(max_len_cap, max(len(v['seq']) for v in tr) + 1)
    m = TinyBERT(vocab, n_cls, max_len=max_len).to(DEVICE)
    dl = DataLoader(DS(tr, m, max_len, y2i, label_key, mlm=True, shuf=shuf, seed=seed),
                    batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(pre):
        m.train()
        for ids, tgt, _ in dl:
            ids, tgt = ids.to(DEVICE), tgt.to(DEVICE)
            loss = F.cross_entropy(m(ids, mlm=True)['mlm'].view(-1, m.full), tgt.view(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
    dltr = DataLoader(DS(tr, m, max_len, y2i, label_key, shuf=shuf, seed=seed),
                      batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    for _ in range(ft):
        m.train()
        for ids, y in dltr:
            ids, y = ids.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(m(ids, cls=True)['cls'], y)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    dlte = DataLoader(DS(te, m, max_len, y2i, label_key, shuf=shuf, seed=seed), batch_size=256)
    pred, true = [], []
    with torch.no_grad():
        for ids, y in dlte:
            pred.append(m(ids.to(DEVICE), cls=True)['cls'].argmax(-1).cpu().numpy())
            true.append(y.numpy())
    pred, true = np.concatenate(pred), np.concatenate(true)
    return f1_score(true, pred, average='macro', labels=range(n_cls), zero_division=0)


def bag_lr(tr, te, vocab, y2i, label_key):
    def hist(vocs):
        X = np.zeros((len(vocs), vocab), np.float32)
        for i, v in enumerate(vocs):
            for t in v['seq']:
                X[i, t] += 1
            X[i] /= max(1, X[i].sum())
        return X, np.array([y2i[v[label_key]] for v in vocs])
    Xtr, ytr = hist(tr); Xte, yte = hist(te)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xtr, ytr)
    return f1_score(yte, clf.predict(Xte), average='macro', labels=range(len(y2i)), zero_division=0)


def build_vocs(token_arr, file_arr, pos_arr, ctx, em_arr):
    df = pd.DataFrame({'file': file_arr, 'pos': pos_arr, 'tok': token_arr, 'ctx': ctx, 'em': em_arr})
    df = df[(df['tok'] >= 0) & (df['em'] != 0) & (df['ctx'].isin(HP1_CTX))]
    vocs = []
    for fn, g in df.sort_values('pos').groupby('file', sort=False):
        seq = g['tok'].to_numpy().tolist()
        if not seq:
            continue
        vocs.append({'seq': seq,
                     'ctx': int(np.bincount(g['ctx'].to_numpy()).argmax()),
                     'em': abs(int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0]))})
    return vocs


def ci95(d):
    d = np.array(d, float); m = d.mean()
    h = 2.776 * d.std(ddof=1) / np.sqrt(len(d))
    return m, m - h, m + h
