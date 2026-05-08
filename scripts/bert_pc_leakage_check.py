"""Sanity check: confirms PC token-vocabulary leaks context regardless of ID layout.

Three setups, single seed (seed=0), small Transformer:
  PC-orig    — per-context tokens with disjoint ID ranges (the original setup)
  PC-shuff   — same per-context clustering, IDs randomly permuted (so range != context)
  PC-share   — per-context tokens RELABELED to a SHARED vocabulary of size 15
                 (each context gets its 15 codes, but mapped to {0..14} shared with all)

If all three give F1 ≈ 1, then leakage is in the SET of tokens (which set of token IDs
appear in a vocalization), not in the ID layout. If only PC-orig gives F1=1, then the
ID layout was the only issue and PC-shuff/PC-share would work.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, time
from pathlib import Path
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}')
CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX2IDX = {c: i for i, c in enumerate(HP1_CTX)}

st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy()
pos_arr = seg_df['pos_segment'].to_numpy()


def per_context_kmeans_offset(emb, ctx_arr, K=15):
    """Standard PC: each context gets disjoint ID range."""
    labels = np.full(len(emb), -1, dtype=np.int32)
    offset = 0
    for c in HP1_CTX:
        mc = ctx_arr == c
        if mc.sum() < 30: continue
        km = KMeans(n_clusters=min(K, mc.sum()//5), n_init=10, random_state=0).fit(emb[mc])
        labels[mc] = km.labels_ + offset
        offset += K
    return labels


def per_context_kmeans_shared(emb, ctx_arr, K=15):
    """PC with shared vocab: same K codes for all contexts (so ID == cluster index in [0,K-1])."""
    labels = np.full(len(emb), -1, dtype=np.int32)
    for c in HP1_CTX:
        mc = ctx_arr == c
        if mc.sum() < 30: continue
        km = KMeans(n_clusters=min(K, mc.sum()//5), n_init=10, random_state=0).fit(emb[mc])
        labels[mc] = km.labels_
    return labels


print('Computing tokens...')
pc_orig = per_context_kmeans_offset(emb, ctx, K=15)
pc_share = per_context_kmeans_shared(emb, ctx, K=15)

# PC-shuff: take pc_orig and permute the IDs randomly
rng = np.random.default_rng(123)
unique_ids = np.array(sorted(set(pc_orig[pc_orig >= 0].tolist())))
perm = rng.permutation(len(unique_ids))
id_map = {old: new for old, new in zip(unique_ids, unique_ids[perm])}
pc_shuff = pc_orig.copy()
mask = pc_orig >= 0
pc_shuff[mask] = np.array([id_map[i] for i in pc_orig[mask]])

variants = {'PC-orig': pc_orig, 'PC-shuff': pc_shuff, 'PC-share': pc_share}
for name, t in variants.items():
    valid = t >= 0
    print(f"  {name}: {len(set(t[valid].tolist()))} unique IDs, {valid.sum()} segments")


def build_vocs(token_arr):
    df = pd.DataFrame({'file': file_arr, 'pos': pos_arr, 'tok': token_arr,
                       'ctx': ctx, 'em': em_arr})
    df = df[df['tok'] >= 0]
    df = df[df['em'] != 0]
    df = df[df['ctx'].isin(HP1_CTX)]
    vocs = []
    for fname, g in df.sort_values('pos').groupby('file', sort=False):
        seq = g['tok'].to_numpy().tolist()
        if len(seq) < 1: continue
        cs = g['ctx'].to_numpy()
        es = g['em'].to_numpy()
        dom_ctx = int(np.bincount(cs).argmax())
        dom_em = int(Counter(es.tolist()).most_common(1)[0][0])
        vocs.append({'seq': seq, 'ctx': dom_ctx, 'em': dom_em})
    return vocs


class TinyBERT(nn.Module):
    def __init__(self, vocab_size, n_ctx, d_model=64, n_layers=2, n_heads=4, max_len=256):
        super().__init__()
        self.PAD, self.CLS = 0, 1
        self.tok_offset = 2
        self.full_vocab = vocab_size + 2
        self.tok_emb = nn.Embedding(self.full_vocab, d_model, padding_idx=self.PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                            dim_feedforward=d_model * 4,
                                            batch_first=True, activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, n_ctx)
        self.max_len = max_len

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        pad_mask = (x == self.PAD)
        h = self.enc(h, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        return self.cls_head(h[:, 0])


def encode(seq, max_len, model):
    seq = seq[:max_len - 1]
    ids = [model.CLS] + [t + model.tok_offset for t in seq]
    pad_n = max_len - len(ids)
    return ids + [model.PAD] * pad_n


class DS(Dataset):
    def __init__(self, vocs, model, max_len):
        self.vocs = vocs; self.m = model; self.L = max_len
    def __len__(self): return len(self.vocs)
    def __getitem__(self, i):
        v = self.vocs[i]
        return torch.tensor(encode(v['seq'], self.L, self.m), dtype=torch.long), CTX2IDX[v['ctx']]


def split_by_emitter(vocs, seed):
    rng = np.random.default_rng(seed)
    em = sorted(set(v['em'] for v in vocs))
    em = np.array(em); rng.shuffle(em)
    test_em = set(em[:11].tolist())
    return [v for v in vocs if v['em'] not in test_em], [v for v in vocs if v['em'] in test_em]


print('\n=== Sanity check: PC leakage source ===')
results = []
for name, t in variants.items():
    vocs = build_vocs(t)
    vocab_size = max(v['seq'][i] for v in vocs for i in range(len(v['seq']))) + 1
    max_len = min(256, max(len(v['seq']) for v in vocs) + 1)
    model = TinyBERT(vocab_size=vocab_size, n_ctx=len(HP1_CTX), max_len=max_len).to(DEVICE)
    tr, te = split_by_emitter(vocs, seed=0)
    ds_tr = DS(tr, model, max_len); ds_te = DS(te, model, max_len)
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=64, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f'\n[{name}] vocab={vocab_size}, train={len(tr)}, test={len(te)}')
    t0 = time.time()
    for ep in range(8):  # short — just to verify trivial trend
        model.train()
        for ids, y in dl_tr:
            ids = ids.to(DEVICE); y = y.to(DEVICE)
            loss = F.cross_entropy(model(ids), y)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    pred, true = [], []
    with torch.no_grad():
        for ids, y in dl_te:
            ids = ids.to(DEVICE); y = y.to(DEVICE)
            pred.append(model(ids).argmax(-1).cpu().numpy()); true.append(y.cpu().numpy())
    pred = np.concatenate(pred); true = np.concatenate(true)
    f1w = f1_score(true, pred, average='weighted')
    f1m = f1_score(true, pred, average='macro')
    elapsed = time.time() - t0
    print(f'  >>> {name}: f1_w={f1w:.3f}, f1_m={f1m:.3f}  ({elapsed:.1f}s)')
    results.append({'variant': name, 'vocab': vocab_size, 'f1_w': f1w, 'f1_m': f1m})

df = pd.DataFrame(results)
print('\n=== Summary ===')
print(df.to_string(index=False))
df.to_csv('docs/thesis/figures/bert_pc_leakage_check.csv', index=False)
print('\nSaved: docs/thesis/figures/bert_pc_leakage_check.csv')
