"""BERT-style Transformer pilot: per-context tokens vs global tokens.

Three tokenizations compared head-to-head with the SAME small Transformer:
  PC      — per-context k-means (vocab ≈ 120)
  G       — global HDBSCAN-NCA (vocab = 11)
  KM30    — global k-means (vocab = 30)

Pipeline:
  1. Load 152k segments + 3 tokenizations.
  2. Group into vocalization sequences (file_name, sorted by pos_segment).
  3. Cross-bat split: 30 train / 11 test emitters, 3 seeds.
  4. Architecture: small Transformer with [CLS], 2 layers, d=64, 4 heads.
  5. Stage 1 — SSL pretraining: masked token prediction on train sequences.
  6. Stage 2 — fine-tune: context classification (8 classes) on train, eval test.
  7. Report weighted/macro F1 per tokenization per seed.

Pilot scale — small model, short training, M3 Pro friendly.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys, os, time
import numpy as np, pandas as pd, joblib
from pathlib import Path
from collections import Counter, defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}')
CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX2IDX = {c: i for i, c in enumerate(HP1_CTX)}

# ---------- Data loading ----------
print('Loading state...')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
hdb_nca = np.load(CACHE / 'hdb_nca_labels_152k_21x32.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy()
pos_arr = seg_df['pos_segment'].to_numpy()


def per_context_kmeans(emb, ctx_arr, K=15):
    labels = np.full(len(emb), -1, dtype=np.int32)
    offset = 0
    for c in HP1_CTX:
        mc = ctx_arr == c
        if mc.sum() < 30: continue
        km = KMeans(n_clusters=min(K, mc.sum()//5), n_init=10, random_state=0).fit(emb[mc])
        labels[mc] = km.labels_ + offset
        offset += K
    return labels


def kmeans_global(emb, K):
    return KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(emb)


print('Computing tokenizations...')
toks = {}
toks['G'] = hdb_nca.astype(np.int32)
toks['KM30'] = kmeans_global(emb, 30).astype(np.int32)
toks['PC'] = per_context_kmeans(emb, ctx, K=15).astype(np.int32)
print(f"  G:     {len(set(toks['G'][toks['G']>=0].tolist()))} clusters")
print(f"  KM30:  {len(set(toks['KM30'].tolist()))} clusters")
print(f"  PC:    {len(set(toks['PC'][toks['PC']>=0].tolist()))} clusters")


# ---------- Build vocalization sequences ----------
def build_vocs(token_arr):
    """Group segments by file_name, sorted by pos_segment, into sequences."""
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


vocs_per_tok = {name: build_vocs(arr) for name, arr in toks.items()}
for name, vocs in vocs_per_tok.items():
    print(f"  {name}: {len(vocs)} vocalizations")


# ---------- Model ----------
class TinyBERT(nn.Module):
    def __init__(self, vocab_size, n_ctx, d_model=64, n_layers=2, n_heads=4,
                 max_len=512, dropout=0.1):
        super().__init__()
        # 0 = PAD, 1 = CLS, 2 = MASK, 3..3+vocab_size-1 = tokens
        self.PAD, self.CLS, self.MASK = 0, 1, 2
        self.tok_offset = 3
        self.full_vocab = vocab_size + 3
        self.tok_emb = nn.Embedding(self.full_vocab, d_model, padding_idx=self.PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                            dim_feedforward=d_model * 4,
                                            dropout=dropout, batch_first=True,
                                            activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, self.full_vocab)
        self.cls_head = nn.Linear(d_model, n_ctx)
        self.max_len = max_len

    def forward(self, x, return_mlm=False, return_cls=False):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        pad_mask = (x == self.PAD)
        h = self.enc(h, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        out = {}
        if return_mlm:
            out['mlm'] = self.mlm_head(h)  # (B, L, V_full)
        if return_cls:
            out['cls'] = self.cls_head(h[:, 0])  # CLS at position 0
        return out


def encode_seq(seq, max_len, model):
    """Build [CLS, t1+offset, t2+offset, ..., PAD, PAD]."""
    seq = seq[:max_len - 1]
    ids = [model.CLS] + [t + model.tok_offset for t in seq]
    pad_n = max_len - len(ids)
    ids = ids + [model.PAD] * pad_n
    return ids


class VocDataset(Dataset):
    def __init__(self, vocs, model, max_len, mlm_p=0.15, do_mlm=False):
        self.vocs = vocs
        self.model = model
        self.max_len = max_len
        self.mlm_p = mlm_p
        self.do_mlm = do_mlm

    def __len__(self):
        return len(self.vocs)

    def __getitem__(self, i):
        v = self.vocs[i]
        ids = encode_seq(v['seq'], self.max_len, self.model)
        ids_t = torch.tensor(ids, dtype=torch.long)
        ctx_idx = CTX2IDX[v['ctx']]
        if not self.do_mlm:
            return ids_t, ctx_idx
        # masked language modeling target
        target = ids_t.clone()
        # don't mask CLS or PAD
        candidates = (ids_t >= self.model.tok_offset)
        rnd = torch.rand_like(ids_t.float())
        mask_pos = candidates & (rnd < self.mlm_p)
        # 80% replace with MASK, 10% random, 10% keep
        rnd2 = torch.rand_like(rnd)
        replace_mask = mask_pos & (rnd2 < 0.8)
        replace_rand = mask_pos & (rnd2 >= 0.8) & (rnd2 < 0.9)
        ids_t[replace_mask] = self.model.MASK
        if replace_rand.sum() > 0:
            ids_t[replace_rand] = torch.randint(self.model.tok_offset,
                                                 self.model.full_vocab,
                                                 (replace_rand.sum().item(),))
        # only compute loss at masked positions
        ignore_idx = -100
        target[~mask_pos] = ignore_idx
        return ids_t, target, ctx_idx


def split_by_emitter(vocs, seed):
    rng = np.random.default_rng(seed)
    emitters = sorted(set(v['em'] for v in vocs))
    em_arr_local = np.array(emitters); rng.shuffle(em_arr_local)
    test_em = set(em_arr_local[:11].tolist())
    tr = [v for v in vocs if v['em'] not in test_em]
    te = [v for v in vocs if v['em'] in test_em]
    return tr, te


def evaluate(model, ds_test, batch_size=64):
    model.eval()
    dl = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    pred, true = [], []
    with torch.no_grad():
        for ids, y in dl:
            ids = ids.to(DEVICE); y = y.to(DEVICE)
            out = model(ids, return_cls=True)
            pred.append(out['cls'].argmax(-1).cpu().numpy())
            true.append(y.cpu().numpy())
    pred = np.concatenate(pred); true = np.concatenate(true)
    f1w = f1_score(true, pred, average='weighted')
    f1m = f1_score(true, pred, average='macro')
    return f1w, f1m


# ---------- Run experiments ----------
def run_one(name, vocs, seed, n_pretrain_epochs=10, n_finetune_epochs=20):
    vocab_size = max(v['seq'][i] for v in vocs for i in range(len(v['seq']))) + 1
    max_len = min(256, max(len(v['seq']) for v in vocs) + 1)
    model = TinyBERT(vocab_size=vocab_size, n_ctx=len(HP1_CTX), max_len=max_len).to(DEVICE)
    tr, te = split_by_emitter(vocs, seed)

    # Stage 1: SSL pretraining (masked token prediction)
    ds_pre = VocDataset(tr, model, max_len, mlm_p=0.15, do_mlm=True)
    dl_pre = DataLoader(ds_pre, batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    print(f"  [{name}/seed={seed}] pretrain: vocab={vocab_size}, max_len={max_len}, "
          f"train_n={len(tr)}, test_n={len(te)}")
    for ep in range(n_pretrain_epochs):
        model.train(); tot, n = 0.0, 0
        for ids, target, _ in dl_pre:
            ids = ids.to(DEVICE); target = target.to(DEVICE)
            out = model(ids, return_mlm=True)
            logits = out['mlm']
            loss = F.cross_entropy(logits.view(-1, model.full_vocab),
                                    target.view(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * ids.size(0); n += ids.size(0)
        if ep == 0 or ep == n_pretrain_epochs - 1:
            print(f"    pretrain ep{ep}: MLM loss = {tot/n:.4f}")

    # Stage 2: Fine-tune for context classification
    ds_tr = VocDataset(tr, model, max_len, do_mlm=False)
    ds_te = VocDataset(te, model, max_len, do_mlm=False)
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_f1w = 0
    for ep in range(n_finetune_epochs):
        model.train(); tot, n = 0.0, 0
        for ids, y in dl_tr:
            ids = ids.to(DEVICE); y = y.to(DEVICE)
            out = model(ids, return_cls=True)
            loss = F.cross_entropy(out['cls'], y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * ids.size(0); n += ids.size(0)
        if (ep + 1) % 5 == 0:
            f1w, f1m = evaluate(model, ds_te)
            print(f"    finetune ep{ep+1}: train_loss={tot/n:.4f}, f1_w={f1w:.3f}, f1_m={f1m:.3f}")
            best_f1w = max(best_f1w, f1w)

    # Final eval
    f1w, f1m = evaluate(model, ds_te)
    return f1w, f1m


print('\n=== BERT-style pilot ===')
results = []
for seed in range(3):
    print(f'\n--- Seed {seed} ---')
    for name, vocs in vocs_per_tok.items():
        t0 = time.time()
        f1w, f1m = run_one(name, vocs, seed)
        elapsed = time.time() - t0
        print(f"  >>> {name} seed={seed}: f1_w={f1w:.3f}, f1_m={f1m:.3f}  ({elapsed:.1f}s)")
        results.append({'name': name, 'seed': seed, 'f1_w': f1w, 'f1_m': f1m, 'time_s': elapsed})

df = pd.DataFrame(results)
print('\n=== Summary (mean ± std over seeds) ===')
agg = df.groupby('name').agg(f1w_mean=('f1_w', 'mean'), f1w_std=('f1_w', 'std'),
                              f1m_mean=('f1_m', 'mean'), f1m_std=('f1_m', 'std'),
                              time_s_mean=('time_s', 'mean')).reset_index()
print(agg.to_string(index=False))

df.to_csv('docs/thesis/figures/bert_pilot_results.csv', index=False)
agg.to_csv('docs/thesis/figures/bert_pilot_summary.csv', index=False)
print('\nSaved: docs/thesis/figures/bert_pilot_*.csv')
