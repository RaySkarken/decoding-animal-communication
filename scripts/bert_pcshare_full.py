"""Полный эксперимент: PC-share + BERT с предтренингом, 5 сидов.

PC-share: per-context k-means даёт центроиды, идентификаторы используют общий
диапазон {0..K-1} (без сдвига по контексту). Утечка контекста через словарь устранена.

Для сравнения параллельно прогоняются:
  KM-15 — глобальный k-means с тем же |V|=15, тот же протокол
  G     — Assom HDB+NCA (V=11)

Архитектура BERT — та же, что в bert_pilot.py.
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

print('Loading state...')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
hdb_nca = np.load(CACHE / 'hdb_nca_labels_152k_21x32.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy()
pos_arr = seg_df['pos_segment'].to_numpy()


def per_context_kmeans_shared(emb, ctx_arr, K=15):
    """PC-share: per-context k-means с общим диапазоном идентификаторов {0..K-1}."""
    labels = np.full(len(emb), -1, dtype=np.int32)
    for c in HP1_CTX:
        mc = ctx_arr == c
        if mc.sum() < 30: continue
        km = KMeans(n_clusters=min(K, mc.sum()//5), n_init=10, random_state=0).fit(emb[mc])
        labels[mc] = km.labels_
    return labels


def kmeans_global(emb, K):
    return KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(emb)


print('Computing tokenizations...')
toks = {}
toks['G'] = hdb_nca.astype(np.int32)
toks['KM15'] = kmeans_global(emb, 15).astype(np.int32)
toks['PCshare'] = per_context_kmeans_shared(emb, ctx, K=15).astype(np.int32)
for name, t in toks.items():
    valid = t >= 0
    print(f"  {name}: |V|={len(set(t[valid].tolist()))}, segments={valid.sum()}")


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


vocs_per_tok = {name: build_vocs(arr) for name, arr in toks.items()}
for name, vocs in vocs_per_tok.items():
    print(f"  {name}: {len(vocs)} vocalizations")


class TinyBERT(nn.Module):
    def __init__(self, vocab_size, n_ctx, d_model=64, n_layers=2, n_heads=4,
                 max_len=256, dropout=0.1):
        super().__init__()
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
        if return_mlm: out['mlm'] = self.mlm_head(h)
        if return_cls: out['cls'] = self.cls_head(h[:, 0])
        return out


def encode(seq, max_len, model):
    seq = seq[:max_len - 1]
    ids = [model.CLS] + [t + model.tok_offset for t in seq]
    pad_n = max_len - len(ids)
    return ids + [model.PAD] * pad_n


class DS(Dataset):
    def __init__(self, vocs, model, max_len, do_mlm=False, mlm_p=0.15):
        self.vocs = vocs; self.m = model; self.L = max_len
        self.do_mlm = do_mlm; self.mlm_p = mlm_p
    def __len__(self): return len(self.vocs)
    def __getitem__(self, i):
        v = self.vocs[i]
        ids = torch.tensor(encode(v['seq'], self.L, self.m), dtype=torch.long)
        ctx_idx = CTX2IDX[v['ctx']]
        if not self.do_mlm:
            return ids, ctx_idx
        target = ids.clone()
        cand = (ids >= self.m.tok_offset)
        rnd = torch.rand_like(ids.float())
        mask_pos = cand & (rnd < self.mlm_p)
        rnd2 = torch.rand_like(rnd)
        replace_mask = mask_pos & (rnd2 < 0.8)
        replace_rand = mask_pos & (rnd2 >= 0.8) & (rnd2 < 0.9)
        ids[replace_mask] = self.m.MASK
        if replace_rand.sum() > 0:
            ids[replace_rand] = torch.randint(self.m.tok_offset, self.m.full_vocab,
                                                (replace_rand.sum().item(),))
        target[~mask_pos] = -100
        return ids, target, ctx_idx


def split_by_emitter(vocs, seed):
    rng = np.random.default_rng(seed)
    em = sorted(set(v['em'] for v in vocs))
    em = np.array(em); rng.shuffle(em)
    test_em = set(em[:11].tolist())
    return [v for v in vocs if v['em'] not in test_em], [v for v in vocs if v['em'] in test_em]


def evaluate(model, ds_test, batch=64):
    model.eval()
    dl = DataLoader(ds_test, batch_size=batch, shuffle=False)
    pred, true = [], []
    with torch.no_grad():
        for ids, y in dl:
            ids = ids.to(DEVICE); y = y.to(DEVICE)
            out = model(ids, return_cls=True)
            pred.append(out['cls'].argmax(-1).cpu().numpy())
            true.append(y.cpu().numpy())
    pred = np.concatenate(pred); true = np.concatenate(true)
    return f1_score(true, pred, average='weighted'), f1_score(true, pred, average='macro')


def run_one(name, vocs, seed, n_pre=10, n_ft=20):
    vocab_size = max(v['seq'][i] for v in vocs for i in range(len(v['seq']))) + 1
    max_len = min(256, max(len(v['seq']) for v in vocs) + 1)
    model = TinyBERT(vocab_size=vocab_size, n_ctx=len(HP1_CTX), max_len=max_len).to(DEVICE)
    tr, te = split_by_emitter(vocs, seed)

    print(f"  [{name}/seed={seed}] |V|={vocab_size}, train={len(tr)}, test={len(te)}", flush=True)

    # Stage 1 — SSL pretrain
    ds_pre = DS(tr, model, max_len, do_mlm=True)
    dl_pre = DataLoader(ds_pre, batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for ep in range(n_pre):
        model.train(); tot, n = 0., 0
        for ids, target, _ in dl_pre:
            ids = ids.to(DEVICE); target = target.to(DEVICE)
            out = model(ids, return_mlm=True)
            loss = F.cross_entropy(out['mlm'].view(-1, model.full_vocab),
                                    target.view(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * ids.size(0); n += ids.size(0)
        if ep == 0 or ep == n_pre - 1:
            print(f"    pre ep{ep}: MLM={tot/n:.4f}", flush=True)

    # Stage 2 — fine-tune
    ds_tr = DS(tr, model, max_len, do_mlm=False)
    ds_te = DS(te, model, max_len, do_mlm=False)
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_f1w = 0
    for ep in range(n_ft):
        model.train(); tot, n = 0., 0
        for ids, y in dl_tr:
            ids = ids.to(DEVICE); y = y.to(DEVICE)
            out = model(ids, return_cls=True)
            loss = F.cross_entropy(out['cls'], y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * ids.size(0); n += ids.size(0)
        if (ep + 1) % 5 == 0:
            f1w, f1m = evaluate(model, ds_te)
            print(f"    ft ep{ep+1}: loss={tot/n:.4f}, f1w={f1w:.3f}, f1m={f1m:.3f}", flush=True)
            best_f1w = max(best_f1w, f1w)

    f1w, f1m = evaluate(model, ds_te)
    return f1w, f1m, best_f1w


print('\n=== PC-share + BERT, 5 seeds ===')
results = []
for seed in range(5):
    print(f'\n--- Seed {seed} ---', flush=True)
    for name, vocs in vocs_per_tok.items():
        t0 = time.time()
        f1w, f1m, best = run_one(name, vocs, seed)
        elapsed = time.time() - t0
        print(f"  >>> {name}/seed={seed}: f1w={f1w:.3f} (best {best:.3f}), f1m={f1m:.3f}  ({elapsed:.1f}s)",
              flush=True)
        results.append({'name': name, 'seed': seed, 'f1_w': f1w, 'f1_m': f1m,
                        'f1_w_best': best, 'time_s': elapsed})

df = pd.DataFrame(results)
df.to_csv('docs/thesis/figures/bert_pcshare_full_results.csv', index=False)
agg = df.groupby('name').agg(f1w_mean=('f1_w', 'mean'), f1w_std=('f1_w', 'std'),
                              f1m_mean=('f1_m', 'mean'), f1m_std=('f1_m', 'std')).reset_index()
agg.to_csv('docs/thesis/figures/bert_pcshare_full_summary.csv', index=False)
print('\n=== Summary ===')
print(agg.to_string(index=False))
print('\nSaved: docs/thesis/figures/bert_pcshare_full_*.csv')
