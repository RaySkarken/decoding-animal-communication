"""Do SSL-learned tokens beat mel-UMAP tokens, and does order matter for them?

The thesis showed domain-matched contrastive SSL on mel (continuous, linear probe)
reaches 0.385 macro F1 — the strongest representation. Here we VQ-tokenize those
SSL embeddings and ask:
  (1) do SSL tokens beat mel-UMAP k-means tokens for behavioral-context BERT?
  (2) does token ORDER matter for the better representation (shuffle control)?

Pipeline: SSL encoder (NT-Xent, positives = segments of same voc, NO labels) ->
embed all 152k segments -> k-means V=30 -> token sequences -> BERT (MLM+CLS) real
vs shuffled + bag-LR. Cross-bat 30/11, 5 seeds, macro F1 + 95% CI.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import time
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, silhouette_score
from sklearn.linear_model import LogisticRegression

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX2IDX = {c: i for i, c in enumerate(HP1_CTX)}
N_SEEDS = 5
V_TOK = 30
print(f'Device: {DEVICE}', flush=True)

print('Loading state...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
mel = st['tf_specs'].reshape(len(seg_df), -1).astype(np.float32)   # 672D
ctx = seg_df['context'].to_numpy(); em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy(); pos_arr = seg_df['pos_segment'].to_numpy()
mu, sd = mel.mean(0), mel.std(0) + 1e-6
mel_n = (mel - mu) / sd
print(f'  mel: {mel.shape}', flush=True)


# ─────────── SSL encoder (proven sensorscan-style) ───────────
class Encoder(nn.Module):
    def __init__(self, in_dim=672, hidden=256, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


class Proj(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.h = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

    def forward(self, x):
        return self.h(x)


def nt_xent(z1, z2, t=0.5):
    B = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)
    sim = z @ z.t() / t
    sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z.device), -1e9)
    tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def train_ssl_encoder(mel_n, file_arr, em_arr, ctx, epochs=25, cache=None, seed=0):
    enc = Encoder().to(DEVICE)
    if cache is not None and Path(cache).exists():
        enc.load_state_dict(torch.load(cache, map_location=DEVICE)); print('[ssl] loaded', flush=True)
        return enc
    torch.manual_seed(seed); np.random.seed(seed)
    # positive pairs: 2 segments from same voc (use all vocs, no labels)
    f2i = {}
    for i, f in enumerate(file_arr):
        if em_arr[i] == 0 or ctx[i] not in HP1_CTX:
            continue
        f2i.setdefault(f, []).append(i)
    files = [f for f, v in f2i.items() if len(v) >= 2]
    print(f'[ssl] vocs with >=2 segs: {len(files)}', flush=True)
    proj = Proj().to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for ep in range(epochs):
        rng.shuffle(files)
        enc.train(); proj.train(); tot = 0.0; nb = 0
        for s in range(0, len(files) - 256, 256):
            batch = files[s:s + 256]
            ia, ib = [], []
            for f in batch:
                a, b = rng.choice(f2i[f], 2, replace=False)
                ia.append(a); ib.append(b)
            xa = torch.from_numpy(mel_n[ia]).to(DEVICE)
            xb = torch.from_numpy(mel_n[ib]).to(DEVICE)
            za, zb = proj(enc(xa)), proj(enc(xb))
            loss = nt_xent(za, zb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f'[ssl] ep{ep+1}: NT-Xent={tot/max(nb,1):.4f} ({time.time()-t0:.0f}s)', flush=True)
    if cache is not None:
        torch.save(enc.state_dict(), cache)
    return enc


@torch.no_grad()
def embed_all(enc, mel_n):
    enc.eval()
    out = []
    for s in range(0, len(mel_n), 4096):
        out.append(enc(torch.from_numpy(mel_n[s:s + 4096]).to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


# ─────────── TinyBERT order-test (identical to seq_order_test) ───────────
class TinyBERT(nn.Module):
    def __init__(self, vocab, n_ctx, d=64, nl=2, nh=4, max_len=64, drop=0.1):
        super().__init__()
        self.PAD, self.CLS, self.MASK, self.off = 0, 1, 2, 3
        self.full = vocab + 3
        self.te = nn.Embedding(self.full, d, padding_idx=0)
        self.pe = nn.Embedding(max_len, d)
        l = nn.TransformerEncoderLayer(d, nh, d * 4, drop, batch_first=True, activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(l, nl); self.norm = nn.LayerNorm(d)
        self.mlm = nn.Linear(d, self.full); self.cls = nn.Linear(d, n_ctx); self.max_len = max_len

    def forward(self, x, mlm=False, cls=False):
        L = x.size(1); pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.norm(self.enc(self.te(x) + self.pe(pos), src_key_padding_mask=(x == 0)))
        o = {}
        if mlm: o['mlm'] = self.mlm(h)
        if cls: o['cls'] = self.cls(h[:, 0])
        return o


def enc_seq(seq, max_len, m, srng=None):
    seq = list(seq[:max_len - 1])
    if srng is not None: srng.shuffle(seq)
    ids = [m.CLS] + [t + m.off for t in seq]
    return ids + [m.PAD] * (max_len - len(ids))


class DS(Dataset):
    def __init__(self, vocs, m, max_len, mlm=False, shuf=False, seed=0):
        self.v, self.m, self.max_len, self.mlm, self.shuf, self.seed = vocs, m, max_len, mlm, shuf, seed
    def __len__(self): return len(self.v)
    def __getitem__(self, i):
        sr = np.random.default_rng(self.seed * 1000003 + i) if self.shuf else None
        ids = torch.tensor(enc_seq(self.v[i]['seq'], self.max_len, self.m, sr), dtype=torch.long)
        y = CTX2IDX[self.v[i]['ctx']]
        if not self.mlm: return ids, y
        tgt = ids.clone(); cand = ids >= self.m.off
        mp = cand & (torch.rand_like(ids.float()) < 0.15)
        ids[mp] = self.m.MASK; tgt[~mp] = -100
        return ids, tgt, y


def train_eval(tr, te, vocab, seed, shuf, pre=8, ft=15):
    torch.manual_seed(seed); np.random.seed(seed)
    max_len = min(64, max(len(v['seq']) for v in tr) + 1)
    m = TinyBERT(vocab, len(HP1_CTX), max_len=max_len).to(DEVICE)
    dl = DataLoader(DS(tr, m, max_len, mlm=True, shuf=shuf, seed=seed), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(pre):
        m.train()
        for ids, tgt, _ in dl:
            ids, tgt = ids.to(DEVICE), tgt.to(DEVICE)
            loss = F.cross_entropy(m(ids, mlm=True)['mlm'].view(-1, m.full), tgt.view(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
    dltr = DataLoader(DS(tr, m, max_len, shuf=shuf, seed=seed), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    for _ in range(ft):
        m.train()
        for ids, y in dltr:
            ids, y = ids.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(m(ids, cls=True)['cls'], y)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); dlte = DataLoader(DS(te, m, max_len, shuf=shuf, seed=seed), batch_size=256)
    pred, true = [], []
    with torch.no_grad():
        for ids, y in dlte:
            pred.append(m(ids.to(DEVICE), cls=True)['cls'].argmax(-1).cpu().numpy()); true.append(y.numpy())
    pred, true = np.concatenate(pred), np.concatenate(true)
    return f1_score(true, pred, average='macro', zero_division=0)


def bag_lr(tr, te, vocab, seed):
    def hist(vocs):
        X = np.zeros((len(vocs), vocab), np.float32)
        for i, v in enumerate(vocs):
            for t in v['seq']: X[i, t] += 1
            X[i] /= max(1, X[i].sum())
        return X, np.array([CTX2IDX[v['ctx']] for v in vocs])
    Xtr, ytr = hist(tr); Xte, yte = hist(te)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xtr, ytr)
    return f1_score(yte, clf.predict(Xte), average='macro', zero_division=0)


def build_vocs(tok):
    df = pd.DataFrame({'file': file_arr, 'pos': pos_arr, 'tok': tok, 'ctx': ctx, 'em': em_arr})
    df = df[(df['tok'] >= 0) & (df['em'] != 0) & (df['ctx'].isin(HP1_CTX))]
    vocs = []
    for fn, g in df.sort_values('pos').groupby('file', sort=False):
        seq = g['tok'].to_numpy().tolist()
        if not seq: continue
        vocs.append({'seq': seq, 'ctx': int(np.bincount(g['ctx'].to_numpy()).argmax()),
                     'em': abs(int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0]))})
    return vocs


def split(vocs, seed):
    rng = np.random.default_rng(seed); ems = np.array(sorted(set(v['em'] for v in vocs))); rng.shuffle(ems)
    test = set(ems[:11].tolist())
    return [v for v in vocs if v['em'] not in test], [v for v in vocs if v['em'] in test]


def ci95(d):
    d = np.array(d); m = d.mean(); h = 2.776 * d.std(ddof=1) / np.sqrt(len(d)); return m, m - h, m + h


if __name__ == '__main__':
    enc = train_ssl_encoder(mel_n, file_arr, em_arr, ctx, cache=CACHE / 'ssl_encoder_token.pt')
    Z = embed_all(enc, mel_n)
    print(f'SSL embeddings: {Z.shape}', flush=True)
    ssl_tok = KMeans(n_clusters=V_TOK, n_init=10, random_state=0).fit_predict(Z).astype(np.int32)
    ridx = np.random.default_rng(0).choice(len(Z), 20000, replace=False)
    sil = silhouette_score(Z[ridx], ssl_tok[ridx])
    print(f'SSL-token silhouette: {sil:.3f}', flush=True)

    vocs = build_vocs(ssl_tok)
    print(f'vocs: {len(vocs)}', flush=True)
    rows = []
    for seed in range(N_SEEDS):
        t0 = time.time()
        tr, te = split(vocs, seed)
        rm = train_eval(tr, te, V_TOK, seed, shuf=False)
        sm = train_eval(tr, te, V_TOK, seed, shuf=True)
        bm = bag_lr(tr, te, V_TOK, seed)
        print(f'  seed {seed}: SSL-tok BERT real={rm:.3f} shuf={sm:.3f} bag={bm:.3f} ({time.time()-t0:.0f}s)', flush=True)
        rows.append({'tokenizer': 'ssl_kmeans30', 'seed': seed, 'silhouette': round(sil, 4),
                     'bert_real_macro': rm, 'bert_shuf_macro': sm, 'bag_lr_macro': bm})
        pd.DataFrame(rows).to_csv(OUT / 'ssl_token_order.csv', index=False)
    df = pd.DataFrame(rows)
    m, lo, hi = ci95((df.bert_real_macro - df.bert_shuf_macro).tolist())
    print(f'\nSSL tokens: real={df.bert_real_macro.mean():.3f}±{df.bert_real_macro.std():.3f} '
          f'shuf={df.bert_shuf_macro.mean():.3f} bag={df.bag_lr_macro.mean():.3f} '
          f'Δ(real-shuf)={m:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] {"ORDER HELPS" if lo>0 else "order n.s."}', flush=True)
    print(f'Saved: {OUT/"ssl_token_order.csv"}', flush=True)
