"""Next-token prediction as an intrinsic test of sequential structure.

A causal LM trained on real token order should achieve LOWER perplexity than one
trained on within-vocalization shuffled order IF the sequences carry n-gram /
sequential structure. If real ~= shuffled perplexity, the token stream has no
usable order structure (each token is ~conditionally independent given the rest)
— an intrinsic, label-free corroboration of the classification order-tests.

Run per tokenizer (mel-UMAP k-means, SSL k-means). Cross-bat split, 5 seeds.
Metrics: test bits-per-token (lower=better), next-token acc@1. Report Δ(shuf-real):
positive Δ means real order is more predictable (structure exists).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import time, sys
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
N_SEEDS = 5
V_TOK = 30
MAXLEN = 64
PAD, BOS = 0, 1
LN2 = np.log(2)
print(f'Device: {DEVICE}', flush=True)

print('Loading state...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
ctx = seg_df['context'].to_numpy(); em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy(); pos_arr = seg_df['pos_segment'].to_numpy()


def build_vocs(tok):
    df = pd.DataFrame({'file': file_arr, 'pos': pos_arr, 'tok': tok, 'ctx': ctx, 'em': em_arr})
    df = df[(df['tok'] >= 0) & (df['em'] != 0) & (df['ctx'].isin(HP1_CTX))]
    vocs = []
    for fn, g in df.sort_values('pos').groupby('file', sort=False):
        seq = g['tok'].to_numpy().tolist()
        if len(seq) < 2:    # need >=2 tokens for next-token
            continue
        vocs.append({'seq': seq, 'em': abs(int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0]))})
    return vocs


def split(vocs, seed):
    rng = np.random.default_rng(seed); ems = np.array(sorted(set(v['em'] for v in vocs))); rng.shuffle(ems)
    test = set(ems[:11].tolist())
    return [v for v in vocs if v['em'] not in test], [v for v in vocs if v['em'] in test]


class CausalLM(nn.Module):
    def __init__(self, vocab, d=64, nl=2, nh=4, max_len=MAXLEN):
        super().__init__()
        self.full = vocab + 2  # PAD,BOS
        self.te = nn.Embedding(self.full, d, padding_idx=PAD)
        self.pe = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, nh, d * 4, 0.1, batch_first=True, activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl); self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, self.full); self.max_len = max_len

    def forward(self, x):
        L = x.size(1); pos = torch.arange(L, device=x.device).unsqueeze(0)
        cm = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        h = self.norm(self.enc(self.te(x) + self.pe(pos), mask=cm,
                               src_key_padding_mask=(x == PAD), is_causal=True))
        return self.head(h)


def encode(seq, shuf_rng=None):
    seq = list(seq[:MAXLEN - 1])
    if shuf_rng is not None: shuf_rng.shuffle(seq)
    ids = [BOS] + [t + 2 for t in seq]
    return ids + [PAD] * (MAXLEN - len(ids))


class DS(Dataset):
    def __init__(self, vocs, shuf=False, seed=0):
        self.v, self.shuf, self.seed = vocs, shuf, seed
    def __len__(self): return len(self.v)
    def __getitem__(self, i):
        sr = np.random.default_rng(self.seed * 1000003 + i) if self.shuf else None
        ids = encode(self.v[i]['seq'], sr)
        x = torch.tensor(ids[:-1]); y = torch.tensor(ids[1:])
        return x, y


def train_eval(tr, te, vocab, seed, shuf, epochs=20):
    torch.manual_seed(seed); np.random.seed(seed)
    m = CausalLM(vocab).to(DEVICE)
    dl = DataLoader(DS(tr, shuf, seed), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(epochs):
        m.train()
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(m(x).reshape(-1, m.full), y.reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    dlte = DataLoader(DS(te, shuf, seed), batch_size=256)
    tot_nll, ntok, hit = 0.0, 0, 0
    with torch.no_grad():
        for x, y in dlte:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = m(x); logp = F.log_softmax(logits, -1)
            valid = (y != PAD)
            nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            tot_nll += (nll * valid).sum().item(); ntok += valid.sum().item()
            hit += ((logits.argmax(-1) == y) & valid).sum().item()
    bpt = tot_nll / max(ntok, 1) / LN2     # bits per token
    return bpt, hit / max(ntok, 1)


def ci95(d):
    d = np.array(d, float); m = d.mean(); h = 2.776 * d.std(ddof=1) / np.sqrt(len(d)); return m, m - h, m + h


if __name__ == '__main__':
    toks = {'kmeans30': KMeans(V_TOK, n_init=10, random_state=0).fit_predict(emb).astype(np.int32)}
    sslp = CACHE / 'ssl_encoder_token.pt'
    if sslp.exists():
        # reuse SSL encoder from ssl_token_order.py to tokenize
        from ssl_token_order import Encoder, embed_all
        mel = st['tf_specs'].reshape(len(seg_df), -1).astype(np.float32)
        mu, sd = mel.mean(0), mel.std(0) + 1e-6
        enc = Encoder().to(DEVICE); enc.load_state_dict(torch.load(sslp, map_location=DEVICE))
        Z = embed_all(enc, (mel - mu) / sd)
        toks['ssl30'] = KMeans(V_TOK, n_init=10, random_state=0).fit_predict(Z).astype(np.int32)

    rows = []
    for name, tk in toks.items():
        vocs = build_vocs(tk)
        print(f'\n=== {name}: {len(vocs)} vocs ===', flush=True)
        for seed in range(N_SEEDS):
            t0 = time.time()
            tr, te = split(vocs, seed)
            r_bpt, r_acc = train_eval(tr, te, V_TOK, seed, shuf=False)
            s_bpt, s_acc = train_eval(tr, te, V_TOK, seed, shuf=True)
            print(f'  seed {seed}: real bpt={r_bpt:.3f} acc={r_acc:.3f} | shuf bpt={s_bpt:.3f} acc={s_acc:.3f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
            rows.append({'tokenizer': name, 'seed': seed, 'real_bpt': r_bpt, 'real_acc1': r_acc,
                         'shuf_bpt': s_bpt, 'shuf_acc1': s_acc})
        pd.DataFrame(rows).to_csv(OUT / 'next_token_order.csv', index=False)
    df = pd.DataFrame(rows)
    print('\n=== Next-token: does real order predict better than shuffled? ===', flush=True)
    for name in toks:
        sub = df[df.tokenizer == name]
        # shuffled bpt - real bpt > 0 means real order is more predictable
        m, lo, hi = ci95((sub.shuf_bpt - sub.real_bpt).tolist())
        print(f'{name:10s} real_bpt={sub.real_bpt.mean():.3f} shuf_bpt={sub.shuf_bpt.mean():.3f} '
              f'Δ(shuf-real)={m:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] '
              f'{"ORDER PREDICTABLE" if lo>0 else "no seq structure"}', flush=True)
    print(f'Saved: {OUT/"next_token_order.csv"}', flush=True)
