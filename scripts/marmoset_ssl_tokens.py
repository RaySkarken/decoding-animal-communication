"""Does the domain-matched SSL tokenizer also win on marmosets? (cross-species Finding 2)

Train a frame-level contrastive SSL encoder on marmoset mel-frames (positives = two
frames from the same call), tokenize via k-means, and compare call-type/caller macro
F1 against mel-frame k-means tokens. If SSL tokens win here too -> the SSL-tokenizer
advantage generalizes across species.

bag-LR (CPU) primary metric (bag ~= BERT for token quality; CPU-cheap). 5 random
splits stratified by label.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
CACHE = Path('/Volumes/T7/datasets/InfantMarmosetsVox/cache')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
V_TOK = 30
N_SEEDS = 5


def load_calls():
    calls = []
    for f in sorted(CACHE.glob('*.npz')):
        if f.name.startswith('._'):
            continue
        z = np.load(f, allow_pickle=True)
        for fr, ct, cl in zip(z['frames'], z['calltype'], z['caller']):
            fr = np.asarray(fr, dtype=np.float32)
            if fr.ndim == 2 and len(fr) >= 2:
                calls.append({'frames': fr, 'calltype': int(ct), 'caller': int(cl)})
    return calls


class Enc(nn.Module):
    def __init__(self, d_in=32, h=128, out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.BatchNorm1d(h),
                                 nn.Linear(h, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Linear(h, out))

    def forward(self, x): return self.net(x)


def nt_xent(z1, z2, t=0.5):
    B = z1.size(0); z = F.normalize(torch.cat([z1, z2], 0), 1)
    sim = z @ z.t() / t
    sim.masked_fill_(torch.eye(2 * B, dtype=torch.bool, device=z.device), -1e9)
    tgt = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def train_ssl(calls, mu, sd, epochs=15, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    enc = Enc().to(DEVICE); proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)).to(DEVICE)
    # positive pairs = two frames from the same call (calls with >=2 frames)
    pool = [((c['frames'] - mu) / sd) for c in calls if len(c['frames']) >= 2]
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed); t0 = time.time()
    idx = np.arange(len(pool))
    for ep in range(epochs):
        rng.shuffle(idx); tot = 0.0; nb = 0
        for s in range(0, len(idx) - 256, 256):
            batch = idx[s:s + 256]
            a = np.stack([pool[i][rng.integers(len(pool[i]))] for i in batch])
            b = np.stack([pool[i][rng.integers(len(pool[i]))] for i in batch])
            xa = torch.from_numpy(a).float().to(DEVICE); xb = torch.from_numpy(b).float().to(DEVICE)
            loss = nt_xent(proj(enc(xa)), proj(enc(xb)))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f'[ssl] ep{ep+1}: {tot/max(nb,1):.4f} ({time.time()-t0:.0f}s)', flush=True)
    return enc


@torch.no_grad()
def embed(enc, frames, mu, sd):
    enc.eval(); x = torch.from_numpy(((frames - mu) / sd)).float().to(DEVICE)
    return enc(x).cpu().numpy()


def split_strat(vocs, key, seed, frac=0.2):
    rng = np.random.default_rng(seed); by = {}
    for i, v in enumerate(vocs): by.setdefault(v[key], []).append(i)
    tr, te = [], []
    for k, idxs in by.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        n = max(1, int(len(idxs) * frac)); te += idxs[:n].tolist(); tr += idxs[n:].tolist()
    return [vocs[i] for i in tr], [vocs[i] for i in te]


def bag_lr(tr, te, vocab, y2i, key):
    def hist(vs):
        X = np.zeros((len(vs), vocab), np.float32)
        for i, v in enumerate(vs):
            for t in v['seq']: X[i, t] += 1
            s = X[i].sum()
            if s: X[i] /= s
        return X, np.array([y2i[v[key]] for v in vs])
    Xtr, ytr = hist(tr); Xte, yte = hist(te)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xtr, ytr)
    return f1_score(yte, clf.predict(Xte), average='macro', labels=range(len(y2i)), zero_division=0)


def ci95(d):
    d = np.array(d, float); m = d.mean(); h = 2.776 * d.std(ddof=1) / np.sqrt(len(d)); return m, m - h, m + h


if __name__ == '__main__':
    calls = load_calls()
    print(f'cached calls: {len(calls):,}', flush=True)
    allf = np.concatenate([c['frames'] for c in calls])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6

    # mel-frame k-means tokens
    km_mel = KMeans(V_TOK, n_init=10, random_state=0).fit((allf - mu) / sd)
    # SSL tokens
    enc = train_ssl(calls, mu, sd)
    Z = embed(enc, allf, mu, sd)
    km_ssl = KMeans(V_TOK, n_init=10, random_state=0).fit(Z)

    # assign tokens per call
    off = 0
    for c in calls:
        n = len(c['frames'])
        c['mel_seq'] = km_mel.predict(((c['frames'] - mu) / sd)).tolist()
        c['ssl_seq'] = km_ssl.predict(Z[off:off + n]).tolist()
        off += n

    rows = []
    for task in ['calltype', 'caller']:
        labs = sorted({c[task] for c in calls})
        y2i = {l: i for i, l in enumerate(labs)}
        for src in ['mel', 'ssl']:
            for c in calls: c['seq'] = c[f'{src}_seq']
            f1s = []
            for seed in range(N_SEEDS):
                tr, te = split_strat(calls, task, seed)
                f1s.append(bag_lr(tr, te, V_TOK, y2i, task))
            m, lo, hi = ci95(f1s)
            print(f'{task:9s} {src:3s}: macroF1={m:.3f} 95%CI[{lo:.3f},{hi:.3f}]', flush=True)
            rows.append({'task': task, 'tokenizer': src, 'macro_f1': round(m, 4),
                         'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4)})
        pd.DataFrame(rows).to_csv(OUT / 'marmoset_ssl_tokens.csv', index=False)
    print(f'\nSaved: {OUT/"marmoset_ssl_tokens.csv"}', flush=True)
