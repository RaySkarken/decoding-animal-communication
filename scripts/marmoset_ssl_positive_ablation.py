"""Mechanistic: does the SSL POSITIVE-PAIR definition causally determine which task
benefits? (turns the task-dependence observation into a controlled design principle)

On marmosets we train the same frame-level contrastive SSL with three positive-pair
strategies and compare call-type vs caller macro-F1 (bag-LR on resulting tokens):
  - same_call : positives = two frames from the SAME call (vocalization-invariant)
  - augment   : positives = two augmentations of the SAME frame (frame-preserving)
  - adjacent  : positives = temporally adjacent frames within a call (local)

Hypothesis: invariance shapes the tokens. 'same_call' should favor CALLER (identity is
constant across a call) and hurt CALL-TYPE (erases within-call structure); 'augment'
should preserve within-call detail -> recover CALL-TYPE. A clean shift across
strategies = causal evidence that the positive-pair choice picks the downstream winner.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marmoset_ssl_tokens import (load_calls, Enc, nt_xent, embed, split_strat,
                                  bag_lr, ci95, DEVICE)

OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
V_TOK = 30
N_SEEDS = 5
STRATEGIES = ['same_call', 'augment', 'adjacent']


def augment_frame(x, rng):
    x = x + rng.normal(0, 0.1, size=x.shape).astype(np.float32)   # jitter
    x = x * rng.uniform(0.9, 1.1)                                  # scaling
    return x


def sample_pair(call_frames_norm, strategy, rng):
    """Return (a, b) frame vectors for a positive pair from one call's frames (norm)."""
    n = len(call_frames_norm)
    if strategy == 'same_call':
        i, j = rng.integers(n), rng.integers(n)
        return call_frames_norm[i], call_frames_norm[j]
    if strategy == 'augment':
        i = rng.integers(n)
        f = call_frames_norm[i]
        return augment_frame(f, rng), augment_frame(f, rng)
    if strategy == 'adjacent':
        i = rng.integers(n - 1) if n > 1 else 0
        j = min(i + 1, n - 1)
        return call_frames_norm[i], call_frames_norm[j]
    raise ValueError(strategy)


def train_ssl(calls_norm, strategy, epochs=15, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    enc = Enc().to(DEVICE)
    proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)).to(DEVICE)
    pool = [c for c in calls_norm if len(c) >= 2]
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed); idx = np.arange(len(pool)); t0 = time.time()
    for ep in range(epochs):
        rng.shuffle(idx); tot = 0.0; nb = 0
        for s in range(0, len(idx) - 256, 256):
            batch = idx[s:s + 256]
            pairs = [sample_pair(pool[i], strategy, rng) for i in batch]
            a = np.stack([p[0] for p in pairs]); b = np.stack([p[1] for p in pairs])
            xa = torch.from_numpy(a).float().to(DEVICE); xb = torch.from_numpy(b).float().to(DEVICE)
            loss = nt_xent(proj(enc(xa)), proj(enc(xb)))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f'[{strategy}] ep{ep+1}: {tot/max(nb,1):.4f} ({time.time()-t0:.0f}s)', flush=True)
    return enc


if __name__ == '__main__':
    calls = load_calls()
    print(f'cached calls: {len(calls):,}', flush=True)
    allf = np.concatenate([c['frames'] for c in calls])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    calls_norm = [((c['frames'] - mu) / sd).astype(np.float32) for c in calls]

    rows = []
    # mel baseline tokens (reference)
    km_mel = KMeans(V_TOK, n_init=10, random_state=0).fit((allf - mu) / sd)
    off = 0
    for c in calls:
        n = len(c['frames']); c['mel_seq'] = km_mel.predict(((c['frames'] - mu) / sd)).tolist(); off += n

    variants = {'mel': None}
    for strat in STRATEGIES:
        enc = train_ssl(calls_norm, strat)
        Z = embed(enc, allf, mu, sd)
        km = KMeans(V_TOK, n_init=10, random_state=0).fit(Z)
        off = 0
        for c in calls:
            n = len(c['frames']); c[f'{strat}_seq'] = km.predict(Z[off:off + n]).tolist(); off += n
        variants[strat] = True

    for task in ['calltype', 'caller']:
        labs = sorted({c[task] for c in calls}); y2i = {l: i for i, l in enumerate(labs)}
        for src in ['mel'] + STRATEGIES:
            for c in calls: c['seq'] = c[f'{src}_seq']
            f1s = [bag_lr(*split_strat(calls, task, seed), V_TOK, y2i, task) for seed in range(N_SEEDS)]
            m, lo, hi = ci95(f1s)
            print(f'{task:9s} {src:10s}: macroF1={m:.3f} 95%CI[{lo:.3f},{hi:.3f}]', flush=True)
            rows.append({'task': task, 'positive_strategy': src, 'macro_f1': round(m, 4),
                         'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4)})
        pd.DataFrame(rows).to_csv(OUT / 'marmoset_ssl_positive_ablation.csv', index=False)

    df = pd.DataFrame(rows)
    print('\n=== Positive-pair strategy x task (macro F1) ===', flush=True)
    piv = df.pivot(index='positive_strategy', columns='task', values='macro_f1')
    print(piv.to_string(), flush=True)
    print(f'\nSaved: {OUT/"marmoset_ssl_positive_ablation.csv"}', flush=True)
