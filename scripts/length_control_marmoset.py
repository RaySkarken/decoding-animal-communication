"""Length confound control for the marmoset order effect.

The bat order-null vs marmoset order-help could be a sequence-LENGTH artifact
(bat ~4 sub-units, marmoset ~11). Here we stratify marmoset calls by frame count
and run the order test (call-type) per length band. If order still helps at SHORT
lengths (matching bats, ~2-4 frames), the cross-species difference is not merely
length. Uses cached per-WAV features from marmoset_order.py.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys, time
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seqlib as S

CACHE = Path('/Volumes/T7/datasets/InfantMarmosetsVox/cache')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
V_TOK = 30
N_SEEDS = 3
BANDS = [(2, 4), (5, 8), (9, 48)]   # short (bat-like), mid, long


def load_calls():
    calls = []
    for f in sorted(CACHE.glob('*.npz')):
        if f.name.startswith('._'):
            continue   # skip macOS AppleDouble metadata forks
        z = np.load(f, allow_pickle=True)
        for fr, ct, cl in zip(z['frames'], z['calltype'], z['caller']):
            fr = np.asarray(fr, dtype=np.float32)
            if fr.ndim == 2 and len(fr) >= 2:
                calls.append({'frames': fr, 'calltype': int(ct), 'caller': int(cl)})
    return calls


def split_strat(vocs, key, seed, frac=0.2):
    rng = np.random.default_rng(seed); by = {}
    for i, v in enumerate(vocs): by.setdefault(v[key], []).append(i)
    tr, te = [], []
    for k, idxs in by.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        n = max(1, int(len(idxs) * frac)); te += idxs[:n].tolist(); tr += idxs[n:].tolist()
    return [vocs[i] for i in tr], [vocs[i] for i in te]


if __name__ == '__main__':
    calls = load_calls()
    print(f'cached calls: {len(calls):,}', flush=True)
    if len(calls) < 1000:
        print('not enough cached calls', flush=True); sys.exit(0)
    allf = np.concatenate([c['frames'] for c in calls])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    km = KMeans(V_TOK, n_init=10, random_state=0).fit((allf - mu) / sd)
    for c in calls:
        c['seq'] = km.predict((c['frames'] - mu) / sd).tolist()

    rows = []
    for lo, hi in BANDS:
        band = [c for c in calls if lo <= len(c['seq']) <= hi]
        # need >=2 classes with enough samples
        labs = sorted({c['calltype'] for c in band})
        labs = [l for l in labs if sum(c['calltype'] == l for c in band) >= 30]
        band = [c for c in band if c['calltype'] in labs]
        if len(band) < 500 or len(labs) < 2:
            print(f'band [{lo},{hi}]: too few ({len(band)} calls, {len(labs)} classes) — skip', flush=True)
            continue
        y2i = {l: i for i, l in enumerate(labs)}
        meanlen = np.mean([len(c['seq']) for c in band])
        print(f'\n=== band frames[{lo},{hi}]: {len(band)} calls, {len(labs)} calltypes, mean_len={meanlen:.1f} ===', flush=True)
        for seed in range(N_SEEDS):
            t0 = time.time()
            tr, te = split_strat(band, 'calltype', seed)
            rm = S.train_eval_bert(tr, te, V_TOK, y2i, 'calltype', seed, shuf=False)
            sm = S.train_eval_bert(tr, te, V_TOK, y2i, 'calltype', seed, shuf=True)
            print(f'  seed {seed}: real={rm:.3f} shuf={sm:.3f} Δ={rm-sm:+.3f} ({time.time()-t0:.0f}s)', flush=True)
            rows.append({'band': f'{lo}-{hi}', 'mean_len': round(meanlen, 1), 'seed': seed,
                         'real_macro': rm, 'shuf_macro': sm})
        pd.DataFrame(rows).to_csv(OUT / 'length_control_marmoset.csv', index=False)

    df = pd.DataFrame(rows)
    print('\n=== Does order help per length band? (call-type) ===', flush=True)
    for band in df.band.unique():
        sub = df[df.band == band]
        m, lo, hi = S.ci95((sub.real_macro - sub.shuf_macro).tolist())
        print(f'band {band} (len~{sub.mean_len.iloc[0]}): Δ={m:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] '
              f'{"ORDER HELPS" if lo>0 else "n.s."}', flush=True)
    print(f'Saved: {OUT/"length_control_marmoset.csv"}', flush=True)
