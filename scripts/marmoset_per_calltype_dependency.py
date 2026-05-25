"""Localize the marmoset order effect: which CALL-TYPES carry sequential structure?

Marmoset call-types differ in known acoustic structure: 'twitter' / 'trill-twitter' /
'pheetwitter' are repeated-phrase / combined calls (expected high sequential structure),
while 'phee' is a single tonal sweep (expected low). We compute per-call-type bigram
sequential dependency (real vs shuffled) on the frame tokens. If dependency concentrates
in the structured call-types, the statistical order effect is grounded in call biology.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
MARM = Path('/Volumes/T7/datasets/InfantMarmosetsVox/cache')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
V = 30
NAMES = {0: 'Peep(PrePhee)', 1: 'Phee', 2: 'Twitter', 3: 'Trill', 4: 'Trillphee',
         5: 'TsikTse', 6: 'Egg', 7: 'Pheecry', 8: 'TrllTwitter', 9: 'Pheetwitter', 10: 'Peep'}


def load():
    calls = []
    for f in sorted(MARM.glob('*.npz')):
        if f.name.startswith('._'):
            continue
        z = np.load(f, allow_pickle=True)
        for fr, ct in zip(z['frames'], z['calltype']):
            fr = np.asarray(fr, dtype=np.float32)
            if fr.ndim == 2 and len(fr) >= 2:
                calls.append({'frames': fr, 'calltype': int(ct)})
    return calls


def dependency(seqs, shuffle=False, seed=0):
    rng = np.random.default_rng(seed)
    uni = np.zeros(V) + 1e-9; bi = np.zeros((V, V)) + 1e-9
    for s in seqs:
        s = list(s)
        if shuffle: rng.shuffle(s)
        for t in s: uni[t] += 1
        for a, b in zip(s[:-1], s[1:]): bi[a, b] += 1
    pu = uni / uni.sum(); H1 = -(pu * np.log2(pu)).sum()
    pa = bi.sum(1) / bi.sum(); H2 = 0.0
    for a in range(V):
        pb = bi[a] / bi[a].sum(); H2 += pa[a] * (-(pb * np.log2(pb)).sum())
    return (H1 - H2) / H1


if __name__ == '__main__':
    calls = load()
    allf = np.concatenate([c['frames'] for c in calls])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    km = KMeans(V, n_init=10, random_state=0).fit((allf - mu) / sd)
    for c in calls:
        c['seq'] = km.predict((c['frames'] - mu) / sd).tolist()

    rows = []
    for ct in sorted({c['calltype'] for c in calls}):
        seqs = [c['seq'] for c in calls if c['calltype'] == ct]
        if len(seqs) < 100:
            continue
        dep = dependency(seqs, shuffle=False)
        deps = dependency(seqs, shuffle=True)
        mlen = np.mean([len(s) for s in seqs])
        rows.append({'calltype': ct, 'name': NAMES.get(ct, str(ct)), 'n': len(seqs),
                     'mean_len': round(mlen, 1), 'dependency': round(dep, 3),
                     'shuf_dependency': round(deps, 3), 'dep_gain': round(dep - deps, 3)})
    df = pd.DataFrame(rows).sort_values('dep_gain', ascending=False)
    print('\n=== Per-call-type sequential dependency (marmoset) ===', flush=True)
    print(df.to_string(index=False), flush=True)
    df.to_csv(OUT / 'marmoset_per_calltype_dependency.csv', index=False)
    print('\ndep_gain = real - shuffled dependency: higher = more genuine sequential structure.', flush=True)
    print(f'Saved: {OUT/"marmoset_per_calltype_dependency.csv"}', flush=True)
