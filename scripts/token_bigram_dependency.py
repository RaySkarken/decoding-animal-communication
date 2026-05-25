"""Mechanism: do marmoset token sequences have stronger SEQUENTIAL DEPENDENCY than bats?

After the diversity hypothesis failed, we measure the right thing: how much the previous
token reduces uncertainty about the next. Per species (frame-level tokens, V=30):
  H1   = unigram entropy H(t)
  H2   = bigram conditional entropy H(t_{i+1} | t_i)
  dep  = (H1 - H2) / H1   in [0,1]   (fraction of uncertainty removed by 1 token of context)
We also report dep on SHUFFLED sequences (should -> 0). Higher dep for marmosets would
mechanistically explain why order carries information there but not for bats.
All entropies in bits; bigram counts with add-1 smoothing.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_diversity_analysis import bat_segment_seqs, bat_frame_seqs, marmoset_frame_seqs

OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
V = 30


def entropies(seqs, shuffle=False, seed=0):
    rng = np.random.default_rng(seed)
    uni = np.zeros(V) + 1e-9
    bi = np.zeros((V, V)) + 1e-9
    for s in seqs:
        s = list(s)
        if shuffle:
            rng.shuffle(s)
        for t in s:
            if 0 <= t < V: uni[t] += 1
        for a, b in zip(s[:-1], s[1:]):
            if 0 <= a < V and 0 <= b < V: bi[a, b] += 1
    pu = uni / uni.sum()
    H1 = -(pu * np.log2(pu)).sum()
    # H(next|prev) = sum_a p(a) * H(next|prev=a)
    pa = bi.sum(1) / bi.sum()
    H2 = 0.0
    for a in range(V):
        pb = bi[a] / bi[a].sum()
        H2 += pa[a] * (-(pb * np.log2(pb)).sum())
    dep = (H1 - H2) / H1
    return H1, H2, dep


if __name__ == '__main__':
    bseg, st, seg = bat_segment_seqs()
    datasets = {
        'bat_segment': bseg,
        'bat_frame': bat_frame_seqs(st, seg),
        'marmoset_frame': marmoset_frame_seqs(),
    }
    rows = []
    for name, seqs in datasets.items():
        seqs = [s for s in seqs if len(s) >= 2]
        H1, H2, dep = entropies(seqs, shuffle=False)
        _, H2s, deps = entropies(seqs, shuffle=True)
        print(f'{name:15s}: H1={H1:.3f} H2={H2:.3f} dep={dep:.3f} | shuf dep={deps:.3f} '
              f'(n_seq={len(seqs)})', flush=True)
        rows.append({'dataset': name, 'H1_unigram': round(H1, 3), 'H2_bigram_cond': round(H2, 3),
                     'seq_dependency': round(dep, 3), 'shuf_dependency': round(deps, 3),
                     'n_seq': len(seqs)})
    pd.DataFrame(rows).to_csv(OUT / 'token_bigram_dependency.csv', index=False)
    print('\n=== Sequential dependency (fraction of next-token uncertainty removed by prev) ===', flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print('\nHigher real dep + larger drop vs shuffled = more sequential structure.', flush=True)
    print(f'Saved: {OUT/"token_bigram_dependency.csv"}', flush=True)
