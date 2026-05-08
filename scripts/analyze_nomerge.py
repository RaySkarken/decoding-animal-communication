"""
Investigate the 4-token a1-nomerge-mel result:
- What happened to baseline clusters 4 and 5 (which seem to have been pruned)?
- What is the context purity of the 4 remaining tokens?
- Is this a qualitatively better repertoire or a collapse?
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from collections import Counter

CHECKPOINT_DIR = Path('/Volumes/T7/cache/assom_paper_repro')
exp = joblib.load(CHECKPOINT_DIR / 'nomerge_experiment.joblib')
state_in = joblib.load(CHECKPOINT_DIR / 'ablation_state.joblib')

seg_df = state_in['seg_df']
hdbnca_lbl = state_in['hdb_nca_labels']

assom = exp['states']['assom']
a1_nm = exp['states']['a1-nomerge-mel']

CTX_MAP = {0: 'Unknown', 1: 'Separation', 2: 'Biting', 3: 'Feeding',
           4: 'Fighting', 5: 'Grooming', 6: 'Isolation', 7: 'Kissing',
           8: 'Landing', 9: 'MatingPrt', 10: 'ThreatLk', 11: 'General', 12: 'Sleep'}
contexts = seg_df['context'].to_numpy()

print('='*70)
print('a1-nomerge-mel — what actually got kept')
print('='*70)

cnt_b = Counter(int(x) for x in assom.labels if x >= 0)
cnt_a = Counter(int(x) for x in a1_nm.labels if x >= 0)

print('\n--- Baseline (6 tokens) → a1-nomerge-mel (4 tokens): transition matrix ---')
print(f'  {"base":>5} {"size":>6} -> went to:')
for pid in sorted(cnt_b):
    parent_members = np.where(assom.labels == pid)[0]
    child_dist = Counter(int(x) for x in a1_nm.labels[parent_members])
    parts = ', '.join(f'tok{k}:{v}({v/len(parent_members):.0%})'
                       for k, v in sorted(child_dist.items(), key=lambda x: -x[1]))
    print(f'  {pid:5} {len(parent_members):6} -> {parts}')

print('\n--- Final 4 tokens: composition ---')
print(f'  {"tok":>4} {"size":>6}   {"parents (from baseline)":<35}   {"top contexts":<60}')
for tid in sorted(cnt_a):
    members = np.where(a1_nm.labels == tid)[0]
    parent_dist = Counter(int(x) for x in assom.labels[members])
    parent_str = ', '.join(f'base_{k}:{v/len(members):.0%}' for k, v in sorted(parent_dist.items(), key=lambda x: -x[1]))
    ctx_dist = Counter(contexts[members])
    ctx_tops = ctx_dist.most_common(3)
    ctx_str = ', '.join(f'{CTX_MAP.get(int(c), c)}:{n/len(members):.0%}' for c, n in ctx_tops)
    print(f'  {tid:4} {len(members):6}   {parent_str:<35}   {ctx_str}')

# Per-token context entropy
def context_entropy_per_token(labels, contexts):
    out = {}
    for tid in sorted(set(labels)):
        if tid < 0: continue
        members = np.where(labels == tid)[0]
        ctx = contexts[members]
        counts = np.array(list(Counter(ctx).values()))
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        H = -np.sum(probs * np.log2(probs))
        out[tid] = float(H)
    return out

print('\n--- Context entropy per token (lower = more context-specific) ---')
Hb = context_entropy_per_token(assom.labels, contexts)
Ha = context_entropy_per_token(a1_nm.labels, contexts)
print('  Baseline:')
for tid, H in sorted(Hb.items()):
    print(f'    tok{tid}: H = {H:.2f} bits')
print(f'  Mean H = {np.mean(list(Hb.values())):.2f} bits')
print('\n  a1-nomerge-mel:')
for tid, H in sorted(Ha.items()):
    print(f'    tok{tid}: H = {H:.2f} bits')
print(f'  Mean H = {np.mean(list(Ha.values())):.2f} bits')

# "Repertoire usefulness": fraction of segments in context-specific tokens
def context_purity(labels, contexts, threshold=0.5):
    """Fraction of segments in tokens whose top context share >= threshold."""
    pure_segments = 0
    total = 0
    for tid in sorted(set(labels)):
        if tid < 0: continue
        members = np.where(labels == tid)[0]
        if len(members) == 0: continue
        ctx = contexts[members]
        top_share = Counter(ctx).most_common(1)[0][1] / len(ctx)
        if top_share >= threshold:
            pure_segments += len(members)
        total += len(members)
    return pure_segments / total if total else 0

print('\n--- Context purity (fraction of segments in ≥50%-pure tokens) ---')
print(f'  Baseline:        {context_purity(assom.labels, contexts):.1%}')
print(f'  a1-nomerge-mel:  {context_purity(a1_nm.labels, contexts):.1%}')
