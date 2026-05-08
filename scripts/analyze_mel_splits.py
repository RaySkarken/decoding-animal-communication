"""
Analyze what A1-on-Mel actually did:
1. Which baseline clusters got split?
2. How does token-size distribution compare between baseline and A1-on-Mel?
3. Is the catch-all Cluster 5 (20883 points, silh −0.044) split into meaningful pieces?
4. Why does F1 bos crash from 0.469 to 0.217?
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
exp = joblib.load(CHECKPOINT_DIR / 'mel_space_experiment.joblib')
state_in = joblib.load(CHECKPOINT_DIR / 'ablation_state.joblib')

seg_df = state_in['seg_df']
hdbnca_lbl = state_in['hdb_nca_labels']
tf_specs = state_in['tf_specs']
mel_flat = tf_specs.reshape(len(tf_specs), -1).astype(np.float32)

assom = exp['states']['assom']
a1_mel = exp['states']['a1-mel']

print('='*70)
print('ANALYSIS — What did A1 on Mel actually do?')
print('='*70)

print('\n--- Baseline Assom labels (6 tokens) ---')
cnt_b = Counter(assom.labels)
for tid, n in sorted(cnt_b.items(), key=lambda x: -x[1]):
    if tid < 0: continue
    pct = n / len(assom.labels) * 100
    print(f'  token {tid}: {n:>6} segments ({pct:5.1f}%)')

print('\n--- A1-on-Mel labels (10 tokens) ---')
cnt_a = Counter(a1_mel.labels)
for tid, n in sorted(cnt_a.items(), key=lambda x: -x[1]):
    if tid < 0: continue
    pct = n / len(a1_mel.labels) * 100
    print(f'  token {tid}: {n:>6} segments ({pct:5.1f}%)')

print('\n--- Mapping: which baseline cluster each A1-on-Mel cluster came from ---')
# For each A1-on-Mel cluster, find the majority baseline cluster of its members
print(f'  {"A1 tok":>7} {"size":>6} {"majority parent":>16} {"purity":>7}')
for tid in sorted(set(a1_mel.labels)):
    if tid < 0: continue
    members = np.where(a1_mel.labels == tid)[0]
    parent_counts = Counter(assom.labels[members])
    top_parent, top_n = parent_counts.most_common(1)[0]
    purity = top_n / len(members)
    print(f'  {tid:7} {len(members):6}  {top_parent:16}  {purity:6.1%}')

print('\n--- Reverse: how each baseline cluster got split by A1 ---')
print(f'  {"parent":>7} {"size":>6} -> split into:')
for pid in sorted(set(assom.labels)):
    if pid < 0: continue
    parent_members = np.where(assom.labels == pid)[0]
    child_dist = Counter(a1_mel.labels[parent_members])
    main_children = sorted(child_dist.items(), key=lambda x: -x[1])
    share_str = ', '.join(f'tok{k}:{v}' for k,v in main_children)
    print(f'  {pid:7} {len(parent_members):6} -> {share_str}')

# Context distribution per token — does A1-on-Mel give more context-pure clusters?
print('\n--- Context usage by A1-on-Mel cluster (top context share) ---')
ctx_map = {0: 'Unknown', 1: 'Separation', 2: 'Biting', 3: 'Feeding',
           4: 'Fighting', 5: 'Grooming', 6: 'Isolation', 7: 'Kissing',
           8: 'Landing', 9: 'MatingPrt', 10: 'ThreatLk', 11: 'General', 12: 'Sleep'}
contexts = seg_df['context'].to_numpy()

print('\n  A1 tok | top 3 contexts (share)')
for tid in sorted(set(a1_mel.labels)):
    if tid < 0: continue
    members = np.where(a1_mel.labels == tid)[0]
    ctx_counts = Counter(contexts[members])
    tops = ctx_counts.most_common(3)
    s = ', '.join(f'{ctx_map.get(int(c), c)}:{n/len(members):.0%}' for c, n in tops)
    print(f'  {tid:6} | {s}')

print('\n  Baseline Assom token | top 3 contexts (share)')
for tid in sorted(set(assom.labels)):
    if tid < 0: continue
    members = np.where(assom.labels == tid)[0]
    ctx_counts = Counter(contexts[members])
    tops = ctx_counts.most_common(3)
    s = ', '.join(f'{ctx_map.get(int(c), c)}:{n/len(members):.0%}' for c, n in tops)
    print(f'  {tid:6} | {s}')

# Distribution balance (Gini-like)
def entropy_norm(counter):
    n = sum(counter.values())
    if n == 0: return 0
    probs = np.array([v/n for v in counter.values() if v > 0])
    H = -np.sum(probs * np.log2(probs))
    return H / np.log2(len(probs)) if len(probs) > 1 else 0

cnt_b_nonzero = {k: v for k, v in cnt_b.items() if k >= 0}
cnt_a_nonzero = {k: v for k, v in cnt_a.items() if k >= 0}
print('\n--- Distribution balance (normalized entropy, 1 = uniform) ---')
print(f'  Baseline:  H_norm = {entropy_norm(cnt_b_nonzero):.3f}')
print(f'  A1-on-Mel: H_norm = {entropy_norm(cnt_a_nonzero):.3f}')
