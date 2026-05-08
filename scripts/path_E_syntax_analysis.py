"""Path E: extend Assom's HP2/HP3/network analyses to per-context vocabularies.

For each of 8 working contexts, two parallel analyses:
  - on GLOBAL vocab (HDBSCAN-NCA, |V|=11) — paper-style
  - on PER-CONTEXT vocab (DP-GMM with oracle, |V|≈109) — our extension

Compute:
  HP2 — Wilcoxon pairwise context distinguishability + JS divergence between context distributions
  HP3 — Maximal Repeats: power-law vs exponential α, χ², Hilberg slope
  Networks — small-world ω*, average clustering, density per context
  Vocab structure — entropy per context, type/token ratio per context

Output: docs/thesis/figures/path_E_syntax_analysis.csv + summary tables
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, networkx as nx, math
from collections import Counter, defaultdict
from itertools import pairwise, combinations
from pathlib import Path
from scipy.stats import ranksums
from sklearn.mixture import BayesianGaussianMixture

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
hdb_nca = np.load(CACHE / 'hdb_nca_labels_152k_21x32.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
HP1_CTX = [2,3,4,5,6,7,9,10]
CTX = {2:'Biting',3:'Feeding',4:'Fighting',5:'Grooming',6:'Isolation',7:'Kissing',9:'Mating',10:'Threat'}

# ── Build per-context syllable labels (oracle, full data) ──────────────────
print('Fitting per-context DP-GMM on full data (oracle context)...')
labels_pc = np.full(len(seg_df), -1, dtype=np.int32)
offset = 0
for c in HP1_CTX:
    mc = ctx == c
    if mc.sum() < 30: continue
    bgm = BayesianGaussianMixture(n_components=15,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1, covariance_type='full',
        max_iter=200, random_state=0).fit(emb[mc])
    labels_pc[mc] = bgm.predict(emb[mc]) + offset
    offset += bgm.n_components
print(f'  per-context vocab: {len(set(labels_pc[labels_pc>=0].tolist()))}')

# ── Build sequences per file (only HP1 contexts, identified bats) ─────────
print('\nBuilding sequences per file...')
def build_sequences(label_array):
    seqs = []
    for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
        seg_ids = g.index.to_list()
        if not seg_ids: continue
        dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
        if dom_em == 0: continue
        dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
        if dom_ctx not in HP1_CTX: continue
        seq = [int(label_array[i]) for i in seg_ids if label_array[i] >= 0]
        if len(seq) < 2: continue
        seqs.append({'file_name': fname, 'context': dom_ctx, 'seq': seq})
    return pd.DataFrame(seqs)

seq_global = build_sequences(hdb_nca)
seq_pc = build_sequences(labels_pc)
print(f'  global: {len(seq_global)} seqs, vocab={len(set(s for seq in seq_global.seq for s in seq))}')
print(f'  per-context: {len(seq_pc)} seqs, vocab={len(set(s for seq in seq_pc.seq for s in seq))}')


# ════════════════════════════════════════════════════════════════════════
# HP2: Wilcoxon rank-sum on syllable usage between context pairs
# ════════════════════════════════════════════════════════════════════════
def hp2_pairwise_wilcoxon(seq_df_, label_set):
    contexts = sorted(seq_df_.context.unique())
    profiles = {}
    for c in contexts:
        flat = [s for seq in seq_df_[seq_df_.context==c]['seq'] for s in seq]
        cnt = Counter(flat); tot = sum(cnt.values())
        profiles[c] = np.array([cnt.get(s, 0)/max(tot,1) for s in label_set])
    pairs = list(combinations(contexts, 2))
    ps = []
    sig_at_05 = 0; sig_at_001 = 0
    for c1, c2 in pairs:
        _, p = ranksums(profiles[c1], profiles[c2])
        ps.append({'c1':CTX.get(c1,c1), 'c2':CTX.get(c2,c2), 'p':p})
        if p < 0.05: sig_at_05 += 1
        if p < 0.001: sig_at_001 += 1
    return pd.DataFrame(ps), sig_at_05, sig_at_001


def js_divergence(p, q):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def hp2_pairwise_js(seq_df_, label_set):
    contexts = sorted(seq_df_.context.unique())
    profiles = {}
    for c in contexts:
        flat = [s for seq in seq_df_[seq_df_.context==c]['seq'] for s in seq]
        cnt = Counter(flat); tot = sum(cnt.values())
        profiles[c] = np.array([cnt.get(s, 0)/max(tot,1) for s in label_set])
    rows = []
    for c1, c2 in combinations(contexts, 2):
        rows.append({'c1':CTX.get(c1,c1), 'c2':CTX.get(c2,c2),
                     'JS_bits': js_divergence(profiles[c1], profiles[c2])})
    return pd.DataFrame(rows)


print('\n══════ HP2 — Wilcoxon + JS divergence on syllable usage per context ══════')
vocab_g = sorted(set(s for seq in seq_global.seq for s in seq))
vocab_p = sorted(set(s for seq in seq_pc.seq for s in seq))

wilc_g, sig05_g, sig001_g = hp2_pairwise_wilcoxon(seq_global, vocab_g)
wilc_p, sig05_p, sig001_p = hp2_pairwise_wilcoxon(seq_pc, vocab_p)
print(f'GLOBAL vocab (|V|={len(vocab_g)}): {sig05_g}/{len(wilc_g)} pairs sig at p<0.05, {sig001_g} at p<0.001')
print(f'PER-CONTEXT (|V|={len(vocab_p)}): {sig05_p}/{len(wilc_p)} pairs sig at p<0.05, {sig001_p} at p<0.001')

js_g = hp2_pairwise_js(seq_global, vocab_g)
js_p = hp2_pairwise_js(seq_pc, vocab_p)
print(f'\nMean pairwise JS divergence (bits):')
print(f'  GLOBAL:      {js_g.JS_bits.mean():.3f} ± {js_g.JS_bits.std():.3f}')
print(f'  PER-CONTEXT: {js_p.JS_bits.mean():.3f} ± {js_p.JS_bits.std():.3f}')


# ════════════════════════════════════════════════════════════════════════
# HP3: Maximal Repeats — power-law vs exponential
# ════════════════════════════════════════════════════════════════════════
def extract_maximal_repeats(sequences, min_count=2, min_len=2, max_len=10):
    """Find subsequences that occur at least min_count times across sequences,
    and are MAXIMAL (cannot be extended without losing a count)."""
    occurrences = defaultdict(int)
    for seq in sequences:
        for L in range(min_len, min(max_len, len(seq))+1):
            for i in range(len(seq) - L + 1):
                tup = tuple(seq[i:i+L])
                occurrences[tup] += 1
    # filter: only those with count >= min_count
    candidates = {k: v for k, v in occurrences.items() if v >= min_count}
    # check maximality: if seq+x has same count, then seq is not maximal
    maximal = {}
    for tup, count in candidates.items():
        is_maximal = True
        # check left extensions
        for prefix_left in candidates:
            if len(prefix_left) == len(tup) + 1 and prefix_left[1:] == tup and candidates[prefix_left] == count:
                is_maximal = False; break
        if is_maximal:
            for prefix_right in candidates:
                if len(prefix_right) == len(tup) + 1 and prefix_right[:-1] == tup and candidates[prefix_right] == count:
                    is_maximal = False; break
        if is_maximal:
            maximal[tup] = count
    return maximal


print('\n══════ HP3 — Maximal Repeats heavy-tail ══════')
mr_g = extract_maximal_repeats(seq_global['seq'].tolist(), min_count=3, min_len=2, max_len=8)
mr_p = extract_maximal_repeats(seq_pc['seq'].tolist(), min_count=3, min_len=2, max_len=8)
print(f'GLOBAL:      {len(mr_g)} maximal repeats found')
print(f'PER-CONTEXT: {len(mr_p)} maximal repeats found')

mr_lengths_g = [len(t) for t in mr_g]
mr_lengths_p = [len(t) for t in mr_p]
mr_counts_g = list(mr_g.values())
mr_counts_p = list(mr_p.values())

print(f'\nMR length distribution:')
print(f'  GLOBAL:      mean={np.mean(mr_lengths_g):.2f}, max={max(mr_lengths_g)}')
print(f'  PER-CONTEXT: mean={np.mean(mr_lengths_p):.2f}, max={max(mr_lengths_p)}')

try:
    import powerlaw
    fit_g = powerlaw.Fit(mr_counts_g, discrete=True, verbose=False)
    fit_p = powerlaw.Fit(mr_counts_p, discrete=True, verbose=False)
    R_g, p_g = fit_g.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    R_p, p_p = fit_p.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    print(f'\nPower-law vs exponential (likelihood ratio test on MR counts):')
    print(f'  GLOBAL:      α={fit_g.alpha:.2f}, R={R_g:.2f}, p={p_g:.4g} [paper: α=1.79]')
    print(f'  PER-CONTEXT: α={fit_p.alpha:.2f}, R={R_p:.2f}, p={p_p:.4g}')
except ImportError:
    print('  (install powerlaw for fit comparison)')


# ════════════════════════════════════════════════════════════════════════
# Network metrics: per-context transition graphs
# ════════════════════════════════════════════════════════════════════════
def build_ctx_graph(seqs):
    g = nx.DiGraph()
    for s in seqs:
        for a, b in pairwise(s):
            if g.has_edge(a, b): g[a][b]['weight'] += 1
            else: g.add_edge(a, b, weight=1)
    return g

def smallworld_omega(g, seed=0):
    gu = g.to_undirected()
    if gu.number_of_nodes() < 4 or gu.number_of_edges() < 2:
        return np.nan, np.nan
    c = nx.average_clustering(gu)
    er = nx.erdos_renyi_graph(gu.number_of_nodes(), nx.density(gu), seed=seed)
    c_rand = nx.average_clustering(er) if er.number_of_edges() else np.nan
    omega = (c_rand / c) if (c > 0 and not np.isnan(c_rand)) else np.nan
    return c, omega


print('\n══════ Network metrics — small-world ω* per context ══════')
print(f'{"Context":12s}  {"GLOBAL ω*":>11s} {"PC ω*":>10s} {"GLOBAL avgC":>13s} {"PC avgC":>10s}')
net_rows = []
for c in HP1_CTX:
    seqs_g = seq_global[seq_global.context==c]['seq'].tolist()
    seqs_p = seq_pc[seq_pc.context==c]['seq'].tolist()
    g_g = build_ctx_graph(seqs_g)
    g_p = build_ctx_graph(seqs_p)
    avgC_g, omega_g = smallworld_omega(g_g)
    avgC_p, omega_p = smallworld_omega(g_p)
    print(f'{CTX[c]:12s}  {omega_g:>11.3f} {omega_p:>10.3f} {avgC_g:>13.3f} {avgC_p:>10.3f}')
    net_rows.append({'context':CTX[c], 'global_omega':omega_g, 'pc_omega':omega_p,
                     'global_avgC':avgC_g, 'pc_avgC':avgC_p,
                     'global_nodes':g_g.number_of_nodes(), 'pc_nodes':g_p.number_of_nodes(),
                     'global_edges':g_g.number_of_edges(), 'pc_edges':g_p.number_of_edges()})


# ════════════════════════════════════════════════════════════════════════
# Per-context entropy + type/token ratio
# ════════════════════════════════════════════════════════════════════════
print('\n══════ Per-context entropy + repertoire stats ══════')
print(f'{"Context":12s} {"H_global":>9s} {"H_pc":>7s} {"|V|_g":>6s} {"|V|_pc":>7s} {"TTR_g":>7s} {"TTR_pc":>7s}')
ent_rows = []
for c in HP1_CTX:
    flat_g = [s for seq in seq_global[seq_global.context==c]['seq'] for s in seq]
    flat_p = [s for seq in seq_pc[seq_pc.context==c]['seq'] for s in seq]
    cnt_g = Counter(flat_g); cnt_p = Counter(flat_p)
    p_g = np.array(list(cnt_g.values()))/max(sum(cnt_g.values()),1)
    p_p = np.array(list(cnt_p.values()))/max(sum(cnt_p.values()),1)
    H_g = -float(np.sum(p_g[p_g>0] * np.log2(p_g[p_g>0])))
    H_p = -float(np.sum(p_p[p_p>0] * np.log2(p_p[p_p>0])))
    Vg = len(cnt_g); Vp = len(cnt_p)
    TTR_g = Vg / max(len(flat_g),1); TTR_p = Vp / max(len(flat_p),1)
    print(f'{CTX[c]:12s} {H_g:>9.3f} {H_p:>7.3f} {Vg:>6d} {Vp:>7d} {TTR_g:>7.4f} {TTR_p:>7.4f}')
    ent_rows.append({'context':CTX[c],'H_global':H_g,'H_pc':H_p,'V_global':Vg,'V_pc':Vp,
                     'TTR_global':TTR_g,'TTR_pc':TTR_p,'n_tokens':len(flat_p)})


# Save all results
out_dir = Path('docs/thesis/figures'); out_dir.mkdir(exist_ok=True, parents=True)
wilc_g.to_csv(out_dir / 'pathE_HP2_wilcoxon_global.csv', index=False)
wilc_p.to_csv(out_dir / 'pathE_HP2_wilcoxon_percontext.csv', index=False)
js_g.to_csv(out_dir / 'pathE_HP2_JS_global.csv', index=False)
js_p.to_csv(out_dir / 'pathE_HP2_JS_percontext.csv', index=False)
pd.DataFrame(net_rows).to_csv(out_dir / 'pathE_network_metrics.csv', index=False)
pd.DataFrame(ent_rows).to_csv(out_dir / 'pathE_entropy_per_context.csv', index=False)


# ════════════════════════════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════════════════════════════
print('\n' + '═'*78)
print('  SUMMARY — Path E: per-context vocabularies enable richer syntax analysis')
print('═'*78)
print(f'{"Analysis":40s} {"Global":>15s} {"Per-context":>15s}')
print('-'*78)
print(f'{"HP2 sig pairs (p<0.05) of 28":40s} {sig05_g:>15d} {sig05_p:>15d}')
print(f'{"HP2 sig pairs (p<0.001) of 28":40s} {sig001_g:>15d} {sig001_p:>15d}')
print(f'{"Mean pairwise JS divergence (bits)":40s} {js_g.JS_bits.mean():>15.3f} {js_p.JS_bits.mean():>15.3f}')
print(f'{"HP3 maximal repeats found":40s} {len(mr_g):>15d} {len(mr_p):>15d}')
print(f'{"HP3 max MR length":40s} {max(mr_lengths_g):>15d} {max(mr_lengths_p):>15d}')
try:
    print(f'{"HP3 power-law alpha":40s} {fit_g.alpha:>15.2f} {fit_p.alpha:>15.2f}')
except: pass
print(f'{"Mean per-context entropy (bits)":40s} {np.mean([r["H_global"] for r in ent_rows]):>15.3f} {np.mean([r["H_pc"] for r in ent_rows]):>15.3f}')
print(f'{"Mean per-context vocab |V_c|":40s} {np.mean([r["V_global"] for r in ent_rows]):>15.1f} {np.mean([r["V_pc"] for r in ent_rows]):>15.1f}')
print('═'*78)
