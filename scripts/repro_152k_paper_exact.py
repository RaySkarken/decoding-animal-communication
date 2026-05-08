"""Paper-EXACT reproduction on 153k corpus with min_dist=1.0 UMAP.

This matches the actual settings in Assom's TF_AE.ipynb (their final figure
generator), which differs from paper's Fig 1b/2 caption text:
    paper text:  min_dist=0.3
    actual code: min_dist=1.0     ← gives the well-separated cluster figure

Steps:
  1. Load tf_specs + new UMAP (md=1.0)
  2. HDBSCAN sweep, pick 7-cluster config
  3. KNN noise reassign
  4. Visualization (paper-style)
  5. DTW-MFCC qt_ward proxy → per-emitter ARI/NMI
  6. HP1 with paper-EXACT context-conditioned Zhang features (a-r)
     - SVC + GridSearchCV like Exp1-Classifier.ipynb
     - Random train/test split (paper's protocol, NOT emitter-split)
     - F1_orig vs F1_perm (permutation test)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, joblib, librosa
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan, networkx as nx
from collections import Counter, defaultdict
from itertools import pairwise
import math
from tqdm.auto import tqdm

from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                              normalized_mutual_info_score, f1_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
STATE = CACHE / 'ablation_state_152k.joblib'
EMB_PATH = CACHE / 'umap_152k_nn30_md1.0.npy'
HDB_PATH = CACHE / 'hdb_labels_152k_md10.npy'
NCA_PATH = CACHE / 'hdb_nca_labels_152k_md10.npy'
PROXY_PATH = CACHE / 'proxy_label_152k_md10.npy'

HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CONTEXT_DICT = {0:'Unknown', 1:'Separation', 2:'Biting', 3:'Feeding',
                4:'Fighting', 5:'Grooming', 6:'Isolation', 7:'Kissing',
                8:'Landing', 9:'Mating protest', 10:'Threat-like',
                11:'General', 12:'Sleeping'}

print(f'[1/6] Loading state + new UMAP (min_dist=1.0)...')
st = joblib.load(STATE)
seg_df = st['seg_df']
tf_specs = st['tf_specs']
emb = np.load(EMB_PATH)
print(f'  N: {len(seg_df)}, embedding: {emb.shape}')

# ── HDBSCAN sweep on new embedding ──────────────────────────────────────────
N = len(emb)
print(f'\n[2/6] HDBSCAN sweep for 7-cluster config on md=1.0 embedding...')
best_cfg = None
for frac in [0.005, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.025, 0.030]:
    for ms in [10, 15, 20]:
        for eps in [0.04, 0.05, 0.06, 0.08, 0.1]:
            mcs = max(10, int(N * frac))
            h = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                cluster_selection_epsilon=eps,
                                cluster_selection_method='leaf',
                                metric='euclidean', core_dist_n_jobs=-1).fit(emb)
            lbl = h.labels_
            n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
            if n_cl == 7:
                nn = lbl >= 0
                if nn.sum() < 100: continue
                sil = silhouette_score(emb[nn][:20000], lbl[nn][:20000])
                cfg = {'frac':frac, 'mcs':mcs, 'ms':ms, 'eps':eps,
                       'n_cl':n_cl, 'sil':sil, 'noise':(lbl==-1).mean(),
                       'labels': lbl}
                if best_cfg is None or sil > best_cfg['sil']:
                    best_cfg = cfg
                    print(f'  candidate: frac={frac}, ms={ms}, eps={eps} → sil={sil:.3f}, noise={(lbl==-1).mean():.1%}')

if best_cfg is None:
    print('NO 7-cluster config found — falling back to silhouette-best')
    raise SystemExit(1)
print(f'\nSelected config: frac={best_cfg["frac"]}, ms={best_cfg["ms"]}, eps={best_cfg["eps"]}')
print(f'  → 7 clusters, sil={best_cfg["sil"]:.3f}, noise={best_cfg["noise"]:.1%}')
labels = best_cfg['labels']
np.save(HDB_PATH, labels)

# ── KNN noise reassign ──────────────────────────────────────────────────────
print(f'\n[3/6] KNN(k=30) noise reassign...')
nn_mask = labels >= 0
knn = KNeighborsClassifier(n_neighbors=30, weights='uniform', n_jobs=-1)
knn.fit(emb[nn_mask], labels[nn_mask])
nca = labels.copy()
nca[~nn_mask] = knn.predict(emb[~nn_mask])
np.save(NCA_PATH, nca)
seg_df['syllable_id'] = nca

# ── Visualization ───────────────────────────────────────────────────────────
print(f'\n[4/6] Saving paper-style visualization...')
n_plot = min(80_000, len(emb))
ix = np.random.default_rng(0).choice(len(emb), n_plot, replace=False)
emb_p = emb[ix]
ctx_p = seg_df.iloc[ix]['context'].values

fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
ax.set_facecolor('black')
ctx_unique = sorted([c for c in set(ctx_p.tolist()) if c in HP1_CTX or c in [1, 8]])
palette = sns.color_palette('Spectral', len(ctx_unique))
for c, color in zip(ctx_unique, palette):
    m = ctx_p == c
    ax.scatter(emb_p[m,0], emb_p[m,1], s=2, alpha=0.6,
                c=[color], label=CONTEXT_DICT.get(c, str(c)))
ax.legend(markerscale=4, fontsize=9, loc='upper right',
           facecolor='lightgray', framealpha=0.95)
ax.set_xticks([]); ax.set_yticks([])
ax.text(0.99, 0.01, f'UMAP: n_neighbors=30, min_dist=1.0',
         transform=ax.transAxes, ha='right', va='bottom',
         color='white', fontsize=9)
plt.tight_layout()
plt.savefig('docs/thesis/figures/umap_152k_md10_by_context.png', dpi=160,
             facecolor='black', bbox_inches='tight')
plt.savefig('docs/thesis/figures/umap_152k_md10_by_context.pdf',
             facecolor='black', bbox_inches='tight')
plt.close()
print(f'  saved: docs/thesis/figures/umap_152k_md10_by_context.{{png,pdf}}')

# ── DTW-MFCC qt_ward proxy ──────────────────────────────────────────────────
print(f'\n[5/6] DTW-MFCC qt_ward proxy on identified bats...')
em = seg_df['emitter'].to_numpy()
em_abs = np.abs(em)
N_PER_EM = 400; Q_PROXY = 0.05; N_MFCC = 13

if PROXY_PATH.exists():
    proxy = np.load(PROXY_PATH)
    print(f'  cached → {PROXY_PATH.name}')
else:
    rng = np.random.default_rng(0)
    proxy = np.full(len(seg_df), -1, dtype=np.int32)
    offset = 0
    bats = sorted(set(em_abs[em != 0].tolist()))
    for b in tqdm(bats, desc='bats'):
        ix2 = np.where(em_abs == b)[0]
        if len(ix2) < 5: continue
        if len(ix2) > N_PER_EM:
            ix2 = rng.choice(ix2, size=N_PER_EM, replace=False)
        ix2 = np.sort(ix2)
        mfccs = [librosa.feature.mfcc(S=tf_specs[i].T, n_mfcc=N_MFCC) for i in ix2]
        n = len(mfccs)
        D = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i+1, n):
                D_, wp = librosa.sequence.dtw(X=mfccs[i], Y=mfccs[j], metric='euclidean')
                d = float(D_[-1,-1]) / max(len(wp), 1)
                D[i,j] = d; D[j,i] = d
        D = (D - D.min()) / (D.max() - D.min() + 1e-9)
        Z = linkage(squareform(D, checks=False), method='ward')
        cut = float(np.quantile(cophenet(Z), Q_PROXY))
        lbl_ = fcluster(Z, t=cut, criterion='distance')
        n_types = len(set(lbl_.tolist()))
        proxy[ix2] = lbl_ + offset
        offset += n_types + 1
    np.save(PROXY_PATH, proxy)

# Per-emitter ARI/NMI
rec = []
for b in sorted(set(em_abs[em != 0].tolist())):
    m = (em_abs == b) & (proxy >= 0)
    if m.sum() < 30: continue
    rec.append({'bat': b, 'n': int(m.sum()),
                'ari': adjusted_rand_score(proxy[m], nca[m]),
                'nmi': normalized_mutual_info_score(proxy[m], nca[m])})
p = pd.DataFrame(rec)
print(f'\nPer-emitter ARI: {p.ari.mean():.3f} ± {p.ari.std():.3f}  [paper: 0.12 ± 0.01]')
print(f'Per-emitter NMI: {p.nmi.mean():.3f} ± {p.nmi.std():.3f}  [paper: 0.30 ± 0.01]')


# ── HP1 paper-EXACT (context-conditioned features) ─────────────────────────
print(f'\n[6/6] HP1 with paper-exact context-conditioned features...')

# Build per-file vocalization sequences (using nca syllable IDs)
sequences = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids: continue
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    if dom_ctx not in HP1_CTX: continue
    seq = [int(nca[i]) for i in seg_ids]
    if len(seq) < 2: continue
    sequences.append({'file_name': fname, 'context': dom_ctx, 'seq': seq})
seq_df = pd.DataFrame(sequences)
print(f'Vocalizations in HP1 contexts: {len(seq_df)}')


# Helper functions matching Exp1-Classifier.ipynb
def num_transition_types_in_context(df, c):
    pairs = set()
    for s in df[df.context == c]['seq']:
        pairs.update(pairwise(s))
    return len(pairs) or 1


def prob_syl_by_context(df, c):
    flat = [x for s in df[df.context == c]['seq'] for x in s]
    cnt = Counter(flat); tot = sum(cnt.values())
    return {k: v/tot for k, v in cnt.items()} if tot else {}


def conditional_prob_1(seqs, n=1):
    cond = defaultdict(lambda: defaultdict(int))
    for s in seqs:
        ants = []
        for i in range(n):
            if i < len(s): ants.append(s[i])
        for x in s:
            cond[tuple(ants)][x] += 1
            if len(ants) >= n: ants.pop(0)
            ants.append(x)
    out = {}
    for k, d in cond.items():
        tot = sum(d.values())
        out[k] = {kk: vv/tot for kk, vv in d.items()}
    return out


def transitions_dict(seqs):
    pairs_all = []
    for s in seqs:
        pairs_all += list(pairwise(s))
    cnt = Counter(pairs_all); tot = sum(cnt.values())
    return {k: v/tot for k, v in cnt.items()} if tot else {}


def transition_prob(seqs):
    g = nx.DiGraph()
    for s in seqs:
        for a, b in pairwise(s):
            if g.has_edge(a, b): g[a][b]['weight'] += 1
            else: g.add_edge(a, b, weight=1)
    out = {}
    for src in g.nodes():
        tot = sum(g[src][t]['weight'] for t in g.successors(src))
        if tot: out[src] = {t: g[src][t]['weight']/tot for t in g.successors(src)}
    return out


def make_graph(df):
    g = nx.DiGraph()
    seqs = df['seq']
    n_total = len(seqs.explode().unique())
    for n_, f in seqs.explode().value_counts().to_dict().items():
        g.add_node(n_, frequency=f, p_frequency=f / max(n_total, 1))
    edges = [p for s in seqs for p in pairwise(s)]
    for e, v in Counter(edges).items():
        cur = e[0]; tgt = e[1]
        post = sum(1 for x in edges if x[0] == cur)
        ant = sum(1 for x in edges if x[1] == tgt)
        p_trans = (v / post) if post else 1e-7
        p_cond = (v / ant) if ant else 1e-7
        g.add_edge(*e, frequency=v, p_trans=p_trans, p_cond=p_cond)
    return g


def features_a_r(df, data_df, G, sequences_in_context=True):
    """Reproduces Exp1-Classifier.ipynb prepare_data_from_sequences exactly."""
    cols = list(map(chr, range(97, 97 + 18)))   # a..r
    out = {c: [] for c in cols}
    contexts = sorted(df.context.unique())

    _trans_in_ctx = {c: num_transition_types_in_context(df, c) for c in contexts}
    _prob_syl_ctx = {c: prob_syl_by_context(df, c) for c in contexts}
    _cond_prob_ctx = {c: conditional_prob_1(df[df.context == c]['seq'], 1) for c in contexts}
    _trans_probs_ctx = {c: transition_prob(df[df.context == c]['seq']) for c in contexts}
    _trans_dict_ctx = {c: transitions_dict(df[df.context == c]['seq']) for c in contexts}

    _cond_prob_all = conditional_prob_1(df['seq'], 1)
    _cond_prob_all_2 = conditional_prob_1(df['seq'], 2)
    _trans_dict_all = transitions_dict(df['seq'])
    _trans_probs_all = transition_prob(df['seq'])
    _trans_in_total = max(1, int(df['seq'].apply(lambda x: list(pairwise(x)))
                                 .explode().value_counts().sum()))
    freq_syl = df['seq'].explode().value_counts()
    tot_freq = max(1, int(freq_syl.sum()))
    _prob_syl_all = (freq_syl / tot_freq).to_dict()

    for _, row in data_df.iterrows():
        seq = row['seq']; c = row['context']
        a = len(set(seq)); b = len(seq); c_t = len(list(pairwise(seq)))
        d = a / max(c_t, 1)
        if sequences_in_context and c in _trans_in_ctx:
            e = c_t / max(_trans_in_ctx[c], 1)
            p_ctx = _prob_syl_ctx[c]
            f = -sum(p_ctx.get(s, 1e-9) * np.log2(max(p_ctx.get(s, 1e-9), 1e-9)) for s in seq)
            init_prob = p_ctx
            cond1 = _cond_prob_ctx[c]
            tdict = _trans_dict_ctx[c]
            tprob = _trans_probs_ctx[c]
        else:
            e = c_t / _trans_in_total
            f = -sum(_prob_syl_all.get(s, 1e-9) * np.log2(max(_prob_syl_all.get(s, 1e-9), 1e-9)) for s in seq)
            init_prob = _prob_syl_all
            cond1 = _cond_prob_all
            tdict = _trans_dict_all
            tprob = _trans_probs_all
        g_v = init_prob.get(seq[0], 1e-9)
        for i in range(1, len(seq)):
            ant, cur = seq[i-1], seq[i]
            g_v *= cond1.get((ant,), {}).get(cur, 1e-9)
        h_v = math.prod([tdict.get(p, 1e-9) for p in pairwise(seq)]) if len(seq) > 1 else 0
        i_v = a / max(b, 1)
        j = -sum(tprob.get(p[0], {}).get(p[1], 1e-9) * np.log2(max(tprob.get(p[0], {}).get(p[1], 1e-9), 1e-9)) for p in pairwise(seq))
        probs_k = [G.edges[p]['p_trans'] for p in pairwise(seq) if p in G.edges]
        k = math.prod(probs_k) if probs_k else 0
        probs_l = [G.edges[p]['p_cond'] * np.log2(max(G.edges[p]['p_cond'], 1e-9))
                   for p in pairwise(seq) if p in G.edges]
        l = math.prod(probs_l) if probs_l else 0
        probs_m = [G.edges[p]['p_trans'] * np.log2(max(G.edges[p]['p_trans'], 1e-9))
                   for p in pairwise(seq) if p in G.edges]
        m = math.prod(probs_m) if probs_m else 0
        p_cond_n = [init_prob.get(seq[0], 1e-9)]
        for i2 in range(1, len(seq)):
            ant, cur = seq[i2-1], seq[i2]
            p_cond_n.append(_cond_prob_all.get((ant,), {}).get(cur, 1e-9))
        n_v = math.prod(p_cond_n) if p_cond_n else 0
        p_cond_o = [G.nodes[seq[0]]['p_frequency'] if seq[0] in G.nodes else 1e-9]
        p_trans_p = []
        for i2 in range(1, len(seq)):
            ant, cur = seq[i2-1], seq[i2]
            p_cond_o.append(G[ant][cur]['p_cond'] if (ant in G and cur in G[ant]) else 1e-9)
            if i2 < len(seq) - 1:
                suc = seq[i2+1]
                p_trans_p.append(G[cur][suc]['p_trans'] if (cur in G and suc in G[cur]) else 1e-9)
        o = math.prod(p_cond_o) if p_cond_o else 0
        pv = math.prod(p_trans_p) if p_trans_p else 0
        p_cond_q = [init_prob.get(seq[0], 1e-9)]
        for i2 in range(1, len(seq)):
            ant, cur = seq[i2-1], seq[i2]
            v = _cond_prob_all.get((ant,), {}).get(cur, 1e-9)
            if i2 > 1:
                ant2 = seq[i2-2]
                v *= _cond_prob_all_2.get((ant2, ant), {}).get(cur, 1e-9)
            p_cond_q.append(v)
        q = math.prod(p_cond_q) if p_cond_q else 0
        r = math.pow(math.prod([1 / max(x, 1e-9) for x in p_cond_q]), 1 / max(len(p_cond_q), 1)) if p_cond_q else 0
        for col, val in zip(cols, [a, b, c_t, d, e, f, g_v, h_v, i_v, j, k, l, m, n_v, o, pv, q, r]):
            out[col].append(val)
    return pd.DataFrame(out)


print('Building Zhang-18 features (context-conditioned, paper-exact)...')
G = make_graph(seq_df)
feat = features_a_r(seq_df, seq_df, G, sequences_in_context=True)
feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f'Feature matrix: {feat.shape}')

# Train/test split (paper protocol: random, NOT emitter-split)
y = seq_df['context'].values
le = LabelEncoder(); y_enc = le.fit_transform(y)
X_tr, X_te, y_tr, y_te = train_test_split(feat.values, y_enc,
                                           test_size=0.2, stratify=y_enc, random_state=0)

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

# SVC + GridSearchCV (Exp1-Classifier protocol)
print('SVC + GridSearchCV...')
cv_folds = max(2, min(10, int(np.bincount(y_tr).min())))
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
param_grid = {'C': np.logspace(-2, 3, 6),
              'gamma': np.logspace(-3, 2, 6),
              'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_tr_s, y_tr)
y_pred = grid.best_estimator_.predict(X_te_s)
f1_orig = f1_score(y_te, y_pred, average='weighted')
print(f'  best params: {grid.best_params_}')
print(f'  F1_original = {f1_orig:.3f}    [paper: > 0.9]')

# Permutation: shuffle each sequence, re-extract features (still using
# original context as conditioning), test with same trained model
print('Permutation test...')
rng = np.random.default_rng(0)
seq_perm = seq_df.copy()
seq_perm['seq'] = seq_perm['seq'].apply(lambda s: list(rng.permutation(s)))
feat_perm = features_a_r(seq_df, seq_perm, G, sequences_in_context=True)
feat_perm = feat_perm.replace([np.inf, -np.inf], np.nan).fillna(0)
X_perm_s = sc.transform(feat_perm.values)
y_perm_enc = le.transform(seq_perm['context'].values)
y_perm_pred = grid.best_estimator_.predict(X_perm_s)
f1_perm = f1_score(y_perm_enc, y_perm_pred, average='weighted')
print(f'  F1_permuted = {f1_perm:.3f}    [paper: > 0.9]')
print(f'  |Δ| = {abs(f1_orig - f1_perm):.3f}    [paper: ≈ 0]')

# Save summary
summary = pd.DataFrame([
    {'metric': 'N segments', 'ours': len(seg_df), 'paper': 152578},
    {'metric': 'HDBSCAN n_clusters', 'ours': 7, 'paper': 7},
    {'metric': 'Silhouette', 'ours': round(best_cfg['sil'], 3), 'paper': '> 0.5'},
    {'metric': 'Per-emitter ARI', 'ours': f'{p.ari.mean():.3f}±{p.ari.std():.3f}',
     'paper': '0.12 ± 0.01'},
    {'metric': 'Per-emitter NMI', 'ours': f'{p.nmi.mean():.3f}±{p.nmi.std():.3f}',
     'paper': '0.30 ± 0.01'},
    {'metric': 'HP1 F1_orig (paper-exact features)', 'ours': round(f1_orig, 3), 'paper': '> 0.9'},
    {'metric': 'HP1 F1_perm', 'ours': round(f1_perm, 3), 'paper': '> 0.9'},
    {'metric': 'HP1 |F1_orig - F1_perm|', 'ours': round(abs(f1_orig-f1_perm), 3), 'paper': '≈ 0'},
])
print('\n=== SUMMARY ===')
print(summary.to_string(index=False))
summary.to_csv('docs/thesis/figures/repro_152k_paper_exact_summary.csv', index=False)
print(f'\nSaved: docs/thesis/figures/repro_152k_paper_exact_summary.csv')
