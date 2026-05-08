"""HP1 on bat 215 with **qt_ward proxy** labels (Assom's default alphabethType='qt_ward')."""
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, joblib, networkx as nx, math
from collections import Counter, defaultdict
from itertools import pairwise
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df']
proxy = np.load(CACHE / 'proxy_label_152k_21x32.npy')
# IMPORTANT: use qt_ward (DTW proxy) labels — Assom's default alphabethType='qt_ward'
seg_df['syllable_id'] = proxy

em_abs = np.abs(seg_df['emitter'].to_numpy())
sub = seg_df[em_abs == 215].copy()
# only keep segments with proxy label assigned
sub = sub[sub['syllable_id'] >= 0]
print(f'bat 215 proxy-labeled segments: {len(sub)}, vocabulary: {len(set(sub.syllable_id.tolist()))}')

HP1_CTX = [2,3,4,5,6,7,9,10]
sequences = []
for fname, g in sub.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids: continue
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    if dom_ctx not in HP1_CTX: continue
    seq = [int(proxy[i]) for i in seg_ids if proxy[i] >= 0]
    if len(seq) < 2: continue
    sequences.append({'context':dom_ctx, 'seq':seq})
seq_df = pd.DataFrame(sequences)
print(f'bat 215 vocs: {len(seq_df)}')

def num_trans_in_ctx(df, c):
    pairs = set()
    for s in df[df.context == c]['seq']: pairs.update(pairwise(s))
    return len(pairs) or 1
def prob_syl_ctx(df, c):
    flat = [x for s in df[df.context==c]['seq'] for x in s]
    cnt = Counter(flat); tot = sum(cnt.values())
    return {k: v/tot for k, v in cnt.items()} if tot else {}
def cond_prob_1(seqs, n=1):
    cond = defaultdict(lambda: defaultdict(int))
    for s in seqs:
        ants = []
        for i in range(n):
            if i < len(s): ants.append(s[i])
        for x in s:
            cond[tuple(ants)][x] += 1
            if len(ants) >= n: ants.pop(0)
            ants.append(x)
    return {k: {kk: vv/sum(d.values()) for kk, vv in d.items()} for k, d in cond.items()}
def trans_dict(seqs):
    pairs_all = [p for s in seqs for p in pairwise(s)]
    cnt = Counter(pairs_all); tot = sum(cnt.values())
    return {k: v/tot for k, v in cnt.items()} if tot else {}
def trans_prob(seqs):
    g = nx.DiGraph()
    for s in seqs:
        for a,b in pairwise(s):
            if g.has_edge(a,b): g[a][b]['weight'] += 1
            else: g.add_edge(a,b,weight=1)
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
        g.add_node(n_, frequency=f, p_frequency=f/max(n_total,1))
    edges = [p for s in seqs for p in pairwise(s)]
    for e, v in Counter(edges).items():
        cur, tgt = e
        post = sum(1 for x in edges if x[0]==cur)
        ant = sum(1 for x in edges if x[1]==tgt)
        g.add_edge(*e, frequency=v,
                   p_trans=(v/post) if post else 1e-7,
                   p_cond=(v/ant) if ant else 1e-7)
    return g
def features_a_r(df, data_df, G):
    cols = list(map(chr, range(97, 97+18)))
    out = {c: [] for c in cols}
    contexts = sorted(df.context.unique())
    _trans_in_ctx = {c: num_trans_in_ctx(df, c) for c in contexts}
    _prob_syl_ctx = {c: prob_syl_ctx(df, c) for c in contexts}
    _cond_prob_ctx = {c: cond_prob_1(df[df.context==c]['seq'], 1) for c in contexts}
    _trans_probs_ctx = {c: trans_prob(df[df.context==c]['seq']) for c in contexts}
    _trans_dict_ctx = {c: trans_dict(df[df.context==c]['seq']) for c in contexts}
    _cond_prob_all = cond_prob_1(df['seq'], 1)
    _cond_prob_all_2 = cond_prob_1(df['seq'], 2)
    for _, row in data_df.iterrows():
        seq = row['seq']; c = row['context']
        a = len(set(seq)); b = len(seq); c_t = len(list(pairwise(seq)))
        d = a / max(c_t, 1)
        if c not in _trans_in_ctx: continue
        e = c_t / max(_trans_in_ctx[c], 1)
        p_ctx = _prob_syl_ctx[c]
        f = -sum(p_ctx.get(s, 1e-9)*np.log2(max(p_ctx.get(s, 1e-9), 1e-9)) for s in seq)
        init_prob = p_ctx; cond1 = _cond_prob_ctx[c]
        tdict = _trans_dict_ctx[c]; tprob = _trans_probs_ctx[c]
        g_v = init_prob.get(seq[0], 1e-9)
        for i in range(1, len(seq)):
            ant, cur = seq[i-1], seq[i]
            g_v *= cond1.get((ant,), {}).get(cur, 1e-9)
        h_v = math.prod([tdict.get(p, 1e-9) for p in pairwise(seq)]) if len(seq)>1 else 0
        i_v = a / max(b, 1)
        j = -sum(tprob.get(p[0], {}).get(p[1], 1e-9)*np.log2(max(tprob.get(p[0], {}).get(p[1], 1e-9), 1e-9)) for p in pairwise(seq))
        probs_k = [G.edges[p]['p_trans'] for p in pairwise(seq) if p in G.edges]
        k = math.prod(probs_k) if probs_k else 0
        l = math.prod([G.edges[p]['p_cond']*np.log2(max(G.edges[p]['p_cond'], 1e-9)) for p in pairwise(seq) if p in G.edges]) or 0
        m = math.prod([G.edges[p]['p_trans']*np.log2(max(G.edges[p]['p_trans'], 1e-9)) for p in pairwise(seq) if p in G.edges]) or 0
        p_cond_n = [init_prob.get(seq[0], 1e-9)]
        for i2 in range(1, len(seq)):
            ant, cur = seq[i2-1], seq[i2]
            p_cond_n.append(_cond_prob_all.get((ant,), {}).get(cur, 1e-9))
        n_v = math.prod(p_cond_n)
        p_cond_o = [G.nodes[seq[0]]['p_frequency'] if seq[0] in G.nodes else 1e-9]
        p_trans_p = []
        for i2 in range(1, len(seq)):
            ant, cur = seq[i2-1], seq[i2]
            p_cond_o.append(G[ant][cur]['p_cond'] if (ant in G and cur in G[ant]) else 1e-9)
            if i2 < len(seq)-1:
                suc = seq[i2+1]
                p_trans_p.append(G[cur][suc]['p_trans'] if (cur in G and suc in G[cur]) else 1e-9)
        o = math.prod(p_cond_o); pv = math.prod(p_trans_p) if p_trans_p else 0
        p_cond_q = [init_prob.get(seq[0], 1e-9)]
        for i2 in range(1, len(seq)):
            ant, cur = seq[i2-1], seq[i2]
            v = _cond_prob_all.get((ant,), {}).get(cur, 1e-9)
            if i2 > 1: v *= _cond_prob_all_2.get((seq[i2-2], ant), {}).get(cur, 1e-9)
            p_cond_q.append(v)
        q = math.prod(p_cond_q)
        r = math.pow(math.prod([1/max(x,1e-9) for x in p_cond_q]), 1/max(len(p_cond_q),1))
        for col, val in zip(cols, [a,b,c_t,d,e,f,g_v,h_v,i_v,j,k,l,m,n_v,o,pv,q,r]):
            out[col].append(val)
    return pd.DataFrame(out)

print('Building Zhang-18 features...')
G = make_graph(seq_df)
feat = features_a_r(seq_df, seq_df, G).replace([np.inf, -np.inf], np.nan).fillna(0)
rng = np.random.default_rng(0)
seq_perm_df = seq_df.copy()
seq_perm_df['seq'] = seq_perm_df['seq'].apply(lambda s: list(rng.permutation(s)))
feat_perm = features_a_r(seq_df, seq_perm_df, G).replace([np.inf, -np.inf], np.nan).fillna(0)

y = seq_df['context'].values
le = LabelEncoder(); y_enc = le.fit_transform(y)
X_tr, X_te, y_tr, y_te = train_test_split(feat.values, y_enc, test_size=0.25, stratify=y_enc, random_state=0)
sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
cv_folds = max(2, min(5, int(np.bincount(y_tr).min())))
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
print(f'CV folds: {cv_folds}')
grid = GridSearchCV(SVC(), {'C': np.logspace(-2,3,6),'gamma': np.logspace(-3,2,6),'kernel':['rbf']},
    cv=cv, scoring='f1_weighted', n_jobs=-1).fit(X_tr_s, y_tr)
y_pred = grid.best_estimator_.predict(X_te_s)
f1_orig = f1_score(y_te, y_pred, average='weighted')
X_perm_s = sc.transform(feat_perm.values)
y_perm_pred = grid.best_estimator_.predict(X_perm_s)
y_perm_true = le.transform(seq_perm_df['context'].values)
f1_perm = f1_score(y_perm_true, y_perm_pred, average='weighted')
print(f'\n=== bat 215, 11-cluster vocab ===')
print(f'F1_orig = {f1_orig:.3f}    [paper: > 0.9]')
print(f'F1_perm = {f1_perm:.3f}    [paper: > 0.9]')
print(f'|delta|  = {abs(f1_orig-f1_perm):.3f}    [paper: ~0]')
