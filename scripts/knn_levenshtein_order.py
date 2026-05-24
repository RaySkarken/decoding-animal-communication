"""Sarkar-2025 classifier (kNN + Levenshtein) on bat tokens — WITH shuffle control.

Sarkar et al. (NeurIPS 2025 WS) classify animal-vocalization token sequences with
kNN + normalized Levenshtein distance and claim to "leverage sequential structure",
but never isolate whether ORDER matters. We replicate their classifier on Egyptian
fruit-bat tokens for BEHAVIORAL CONTEXT (a task they did not study) and add the
missing control: real order vs within-vocalization SHUFFLED order.

If real ~= shuffled under an order-sensitive metric (edit distance), the apparent
"sequential structure" is actually the token multiset, not the order.

Cross-bat (30/11), 5 seeds, macro F1 + UAR, 95% CI on (real - shuffled).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import time
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import f1_score, recall_score
from rapidfuzz import process, distance as rfdist

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX2IDX = {c: i for i, c in enumerate(HP1_CTX)}
N_SEEDS = 5
K_NN = 5
N_TRAIN_REF = 5000   # reference set subsample for tractable cdist
N_TEST = 2500        # test subsample

print('Loading state...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
ctx = seg_df['context'].to_numpy(); em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy(); pos_arr = seg_df['pos_segment'].to_numpy()


def kmeans_global(e, K):
    return KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(e).astype(np.int32)


def agglomerative_global(e, K, sample=8000):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(e), size=min(sample, len(e)), replace=False)
    lab = AgglomerativeClustering(n_clusters=K, linkage='ward').fit_predict(e[idx])
    cents = np.stack([e[idx][lab == k].mean(0) for k in range(K)])
    out = np.empty(len(e), np.int32)
    for s in range(0, len(e), 20000):
        ch = e[s:s + 20000]
        out[s:s + 20000] = (((ch[:, None, :] - cents[None]) ** 2).sum(-1)).argmin(1)
    return out


def build_vocs(token_arr):
    df = pd.DataFrame({'file': file_arr, 'pos': pos_arr, 'tok': token_arr,
                       'ctx': ctx, 'em': em_arr})
    df = df[(df['tok'] >= 0) & (df['em'] != 0) & (df['ctx'].isin(HP1_CTX))]
    vocs = []
    for fname, g in df.sort_values('pos').groupby('file', sort=False):
        seq = g['tok'].to_numpy().tolist()
        if len(seq) < 1:
            continue
        dom_ctx = int(np.bincount(g['ctx'].to_numpy()).argmax())
        dom_em = int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0])
        vocs.append({'seq': seq, 'ctx': dom_ctx, 'em': abs(dom_em)})
    return vocs


def to_str(seq, shuffle_rng=None):
    seq = list(seq)
    if shuffle_rng is not None:
        shuffle_rng.shuffle(seq)
    return ''.join(chr(33 + t) for t in seq)   # vocab<120 -> printable range


def split_by_emitter(vocs, seed):
    rng = np.random.default_rng(seed)
    ems = np.array(sorted(set(v['em'] for v in vocs))); rng.shuffle(ems)
    test_em = set(ems[:11].tolist())
    return ([v for v in vocs if v['em'] not in test_em],
            [v for v in vocs if v['em'] in test_em])


def knn_levenshtein(train, test, seed, shuffle):
    rng_tr = np.random.default_rng(seed) if shuffle else None
    rng_te = np.random.default_rng(seed + 7) if shuffle else None
    # subsample for tractable cdist
    r = np.random.default_rng(seed)
    tr = [train[i] for i in r.choice(len(train), min(N_TRAIN_REF, len(train)), replace=False)]
    te = [test[i] for i in r.choice(len(test), min(N_TEST, len(test)), replace=False)]
    tr_str = [to_str(v['seq'], np.random.default_rng(seed * 1000 + i) if shuffle else None) for i, v in enumerate(tr)]
    te_str = [to_str(v['seq'], np.random.default_rng(seed * 9000 + i) if shuffle else None) for i, v in enumerate(te)]
    tr_y = np.array([CTX2IDX[v['ctx']] for v in tr])
    te_y = np.array([CTX2IDX[v['ctx']] for v in te])
    # batched normalized Levenshtein distance matrix (te x tr)
    D = process.cdist(te_str, tr_str, scorer=rfdist.Levenshtein.normalized_distance,
                      workers=-1)
    pred = np.empty(len(te), np.int64)
    for i in range(len(te)):
        nn = np.argpartition(D[i], K_NN)[:K_NN]
        w = 1.0 / (D[i, nn] + 1e-6)
        votes = np.zeros(len(HP1_CTX))
        for j, idx in enumerate(nn):
            votes[tr_y[idx]] += w[j]
        pred[i] = votes.argmax()
    return (f1_score(te_y, pred, average='macro', labels=range(len(HP1_CTX)), zero_division=0),
            recall_score(te_y, pred, average='macro', labels=range(len(HP1_CTX)), zero_division=0))


def ci95(d):
    d = np.array(d); m = d.mean(); s = d.std(ddof=1)
    h = 2.776 * s / np.sqrt(len(d))
    return m, m - h, m + h


if __name__ == '__main__':
    toksets = {'kmeans30': kmeans_global(emb, 30), 'agglo30': agglomerative_global(emb, 30)}
    rows = []
    for name, tk in toksets.items():
        vocs = build_vocs(tk)
        print(f'\n=== {name}: {len(vocs)} vocs ===', flush=True)
        for seed in range(N_SEEDS):
            t0 = time.time()
            tr, te = split_by_emitter(vocs, seed)
            real_m, real_u = knn_levenshtein(tr, te, seed, shuffle=False)
            shuf_m, shuf_u = knn_levenshtein(tr, te, seed, shuffle=True)
            print(f'  seed {seed}: kNN-Lev real macroF1={real_m:.3f} UAR={real_u:.3f} | '
                  f'shuf macroF1={shuf_m:.3f} UAR={shuf_u:.3f}  ({time.time()-t0:.0f}s)', flush=True)
            rows.append({'tokenizer': name, 'seed': seed,
                         'knn_real_macro': real_m, 'knn_real_uar': real_u,
                         'knn_shuf_macro': shuf_m, 'knn_shuf_uar': shuf_u})
        pd.DataFrame(rows).to_csv(OUT / 'knn_levenshtein_order.csv', index=False)
    df = pd.DataFrame(rows)
    print('\n=== kNN-Levenshtein: does ORDER help? (macro F1) ===', flush=True)
    for name in toksets:
        sub = df[df.tokenizer == name]
        m, lo, hi = ci95((sub.knn_real_macro - sub.knn_shuf_macro).tolist())
        print(f'{name:10s} real={sub.knn_real_macro.mean():.3f}±{sub.knn_real_macro.std():.3f} '
              f'shuf={sub.knn_shuf_macro.mean():.3f}±{sub.knn_shuf_macro.std():.3f} '
              f'Δ={m:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] '
              f'{"ORDER HELPS" if lo > 0 else "order n.s."}', flush=True)
    print(f'\nSaved: {OUT/"knn_levenshtein_order.csv"}', flush=True)
