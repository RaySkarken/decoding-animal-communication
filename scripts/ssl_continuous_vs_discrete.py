"""Discretization cost, measured on ONE SSL encoder (apples-to-apples).

Same domain-matched SSL embeddings Z, two ways to decode behavioral context:
  CONTINUOUS: voc-level mean+std pooling of Z -> logistic regression.
  DISCRETE  : k-means(Z) tokens -> bag-of-tokens logistic regression.
The gap is exactly what tokenization throws away. Cross-bat 30/11, 5 seeds, CI.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib, torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX2IDX = {c: i for i, c in enumerate(HP1_CTX)}
N_SEEDS = 5
V_TOK = 120   # best discrete from the sweep

print('Loading state + SSL encoder...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
ctx = seg_df['context'].to_numpy(); em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy(); pos_arr = seg_df['pos_segment'].to_numpy()
from ssl_token_order import Encoder, embed_all
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
mel = st['tf_specs'].reshape(len(seg_df), -1).astype(np.float32)
mu, sd = mel.mean(0), mel.std(0) + 1e-6
enc = Encoder().to(DEVICE); enc.load_state_dict(torch.load(CACHE / 'ssl_encoder_token.pt', map_location=DEVICE))
Z = embed_all(enc, (mel - mu) / sd)
print(f'Z: {Z.shape}', flush=True)
tok = KMeans(V_TOK, n_init=10, random_state=0).fit_predict(Z).astype(np.int32)


def build():
    df = pd.DataFrame({'file': file_arr, 'pos': pos_arr, 'tok': tok, 'ctx': ctx, 'em': em_arr,
                       'idx': np.arange(len(file_arr))})
    df = df[(df['em'] != 0) & (df['ctx'].isin(HP1_CTX))]
    vocs = []
    for fn, g in df.sort_values('pos').groupby('file', sort=False):
        idxs = g['idx'].to_numpy()
        vocs.append({'idx': idxs, 'seq': tok[idxs].tolist(),
                     'ctx': int(np.bincount(g['ctx'].to_numpy()).argmax()),
                     'em': abs(int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0]))})
    return vocs


vocs = build()
print(f'vocs: {len(vocs)}', flush=True)


def split(seed):
    rng = np.random.default_rng(seed); ems = np.array(sorted(set(v['em'] for v in vocs))); rng.shuffle(ems)
    test = set(ems[:11].tolist())
    return [v for v in vocs if v['em'] not in test], [v for v in vocs if v['em'] in test]


def cont_feats(vs):
    X = np.stack([np.concatenate([Z[v['idx']].mean(0), Z[v['idx']].std(0)]) for v in vs])
    y = np.array([CTX2IDX[v['ctx']] for v in vs]); return X, y


def disc_feats(vs):
    X = np.zeros((len(vs), V_TOK), np.float32)
    for i, v in enumerate(vs):
        for t in v['seq']: X[i, t] += 1
        s = X[i].sum()
        if s: X[i] /= s
    y = np.array([CTX2IDX[v['ctx']] for v in vs]); return X, y


def ci95(d):
    d = np.array(d, float); m = d.mean(); h = 2.776 * d.std(ddof=1) / np.sqrt(len(d)); return m, m - h, m + h


if __name__ == '__main__':
    rows = []
    for seed in range(N_SEEDS):
        tr, te = split(seed)
        Xc, yc = cont_feats(tr); Xct, yct = cont_feats(te)
        cont = f1_score(yct, LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xc, yc).predict(Xct),
                        average='macro', labels=range(len(HP1_CTX)), zero_division=0)
        Xd, yd = disc_feats(tr); Xdt, ydt = disc_feats(te)
        disc = f1_score(ydt, LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xd, yd).predict(Xdt),
                        average='macro', labels=range(len(HP1_CTX)), zero_division=0)
        print(f'  seed {seed}: continuous={cont:.3f}  discrete(V={V_TOK})={disc:.3f}  cost={cont-disc:+.3f}', flush=True)
        rows.append({'seed': seed, 'continuous_macro': cont, 'discrete_macro': disc})
    df = pd.DataFrame(rows); df.to_csv(OUT / 'ssl_continuous_vs_discrete.csv', index=False)
    cm, clo, chi = ci95(df.continuous_macro.tolist())
    dm, dlo, dhi = ci95(df.discrete_macro.tolist())
    gm, glo, ghi = ci95((df.continuous_macro - df.discrete_macro).tolist())
    print(f'\nCONTINUOUS macro F1 = {cm:.3f} [{clo:.3f},{chi:.3f}]', flush=True)
    print(f'DISCRETE   macro F1 = {dm:.3f} [{dlo:.3f},{dhi:.3f}]', flush=True)
    print(f'DISCRETIZATION COST = {gm:+.3f} 95%CI[{glo:+.3f},{ghi:+.3f}] '
          f'{"significant" if glo>0 else "n.s."}', flush=True)
    print(f'Saved: {OUT/"ssl_continuous_vs_discrete.csv"}', flush=True)
