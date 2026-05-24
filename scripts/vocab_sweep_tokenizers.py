"""Is the SSL-tokenizer advantage robust across vocabulary size?

Context macro F1 (bag-of-tokens logistic regression; bag~=BERT for these tokens,
and bag-LR is CPU-only so it doesn't contend with GPU runs) for SSL vs mel-UMAP
tokenizers across V in {10,15,30,60,120}. Cross-bat 30/11, 5 seeds, 95% CI.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import time, sys
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
VS = [10, 15, 30, 60, 120]
N_SEEDS = 5

print('Loading state...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
ctx = seg_df['context'].to_numpy(); em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy(); pos_arr = seg_df['pos_segment'].to_numpy()

# SSL embeddings (reuse cached encoder)
sslp = CACHE / 'ssl_encoder_token.pt'
Z = None
if sslp.exists():
    from ssl_token_order import Encoder, embed_all
    DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    mel = st['tf_specs'].reshape(len(seg_df), -1).astype(np.float32)
    mu, sd = mel.mean(0), mel.std(0) + 1e-6
    enc = Encoder().to(DEVICE); enc.load_state_dict(torch.load(sslp, map_location=DEVICE))
    Z = embed_all(enc, (mel - mu) / sd)
    print(f'SSL embeddings: {Z.shape}', flush=True)
else:
    print('WARNING: no SSL encoder cached; running mel-UMAP only', flush=True)


def build_vocs(tok):
    df = pd.DataFrame({'file': file_arr, 'pos': pos_arr, 'tok': tok, 'ctx': ctx, 'em': em_arr})
    df = df[(df['tok'] >= 0) & (df['em'] != 0) & (df['ctx'].isin(HP1_CTX))]
    vocs = []
    for fn, g in df.sort_values('pos').groupby('file', sort=False):
        seq = g['tok'].to_numpy().tolist()
        if not seq: continue
        vocs.append({'seq': seq, 'ctx': int(np.bincount(g['ctx'].to_numpy()).argmax()),
                     'em': abs(int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0]))})
    return vocs


def split(vocs, seed):
    rng = np.random.default_rng(seed); ems = np.array(sorted(set(v['em'] for v in vocs))); rng.shuffle(ems)
    test = set(ems[:11].tolist())
    return [v for v in vocs if v['em'] not in test], [v for v in vocs if v['em'] in test]


def bag_lr(tr, te, vocab):
    def hist(vocs):
        X = np.zeros((len(vocs), vocab), np.float32)
        for i, v in enumerate(vocs):
            for t in v['seq']:
                if t < vocab: X[i, t] += 1
            s = X[i].sum()
            if s: X[i] /= s
        return X, np.array([CTX2IDX[v['ctx']] for v in vocs])
    Xtr, ytr = hist(tr); Xte, yte = hist(te)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xtr, ytr)
    return f1_score(yte, clf.predict(Xte), average='macro', labels=range(len(HP1_CTX)), zero_division=0)


def ci95(d):
    d = np.array(d, float); m = d.mean(); h = 2.776 * d.std(ddof=1) / np.sqrt(len(d)); return m, m - h, m + h


if __name__ == '__main__':
    sources = {'mel_umap': emb}
    if Z is not None:
        sources['ssl'] = Z
    rows = []
    for src_name, X in sources.items():
        for V in VS:
            tok = KMeans(V, n_init=10, random_state=0).fit_predict(X).astype(np.int32)
            vocs = build_vocs(tok)
            f1s = []
            for seed in range(N_SEEDS):
                tr, te = split(vocs, seed)
                f1s.append(bag_lr(tr, te, V))
            m, lo, hi = ci95(f1s)
            print(f'{src_name:8s} V={V:3d}: macroF1={m:.3f} 95%CI[{lo:.3f},{hi:.3f}]', flush=True)
            rows.append({'source': src_name, 'V': V, 'macro_f1_mean': round(m, 4),
                         'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
                         'f1_per_seed': ';'.join(f'{x:.3f}' for x in f1s)})
            pd.DataFrame(rows).to_csv(OUT / 'vocab_sweep_tokenizers.csv', index=False)
    print(f'\nSaved: {OUT/"vocab_sweep_tokenizers.csv"}', flush=True)
