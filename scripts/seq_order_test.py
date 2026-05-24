"""Does token ORDER carry behavioural-context signal? (honest shuffle control)

Central conference experiment. The thesis showed order-agnostic methods
(bag-of-tokens RF, per-context DP-GMM density) classify context above an
oporny baseline, but never tested whether token ORDER adds anything. The
clustering geometry is poor (low silhouette, contexts overlap) — yet the
SEQUENCE domain may still hold signal.

We compare, cross-bat (30 train / 11 test emitters, 5 seeds), macro F1:
  (1) BERT-CLS on real token order        (uses order)
  (2) BERT-CLS on within-voc SHUFFLED order (order destroyed; multiset kept)
  (3) bag-of-tokens logistic regression    (order-agnostic lower bound)

Honest reading:
  - if (1) >> (2): token order carries context info -> combinatorial syntax,
    sequence modeling justified.
  - if (1) ~= (2): order does not help; BERT exploits the multiset only
    (consistent with the thesis's associative-syntax finding) -> honest negative
    for "sequence modeling", but a clean methodological result.
  - (2) ~= (3) is a sanity check (both order-agnostic).

Reported with 95% CI on the paired (real - shuffled) difference per seed.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import time, sys
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, silhouette_score

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
OUT = Path('conference/results')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX2IDX = {c: i for i, c in enumerate(HP1_CTX)}
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
print(f'Device: {DEVICE}, seeds={N_SEEDS}', flush=True)

print('Loading state...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy()
pos_arr = seg_df['pos_segment'].to_numpy()


# ─────────────── tokenizers ───────────────
def kmeans_global(e, K):
    return KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(e).astype(np.int32)


def per_context_kmeans(e, ctx_arr, K=15):
    labels = np.full(len(e), -1, dtype=np.int32)
    off = 0
    for c in HP1_CTX:
        mc = ctx_arr == c
        if mc.sum() < 30:
            continue
        km = KMeans(n_clusters=min(K, mc.sum() // 5), n_init=10, random_state=0).fit(e[mc])
        labels[mc] = km.labels_ + off
        off += K
    return labels


def agglomerative_global(e, K, sample=8000):
    """Ward agglomerative; fit on a sample (O(n^2) memory -> keep small),
    assign all points to nearest sample-cluster centroid."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(e), size=min(sample, len(e)), replace=False)
    ac = AgglomerativeClustering(n_clusters=K, linkage='ward')
    lab_s = ac.fit_predict(e[idx])
    cents = np.stack([e[idx][lab_s == k].mean(0) for k in range(K)])
    # chunked nearest-centroid assignment (avoid 152k x K x d transient)
    out = np.empty(len(e), dtype=np.int32)
    for s in range(0, len(e), 20000):
        chunk = e[s:s + 20000]
        d = ((chunk[:, None, :] - cents[None]) ** 2).sum(-1)
        out[s:s + 20000] = d.argmin(1)
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


def split_by_emitter(vocs, seed):
    rng = np.random.default_rng(seed)
    ems = np.array(sorted(set(v['em'] for v in vocs)))
    rng.shuffle(ems)
    test_em = set(ems[:11].tolist())
    return ([v for v in vocs if v['em'] not in test_em],
            [v for v in vocs if v['em'] in test_em])


# ─────────────── TinyBERT (CLS + MLM) ───────────────
class TinyBERT(nn.Module):
    def __init__(self, vocab_size, n_ctx, d_model=64, n_layers=2, n_heads=4,
                 max_len=256, dropout=0.1):
        super().__init__()
        self.PAD, self.CLS, self.MASK = 0, 1, 2
        self.tok_offset = 3
        self.full_vocab = vocab_size + 3
        self.tok_emb = nn.Embedding(self.full_vocab, d_model, padding_idx=self.PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                            dim_feedforward=d_model * 4, dropout=dropout,
                                            batch_first=True, activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, self.full_vocab)
        self.cls_head = nn.Linear(d_model, n_ctx)
        self.max_len = max_len

    def forward(self, x, return_mlm=False, return_cls=False):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.norm(self.enc(h, src_key_padding_mask=(x == self.PAD)))
        out = {}
        if return_mlm:
            out['mlm'] = self.mlm_head(h)
        if return_cls:
            out['cls'] = self.cls_head(h[:, 0])
        return out


def encode_seq(seq, max_len, model, shuffle_rng=None):
    seq = list(seq[:max_len - 1])
    if shuffle_rng is not None:
        shuffle_rng.shuffle(seq)            # destroy order, keep multiset
    ids = [model.CLS] + [t + model.tok_offset for t in seq]
    return ids + [model.PAD] * (max_len - len(ids))


class VocDS(Dataset):
    def __init__(self, vocs, model, max_len, do_mlm=False, shuffle=False, seed=0):
        self.vocs, self.model, self.max_len = vocs, model, max_len
        self.do_mlm, self.shuffle = do_mlm, shuffle
        self.seed = seed

    def __len__(self):
        return len(self.vocs)

    def __getitem__(self, i):
        v = self.vocs[i]
        # deterministic per-item shuffle so train/eval are stable
        srng = np.random.default_rng(self.seed * 1_000_003 + i) if self.shuffle else None
        ids = torch.tensor(encode_seq(v['seq'], self.max_len, self.model, srng), dtype=torch.long)
        y = CTX2IDX[v['ctx']]
        if not self.do_mlm:
            return ids, y
        target = ids.clone()
        cand = ids >= self.model.tok_offset
        rnd = torch.rand_like(ids.float())
        mpos = cand & (rnd < 0.15)
        ids[mpos] = self.model.MASK
        target[~mpos] = -100
        return ids, target, y


MAX_LEN = 64   # median voc length ~4, p95 ~17; 64 covers ~p99.9, big speedup vs padding to longest


def train_eval_bert(tr, te, vocab_size, seed, shuffle, pre_ep=8, ft_ep=15):
    torch.manual_seed(seed); np.random.seed(seed)
    max_len = min(MAX_LEN, max(len(v['seq']) for v in tr) + 1)
    model = TinyBERT(vocab_size, len(HP1_CTX), max_len=max_len).to(DEVICE)
    # Stage 1: MLM pretrain (on same order regime)
    ds_pre = VocDS(tr, model, max_len, do_mlm=True, shuffle=shuffle, seed=seed)
    dl_pre = DataLoader(ds_pre, batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for ep in range(pre_ep):
        model.train()
        for ids, target, _ in dl_pre:
            ids, target = ids.to(DEVICE), target.to(DEVICE)
            loss = F.cross_entropy(model(ids, return_mlm=True)['mlm'].view(-1, model.full_vocab),
                                   target.view(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
    # Stage 2: finetune CLS
    ds_tr = VocDS(tr, model, max_len, do_mlm=False, shuffle=shuffle, seed=seed)
    ds_te = VocDS(te, model, max_len, do_mlm=False, shuffle=shuffle, seed=seed)
    dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for ep in range(ft_ep):
        model.train()
        for ids, y in dl_tr:
            ids, y = ids.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(ids, return_cls=True)['cls'], y)
            opt.zero_grad(); loss.backward(); opt.step()
    # eval
    model.eval()
    dl_te = DataLoader(ds_te, batch_size=256, shuffle=False)
    pred, true = [], []
    with torch.no_grad():
        for ids, y in dl_te:
            pred.append(model(ids.to(DEVICE), return_cls=True)['cls'].argmax(-1).cpu().numpy())
            true.append(y.numpy())
    pred, true = np.concatenate(pred), np.concatenate(true)
    return (f1_score(true, pred, average='macro', zero_division=0),
            f1_score(true, pred, average='weighted', zero_division=0))


def bag_of_tokens_lr(tr, te, vocab_size, seed):
    def hist(vocs):
        X = np.zeros((len(vocs), vocab_size), np.float32)
        for i, v in enumerate(vocs):
            for t in v['seq']:
                X[i, t] += 1
            X[i] /= max(1, X[i].sum())
        y = np.array([CTX2IDX[v['ctx']] for v in vocs])
        return X, y
    Xtr, ytr = hist(tr); Xte, yte = hist(te)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0).fit(Xtr, ytr)
    p = clf.predict(Xte)
    return (f1_score(yte, p, average='macro', zero_division=0),
            f1_score(yte, p, average='weighted', zero_division=0))


def ci95(diffs):
    d = np.array(diffs); m = d.mean(); s = d.std(ddof=1)
    half = 2.776 * s / np.sqrt(len(d))   # t_{4,0.975}
    return m, m - half, m + half


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    print('Building tokenizations...', flush=True)
    tokset = {
        'kmeans30': kmeans_global(emb, 30),
        'percontext': per_context_kmeans(emb, ctx, K=15),
        'agglo30': agglomerative_global(emb, 30),
    }
    # geometric quality (silhouette on a sample) for the geometry-vs-sequence story
    sil = {}
    rng = np.random.default_rng(0)
    sidx = rng.choice(len(emb), size=20000, replace=False)
    for name, tk in tokset.items():
        m = tk[sidx] >= 0
        try:
            sil[name] = silhouette_score(emb[sidx][m], tk[sidx][m])
        except Exception:
            sil[name] = float('nan')
        print(f'  {name}: vocab={len(set(tk[tk>=0].tolist()))}, silhouette={sil[name]:.3f}', flush=True)

    rows = []
    for name, tk in tokset.items():
        vocs = build_vocs(tk)
        vocab_size = int(max(v['seq'][i] for v in vocs for i in range(len(v['seq']))) + 1)
        print(f'\n=== tokenizer {name}: {len(vocs)} vocs, vocab={vocab_size} ===', flush=True)
        for seed in range(N_SEEDS):
            t0 = time.time()
            tr, te = split_by_emitter(vocs, seed)
            real_m, real_w = train_eval_bert(tr, te, vocab_size, seed, shuffle=False)
            shuf_m, shuf_w = train_eval_bert(tr, te, vocab_size, seed, shuffle=True)
            bag_m, bag_w = bag_of_tokens_lr(tr, te, vocab_size, seed)
            print(f'  seed {seed}: BERT-real macroF1={real_m:.3f} | BERT-shuf={shuf_m:.3f} '
                  f'| bag-LR={bag_m:.3f}  ({time.time()-t0:.0f}s)', flush=True)
            rows.append({'tokenizer': name, 'seed': seed, 'silhouette': round(sil[name], 4),
                         'bert_real_macro': real_m, 'bert_real_weighted': real_w,
                         'bert_shuf_macro': shuf_m, 'bert_shuf_weighted': shuf_w,
                         'bag_lr_macro': bag_m, 'bag_lr_weighted': bag_w})
        df = pd.DataFrame(rows)
        df.to_csv(OUT / 'seq_order_test.csv', index=False)

    df = pd.DataFrame(rows)
    print('\n=== Summary: does ORDER help? (macro F1, mean over seeds) ===', flush=True)
    for name in tokset:
        sub = df[df.tokenizer == name]
        diffs = (sub['bert_real_macro'] - sub['bert_shuf_macro']).tolist()
        m, lo, hi = ci95(diffs)
        print(f'{name:12s} sil={sub.silhouette.iloc[0]:.3f} | '
              f'real={sub.bert_real_macro.mean():.3f}±{sub.bert_real_macro.std():.3f} '
              f'shuf={sub.bert_shuf_macro.mean():.3f}±{sub.bert_shuf_macro.std():.3f} '
              f'bag={sub.bag_lr_macro.mean():.3f} | '
              f'Δ(real-shuf)={m:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] '
              f'{"ORDER HELPS" if lo>0 else "order n.s."}', flush=True)
    print(f'\nSaved: {OUT/"seq_order_test.csv"}', flush=True)
