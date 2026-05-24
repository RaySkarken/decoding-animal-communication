"""Per-context BERT с маскированным языковым моделированием.

Замысел: для каждого контекста c независимо обучается TinyBERT с
masked-token-prediction objective на последовательностях токенов
этого контекста. На инференсе для каждой контрольной вокализации
вычисляется pseudo-perplexity под каждым из 8 контекстных BERT-ов,
предсказание контекста — argmin perplexity. Это языковая
аналогия нашего основного метода (per-context DP-GMM + max-likelihood),
но учитывает ПОРЯДОК токенов.

Используется глобальный k-means с |V|=15 на UMAP-8D (тот же набор
токенов для всех контекстных моделей --- иначе утечка через словарь).

Cross-bat split: 30 train / 11 test эмиттеров, 5 сидов.
Метрика: macro F1 классификации контекста.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}', flush=True)

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX2IDX = {c: i for i, c in enumerate(HP1_CTX)}
N_CTX = len(HP1_CTX)
V_GLOB = 15        # глобальный k-means
N_PRE_EPOCHS = 30  # эпох MLM на каждый контекст
BATCH = 64
MAX_LEN = 64       # хватает: median 4 segs/voc, p95=17

print('Loading state...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')

ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy()
pos_arr = seg_df['pos_segment'].to_numpy()

print(f'Global k-means V={V_GLOB} on UMAP-8D...', flush=True)
km = KMeans(n_clusters=V_GLOB, n_init=10, random_state=0).fit(emb)
toks_glob = km.labels_.astype(np.int32)


# ─── собрать вокализации ───
def build_vocs(token_arr):
    df = pd.DataFrame({'file': file_arr, 'pos': pos_arr, 'tok': token_arr,
                       'ctx': ctx, 'em': em_arr})
    df = df[df['tok'] >= 0]; df = df[df['em'] != 0]
    df = df[df['ctx'].isin(HP1_CTX)]
    vocs = []
    for fname, g in df.sort_values('pos').groupby('file', sort=False):
        seq = g['tok'].to_numpy().tolist()
        if len(seq) < 1: continue
        dom_ctx = int(np.bincount(g['ctx'].to_numpy()).argmax())
        dom_em = int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0])
        vocs.append({'seq': seq, 'ctx': dom_ctx, 'em': abs(dom_em)})
    return vocs


vocs = build_vocs(toks_glob)
print(f'  vocalisations: {len(vocs)}', flush=True)


# ─── модель ───
class TinyBERT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4,
                 max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.PAD, self.MASK = 0, 1
        self.tok_offset = 2
        self.full_vocab = vocab_size + 2
        self.tok_emb = nn.Embedding(self.full_vocab, d_model, padding_idx=self.PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                            dim_feedforward=d_model * 4,
                                            dropout=dropout, batch_first=True,
                                            activation='gelu', norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.full_vocab)
        self.max_len = max_len

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        pad_mask = (x == self.PAD)
        h = self.enc(h, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        return self.head(h)


def encode(seq, max_len, tok_offset, pad=0):
    seq = seq[:max_len]
    ids = [t + tok_offset for t in seq]
    return ids + [pad] * (max_len - len(ids))


class MLMDataset(Dataset):
    def __init__(self, seqs, tok_offset, pad=0, mask=1, max_len=MAX_LEN,
                 mlm_p=0.15):
        self.seqs = seqs; self.tok_offset = tok_offset; self.pad = pad
        self.mask = mask; self.max_len = max_len; self.mlm_p = mlm_p

    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        ids = torch.tensor(encode(self.seqs[i], self.max_len, self.tok_offset, self.pad),
                            dtype=torch.long)
        target = ids.clone()
        cand = (ids >= self.tok_offset)
        rnd = torch.rand_like(ids.float())
        mask_pos = cand & (rnd < self.mlm_p)
        ids[mask_pos] = self.mask
        target[~mask_pos] = -100
        return ids, target


def train_bert(seqs, vocab_size, n_epochs=N_PRE_EPOCHS, lr=3e-4, weight_decay=1e-4,
               seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TinyBERT(vocab_size=vocab_size).to(DEVICE)
    ds = MLMDataset(seqs, tok_offset=model.tok_offset, pad=model.PAD,
                    mask=model.MASK)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(n_epochs):
        model.train()
        for ids, target in dl:
            ids = ids.to(DEVICE); target = target.to(DEVICE)
            out = model(ids)
            loss = F.cross_entropy(out.view(-1, model.full_vocab),
                                    target.view(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def pseudo_logprob(model, seq):
    """Pseudo log-likelihood: для каждой позиции токена маскируем её,
    считаем log p(true | rest), суммируем. Усреднение по длине — perplexity."""
    if len(seq) == 0: return 0.0, 0
    L = min(len(seq), model.max_len)
    seq = seq[:L]
    ids_base = torch.tensor(encode(seq, model.max_len, model.tok_offset),
                             dtype=torch.long, device=DEVICE)
    total_logp = 0.0
    for i in range(L):
        ids = ids_base.clone().unsqueeze(0)
        true_id = ids[0, i].item()
        ids[0, i] = model.MASK
        out = model(ids)
        logp = F.log_softmax(out[0, i], dim=-1)[true_id].item()
        total_logp += logp
    return total_logp, L


def classify_via_perplexity(models, vocs):
    """Для каждой вокализации: predicted ctx = argmax over c of (avg log prob under model_c)."""
    yt, yp = [], []
    for v in vocs:
        seq = v['seq']
        if not seq: continue
        best_c, best_avglp = None, -np.inf
        for c, m in models.items():
            lp, L = pseudo_logprob(m, seq)
            avglp = lp / max(L, 1)
            if avglp > best_avglp:
                best_avglp = avglp; best_c = c
        if best_c is None: continue
        yt.append(v['ctx']); yp.append(best_c)
    return np.array(yt), np.array(yp)


def split_by_emitter(vocs, seed):
    rng = np.random.default_rng(seed)
    em_uniq = sorted(set(v['em'] for v in vocs))
    em_arr_ = np.array(em_uniq); rng.shuffle(em_arr_)
    test_em = set(em_arr_[:11].tolist())
    return [v for v in vocs if v['em'] not in test_em], [v for v in vocs if v['em'] in test_em]


# ─── основной цикл ───
results = []
print('\n=== Per-context BERT-MLM + argmin pseudo-perplexity ===', flush=True)
for seed in range(5):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    train_v, test_v = split_by_emitter(vocs, seed)
    print(f'\n--- seed {seed}: train={len(train_v)}, test={len(test_v)} ---', flush=True)

    # Тренируем 8 BERT-ов
    models = {}
    for c in HP1_CTX:
        seqs_c = [v['seq'] for v in train_v if v['ctx'] == c]
        if len(seqs_c) < 10:
            continue
        models[c] = train_bert(seqs_c, vocab_size=V_GLOB, seed=seed)
    print(f'  trained {len(models)} BERTs in {time.time()-t0:.0f}s', flush=True)

    # Evaluate
    t_eval = time.time()
    yt, yp = classify_via_perplexity(models, test_v)
    f1m = f1_score(yt, yp, average='macro', labels=HP1_CTX, zero_division=0)
    f1w = f1_score(yt, yp, average='weighted', labels=HP1_CTX, zero_division=0)
    elapsed = time.time() - t0
    print(f'  seed {seed}: f1m={f1m:.3f}, f1w={f1w:.3f}, n_test={len(yt)}  '
          f'(eval {time.time()-t_eval:.0f}s, total {elapsed:.0f}s)', flush=True)
    results.append({'seed': seed, 'f1_m': f1m, 'f1_w': f1w, 'n_test': len(yt)})

df = pd.DataFrame(results)
out = Path('docs/thesis/figures/per_context_bert_lm.csv')
df.to_csv(out, index=False)
print(f'\nSaved: {out}', flush=True)
print(f'\n=== Summary (5 seeds) ===')
print(f'  macro F1   = {df["f1_m"].mean():.3f} ± {df["f1_m"].std():.3f}')
print(f'  weighted F1 = {df["f1_w"].mean():.3f} ± {df["f1_w"].std():.3f}')

print(f'\nBaseline для сравнения:')
print(f'  Per-context DP-GMM full на UMAP-8D + равн. приор: macro 0.313 ± 0.005')
print(f'  Опорный пайплайн (Assom + RF):                    macro 0.237 ± 0.018')
