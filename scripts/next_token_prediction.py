"""Сравнение global vs per-context токенизации по next-token-predictability.

Замысел: качество токенизации можно измерить тем, насколько хорошо
языковая модель предсказывает следующий токен в последовательности
вокализации. Если per-context разметка лучше захватывает внутри-
контекстную структуру, то per-context языковая модель должна давать
меньшую perplexity на test, чем общая глобальная модель.

Сравниваем (одинаковый размер словаря |V|=15 в обеих схемах):
  GLOBAL  : 1 BERT с causal LM, обучен на всех train-вокализациях
            (глобальный k-means V=15 на UMAP-8D)
  PER-CTX : 8 BERT-ов (по одному на контекст), каждый обучен только
            на train-вокализациях своего контекста с per-context
            k-means V=15 на UMAP-8D

Архитектура: TinyBERT с causal-маской → autoregressive predictor.

Eval metrics на test-вокализациях:
  - token-level cross-entropy loss (NLL)
  - perplexity = exp(avg NLL)
  - accuracy@1 (top-1 next-token accuracy)
  - accuracy@3 (top-3)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}', flush=True)

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
N_CTX = len(HP1_CTX)
V = 15           # одинаковый размер словаря в обоих режимах
N_EPOCHS = 30
BATCH = 64
MAX_LEN = 64
PAD, BOS = 0, 1
TOK_OFFSET = 2

print('Loading state...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')

ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy()
pos_arr = seg_df['pos_segment'].to_numpy()


def global_kmeans(embed, V):
    return KMeans(n_clusters=V, n_init=10, random_state=0).fit_predict(embed).astype(np.int32)


def per_context_kmeans(embed, ctx_arr, V):
    """Per-context k-means с общим диапазоном {0..V-1}."""
    labels = np.full(len(embed), -1, dtype=np.int32)
    for c in HP1_CTX:
        mc = ctx_arr == c
        if mc.sum() < 30: continue
        km = KMeans(n_clusters=min(V, mc.sum() // 5), n_init=10, random_state=0).fit(embed[mc])
        labels[mc] = km.labels_
    return labels


print(f'Global k-means V={V}...', flush=True)
toks_glob = global_kmeans(emb, V)
print(f'Per-context k-means V={V}...', flush=True)
toks_pc = per_context_kmeans(emb, ctx, V)


def build_vocs(token_arr):
    df = pd.DataFrame({'file': file_arr, 'pos': pos_arr, 'tok': token_arr,
                       'ctx': ctx, 'em': em_arr})
    df = df[df['tok'] >= 0]
    df = df[df['em'] != 0]
    df = df[df['ctx'].isin(HP1_CTX)]
    vocs = []
    for fname, g in df.sort_values('pos').groupby('file', sort=False):
        seq = g['tok'].to_numpy().tolist()
        if len(seq) < 2: continue   # для next-token нужно ≥ 2 токена
        dom_ctx = int(np.bincount(g['ctx'].to_numpy()).argmax())
        dom_em = int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0])
        vocs.append({'seq': seq, 'ctx': dom_ctx, 'em': abs(dom_em)})
    return vocs


vocs_glob = build_vocs(toks_glob)
vocs_pc = build_vocs(toks_pc)
print(f'  vocalisations (global): {len(vocs_glob)}', flush=True)
print(f'  vocalisations (per-ctx): {len(vocs_pc)}', flush=True)


# ─── модель: TinyBERT с causal маской ───
class CausalLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4,
                 max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.full_vocab = vocab_size + 2   # +PAD +BOS
        self.tok_emb = nn.Embedding(self.full_vocab, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.full_vocab)
        self.max_len = max_len

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.tok_emb(x) + self.pos_emb(pos)
        pad_mask = (x == PAD)
        causal = nn.Transformer.generate_square_subsequent_mask(L).to(x.device)
        h = self.enc(h, mask=causal, src_key_padding_mask=pad_mask, is_causal=True)
        h = self.norm(h)
        return self.head(h)


def encode(seq, max_len=MAX_LEN, bos=BOS, pad=PAD, tok_offset=TOK_OFFSET):
    seq = seq[:max_len - 1]
    ids = [bos] + [t + tok_offset for t in seq]
    return ids + [pad] * (max_len - len(ids))


class LMDataset(Dataset):
    def __init__(self, seqs, max_len=MAX_LEN):
        self.seqs = seqs; self.max_len = max_len
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        ids = encode(self.seqs[i], self.max_len)
        x = torch.tensor(ids[:-1], dtype=torch.long)  # input
        y = torch.tensor(ids[1:], dtype=torch.long)   # target (shifted)
        return x, y


def train_lm(seqs, vocab_size, n_epochs=N_EPOCHS, lr=3e-4, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CausalLM(vocab_size=vocab_size).to(DEVICE)
    if not seqs: return model
    ds = LMDataset(seqs)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(n_epochs):
        model.train()
        for x, y in dl:
            x = x.to(DEVICE); y = y.to(DEVICE)
            out = model(x)
            # игнорируем PAD позиции
            loss = F.cross_entropy(out.reshape(-1, model.full_vocab),
                                    y.reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_lm(model, seqs):
    """Возвращает суммарную NLL, количество предсказанных позиций,
    top-1 hits, top-3 hits."""
    model.eval()
    if not seqs: return 0.0, 0, 0, 0
    ds = LMDataset(seqs); dl = DataLoader(ds, batch_size=BATCH, shuffle=False)
    total_nll = 0.0; total_n = 0; top1_hits = 0; top3_hits = 0
    for x, y in dl:
        x = x.to(DEVICE); y = y.to(DEVICE)
        logits = model(x)
        # маска валидных позиций (PAD исключаются)
        valid = (y != PAD)
        logp = F.log_softmax(logits, dim=-1)
        # NLL
        nll = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        total_nll += (nll * valid).sum().item()
        total_n += valid.sum().item()
        # top-k
        top3 = logits.topk(3, dim=-1).indices
        eq1 = (top3[..., 0] == y) & valid
        eq3 = (top3 == y.unsqueeze(-1)).any(-1) & valid
        top1_hits += eq1.sum().item()
        top3_hits += eq3.sum().item()
    return total_nll, total_n, top1_hits, top3_hits


def split_by_emitter(vocs, seed):
    rng = np.random.default_rng(seed)
    em_uniq = sorted(set(v['em'] for v in vocs))
    em_arr_ = np.array(em_uniq); rng.shuffle(em_arr_)
    test_em = set(em_arr_[:11].tolist())
    return ([v for v in vocs if v['em'] not in test_em],
            [v for v in vocs if v['em'] in test_em])


print('\n=== Next-token prediction: global vs per-context (5 сидов) ===', flush=True)
rows = []
for seed in range(5):
    t0 = time.time()
    # Глобальная схема: один BERT
    train_g, test_g = split_by_emitter(vocs_glob, seed)
    train_p, test_p = split_by_emitter(vocs_pc, seed)
    assert len(train_g) == len(train_p)  # фильтры идентичны

    # train GLOBAL
    g_model = train_lm([v['seq'] for v in train_g], vocab_size=V, seed=seed)

    # train PER-CONTEXT (8 моделей)
    pc_models = {}
    for c in HP1_CTX:
        seqs_c = [v['seq'] for v in train_p if v['ctx'] == c]
        if len(seqs_c) < 10: continue
        pc_models[c] = train_lm(seqs_c, vocab_size=V, seed=seed)

    # eval GLOBAL: одна модель на всём test
    nll_g, n_g, h1_g, h3_g = eval_lm(g_model, [v['seq'] for v in test_g])
    ppl_g = float(np.exp(nll_g / max(n_g, 1)))
    acc1_g = h1_g / max(n_g, 1); acc3_g = h3_g / max(n_g, 1)

    # eval PER-CONTEXT: для каждой test-вокализации применяем её контекстную модель
    nll_p, n_p, h1_p, h3_p = 0.0, 0, 0, 0
    for c in HP1_CTX:
        if c not in pc_models: continue
        seqs_c = [v['seq'] for v in test_p if v['ctx'] == c]
        if not seqs_c: continue
        a, b, c1, c3 = eval_lm(pc_models[c], seqs_c)
        nll_p += a; n_p += b; h1_p += c1; h3_p += c3
    ppl_p = float(np.exp(nll_p / max(n_p, 1)))
    acc1_p = h1_p / max(n_p, 1); acc3_p = h3_p / max(n_p, 1)

    elapsed = time.time() - t0
    print(f'\n--- seed {seed} ({elapsed:.0f}s, train={len(train_g)}, test={len(test_g)}, '
          f'eval positions={n_g}) ---', flush=True)
    print(f'  GLOBAL  : PPL={ppl_g:.3f}, acc@1={acc1_g:.3f}, acc@3={acc3_g:.3f}',
          flush=True)
    print(f'  PER-CTX : PPL={ppl_p:.3f}, acc@1={acc1_p:.3f}, acc@3={acc3_p:.3f}',
          flush=True)
    rows.append({'seed': seed, 'n_test_positions': n_g,
                 'global_ppl': ppl_g, 'global_acc1': acc1_g, 'global_acc3': acc3_g,
                 'pc_ppl': ppl_p, 'pc_acc1': acc1_p, 'pc_acc3': acc3_p})


df = pd.DataFrame(rows)
out = Path('docs/thesis/figures/next_token_prediction.csv')
df.to_csv(out, index=False)
print(f'\nSaved: {out}', flush=True)

print(f'\n=== Summary (mean ± std over 5 seeds) ===')
for col_g, col_p, name in [('global_ppl', 'pc_ppl', 'Perplexity (lower better)'),
                            ('global_acc1', 'pc_acc1', 'Accuracy@1 (higher better)'),
                            ('global_acc3', 'pc_acc3', 'Accuracy@3 (higher better)')]:
    g_m, g_s = df[col_g].mean(), df[col_g].std()
    p_m, p_s = df[col_p].mean(), df[col_p].std()
    d = p_m - g_m
    print(f'{name:32s} | GLOBAL: {g_m:.3f} ± {g_s:.3f} | PER-CTX: {p_m:.3f} ± {p_s:.3f} | Δ(pc-glob): {d:+.3f}')
