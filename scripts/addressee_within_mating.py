"""Predicting the addressee inside the Mating-protest context.

Same protocol as within-context caller-ID (scripts/within_context_caller_id.py):
filter to one behavioural context (Mating), pool segments into per-vocalisation
feature vectors, train Random Forest on four representations, evaluate under two
split protocols (stratified-random and cross-bat-by-emitter), report weighted
and macro F1 averaged over 5 seeds.

Representations:
  - Bag-of-syllables, GLOBAL  (HDBSCAN+NCA, |V|=11)
  - Bag-of-syllables, PER-CTX (k-means on Mating, |V|=15)
  - Pooled UMAP-8D (mean+std)
  - Pooled raw mel-spectrogram (mean+std, 672D)

Filter: keep addressee classes with >= 30 vocalisations.

Classifier: RandomForestClassifier(n_estimators=300, class_weight='balanced')
--- same as caller-ID and the Assom baseline.

Output: docs/thesis/figures/addressee_within_mating.csv
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
MATING_CTX = 2
MIN_VOCS_PER_ADDR = 74  # даёт 7 классов адресата, как в исходной (черновой) версии эксперимента
N_SEEDS = 5
N_FOLDS = 5

print('Loading state...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
mel = st['tf_specs'].reshape(len(seg_df), -1).astype(np.float32)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
hdb_nca = np.load(CACHE / 'hdb_nca_labels_152k_21x32.npy')

ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
addr_arr = seg_df['addressee'].to_numpy()
file_arr = seg_df['file_name'].to_numpy()
pos_arr = seg_df['pos_segment'].to_numpy()

# Per-context k-means on Mating (V=15)
print('Computing per-context k-means on Mating...', flush=True)
mating_mask = ctx == MATING_CTX
km = KMeans(n_clusters=15, n_init=10, random_state=0).fit(emb[mating_mask])
pc_labels = np.full(len(emb), -1, dtype=np.int32)
pc_labels[mating_mask] = km.labels_


def build_voc_features(token_arr):
    """For each Mating vocalisation: bag-of-syllables, mel-mean+std, UMAP-mean+std,
    plus addressee, emitter, file. Filters: addressee in {emitter ids}, token >= 0,
    addressee classes with >=30 vocs."""
    mask = (ctx == MATING_CTX) & (em_arr != 0) & (addr_arr != 0) & (token_arr >= 0)
    if mask.sum() == 0: return None
    V = int(token_arr[mask].max()) + 1
    sub = pd.DataFrame({
        'idx': np.arange(len(seg_df))[mask],
        'file': file_arr[mask], 'pos': pos_arr[mask],
        'tok': token_arr[mask], 'em': em_arr[mask], 'addr': addr_arr[mask],
    })
    X_tok, X_mel, X_emb, y_addr, y_em, files = [], [], [], [], [], []
    for fname, g in sub.sort_values('pos').groupby('file', sort=False):
        idxs = g['idx'].to_numpy()
        toks_v = g['tok'].to_numpy()
        freq = np.bincount(toks_v, minlength=V) / max(len(toks_v), 1)
        ent = -np.sum(freq * np.log(freq + 1e-12))
        feat_tok = np.concatenate([freq, [len(toks_v), ent, freq.max(),
                                          float(np.count_nonzero(freq))]])
        m = mel[idxs]; e = emb[idxs]
        feat_mel = np.concatenate([m.mean(0), m.std(0)])
        feat_emb = np.concatenate([e.mean(0), e.std(0)])
        X_tok.append(feat_tok); X_mel.append(feat_mel); X_emb.append(feat_emb)
        y_addr.append(int(Counter(g['addr'].to_numpy().tolist()).most_common(1)[0][0]))
        y_em.append(int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0]))
        files.append(fname)
    X_tok = np.array(X_tok, dtype=np.float32)
    X_mel = np.array(X_mel, dtype=np.float32)
    X_emb = np.array(X_emb, dtype=np.float32)
    y_addr = np.array(y_addr); y_em = np.array(y_em)
    return X_tok, X_mel, X_emb, y_addr, y_em, np.array(files)


def filter_min_per_class(X, y_addr, y_em, min_count=MIN_VOCS_PER_ADDR):
    cnt = Counter(y_addr.tolist())
    keep = np.array([cnt[a] >= min_count for a in y_addr])
    return X[keep], y_addr[keep], y_em[keep]


def stratified_eval(X, y, n_folds=N_FOLDS, n_seeds=N_SEEDS):
    """Stratified k-fold by addressee class, averaged over seeds."""
    f1w_per_seed, f1m_per_seed = [], []
    for s in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
        f1w, f1m = [], []
        for tr, te in skf.split(X, y):
            rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                         random_state=s, n_jobs=-1).fit(X[tr], y[tr])
            pr = rf.predict(X[te])
            f1w.append(f1_score(y[te], pr, average='weighted', zero_division=0))
            f1m.append(f1_score(y[te], pr, average='macro', zero_division=0))
        f1w_per_seed.append(np.mean(f1w)); f1m_per_seed.append(np.mean(f1m))
    return (float(np.mean(f1w_per_seed)), float(np.std(f1w_per_seed)),
            float(np.mean(f1m_per_seed)), float(np.std(f1m_per_seed)))


def crossbat_eval(X, y_addr, y_em, n_seeds=N_SEEDS):
    """Cross-bat split: hold out vocalisations of randomly-chosen emitters; ~30% test."""
    f1w_per_seed, f1m_per_seed = [], []
    em_uniq = np.array(sorted(set(y_em.tolist())))
    n_test_em = max(1, int(round(0.3 * len(em_uniq))))
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        em_perm = em_uniq.copy(); rng.shuffle(em_perm)
        test_em = set(em_perm[:n_test_em].tolist())
        tr = np.array([i for i, e in enumerate(y_em) if e not in test_em])
        te = np.array([i for i, e in enumerate(y_em) if e in test_em])
        if len(te) == 0 or len(tr) == 0: continue
        # На тесте могут оказаться addressee, не встречавшиеся в train --- f1_score обработает
        rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                     random_state=s, n_jobs=-1).fit(X[tr], y_addr[tr])
        pr = rf.predict(X[te])
        f1w_per_seed.append(f1_score(y_addr[te], pr, average='weighted', zero_division=0))
        f1m_per_seed.append(f1_score(y_addr[te], pr, average='macro', zero_division=0))
    if not f1w_per_seed:
        return None
    return (float(np.mean(f1w_per_seed)), float(np.std(f1w_per_seed)),
            float(np.mean(f1m_per_seed)), float(np.std(f1m_per_seed)))


print('Building feature matrices for Mating...', flush=True)
build_global = build_voc_features(hdb_nca.astype(np.int32))
build_pcctx = build_voc_features(pc_labels)
assert build_global is not None and build_pcctx is not None
X_tok_g, X_mel_g, X_emb_g, y_addr_g, y_em_g, files_g = build_global
X_tok_p, X_mel_p, X_emb_p, y_addr_p, y_em_p, files_p = build_pcctx

# Sanity: filenames должны совпадать (один и тот же фильтр)
assert (files_g == files_p).all()
print(f'  vocs before class filter: {len(y_addr_g)}', flush=True)
print(f'  unique addressees before filter: {len(set(y_addr_g.tolist()))}', flush=True)

# Применяем фильтр >=30 vocs/addressee
X_tok_g_f, y_addr_f, y_em_f = filter_min_per_class(X_tok_g, y_addr_g, y_em_g)
X_tok_p_f, _, _ = filter_min_per_class(X_tok_p, y_addr_g, y_em_g)
X_mel_f, _, _ = filter_min_per_class(X_mel_g, y_addr_g, y_em_g)
X_emb_f, _, _ = filter_min_per_class(X_emb_g, y_addr_g, y_em_g)
print(f'  vocs after >=30/class filter: {len(y_addr_f)}', flush=True)
print(f'  unique addressees after filter: {len(set(y_addr_f.tolist()))}', flush=True)
print(f'  unique emitters: {len(set(y_em_f.tolist()))}', flush=True)
print(f'  random-chance F1 ≈ {1/len(set(y_addr_f.tolist())):.3f}', flush=True)


print('\n=== Stratified split (random, within-bat OK) ===', flush=True)
results = []
configs = [
    ('Bag-of-syllables GLOBAL (|V|=11)', X_tok_g_f),
    ('Bag-of-syllables PER-CONTEXT (|V|=15)', X_tok_p_f),
    ('Pooled UMAP-8D (no tokens)', X_emb_f),
    ('Pooled raw mel (no tokens)', X_mel_f),
]
for name, X in configs:
    f1w, f1w_sd, f1m, f1m_sd = stratified_eval(X, y_addr_f)
    print(f'  {name:42s}: f1w={f1w:.3f} ± {f1w_sd:.3f}, f1m={f1m:.3f} ± {f1m_sd:.3f}',
          flush=True)
    results.append({'split': 'stratified', 'method': name,
                    'f1_w': f1w, 'f1_w_std': f1w_sd,
                    'f1_m': f1m, 'f1_m_std': f1m_sd})


print('\n=== Cross-bat split (split by emitter, 30% test) ===', flush=True)
for name, X in configs:
    out = crossbat_eval(X, y_addr_f, y_em_f)
    if out is None:
        print(f'  {name:42s}: SKIP (no test data)', flush=True)
        continue
    f1w, f1w_sd, f1m, f1m_sd = out
    print(f'  {name:42s}: f1w={f1w:.3f} ± {f1w_sd:.3f}, f1m={f1m:.3f} ± {f1m_sd:.3f}',
          flush=True)
    results.append({'split': 'cross-bat', 'method': name,
                    'f1_w': f1w, 'f1_w_std': f1w_sd,
                    'f1_m': f1m, 'f1_m_std': f1m_sd})


df = pd.DataFrame(results)
out_csv = Path('docs/thesis/figures/addressee_within_mating.csv')
df.to_csv(out_csv, index=False)
print(f'\nSaved: {out_csv}', flush=True)
print('\n=== Summary ===')
print(df.to_string(index=False))
