"""Within-context caller ID: внутри каждого контекста предсказать эмиттера.

Это та задача, где per-context словари могут показать преимущество без leakage:
целевая переменная — эмиттер, а не контекст; значит, одинаковая для всех вариантов
токенизации, и трюк с диапазонами идентификаторов не помогает.

Протокол: closed-set, разбиение по вокализациям (80/20) внутри каждого контекста,
стратифицированное по эмиттеру; RF на агрегатах последовательности токенов.

Сравниваются:
  PC-orig         — стандартный per-context k-means (V=15 на контекст)
  PC-share        — общий диапазон {0..14} для всех контекстов
  G (HDB+NCA)     — глобальный V=11
  KM-15           — глобальный k-means V=15
  raw mel agg     — RF без токенизации (контроль)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, time
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAMES = {2: 'Mating', 3: 'Feeding', 4: 'Fighting', 5: 'Isolation',
             6: 'Biting', 7: 'Threat', 9: 'Grooming', 10: 'Kissing'}

print('Loading state...')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
mel = st['tf_specs'].reshape(len(seg_df), -1).astype(np.float32)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
hdb_nca = np.load(CACHE / 'hdb_nca_labels_152k_21x32.npy')
ctx = seg_df['context'].to_numpy()
em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy()
pos_arr = seg_df['pos_segment'].to_numpy()


def per_context_kmeans_orig(emb, ctx_arr, K=15):
    labels = np.full(len(emb), -1, dtype=np.int32); offset = 0
    for c in HP1_CTX:
        mc = ctx_arr == c
        if mc.sum() < 30: continue
        km = KMeans(n_clusters=min(K, mc.sum()//5), n_init=10, random_state=0).fit(emb[mc])
        labels[mc] = km.labels_ + offset; offset += K
    return labels


def per_context_kmeans_shared(emb, ctx_arr, K=15):
    labels = np.full(len(emb), -1, dtype=np.int32)
    for c in HP1_CTX:
        mc = ctx_arr == c
        if mc.sum() < 30: continue
        km = KMeans(n_clusters=min(K, mc.sum()//5), n_init=10, random_state=0).fit(emb[mc])
        labels[mc] = km.labels_
    return labels


print('Computing tokenizations...')
toks = {
    'G': hdb_nca.astype(np.int32),
    'KM15': KMeans(n_clusters=15, n_init=10, random_state=0).fit_predict(emb).astype(np.int32),
    'PC-orig': per_context_kmeans_orig(emb, ctx, K=15),
    'PC-share': per_context_kmeans_shared(emb, ctx, K=15),
}

# Vocalizations: для каждой вокализации внутри контекста c считаем агрегат токенов и эмиттер
def voc_features_for_context(token_arr, c):
    """Возвращает (X_tok, X_mel, y_em, files) — каждая строка = одна вокализация контекста c."""
    mask = (ctx == c) & (em_arr != 0) & (token_arr >= 0)
    df = pd.DataFrame({'idx': np.arange(len(seg_df))[mask],
                       'file': file_arr[mask], 'pos': pos_arr[mask],
                       'tok': token_arr[mask], 'em': em_arr[mask]})
    V = int(token_arr[mask].max()) + 1
    X_tok, X_mel_, y_em, files = [], [], [], []
    for fname, g in df.sort_values('pos').groupby('file', sort=False):
        idxs = g['idx'].to_numpy()
        toks_v = g['tok'].to_numpy()
        # token freq + length + entropy + max_freq + n_unique
        freq = np.bincount(toks_v, minlength=V) / max(len(toks_v), 1)
        ent = -np.sum(freq * np.log(freq + 1e-12))
        feat_tok = np.concatenate([freq, [len(toks_v), ent, freq.max(),
                                            float(np.count_nonzero(freq))]])
        # mel agg
        m = mel[idxs]
        feat_mel = np.concatenate([m.mean(0), m.std(0)])
        X_tok.append(feat_tok); X_mel_.append(feat_mel)
        y_em.append(int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0]))
        files.append(fname)
    return (np.array(X_tok, dtype=np.float32), np.array(X_mel_, dtype=np.float32),
            np.array(y_em), np.array(files))


def run_caller_id(name, X, y_em, n_splits=5):
    """5-fold stratified by emitter, return mean f1_w over folds."""
    # фильтр: классы с < n_splits — выбрасываем
    cnt = Counter(y_em.tolist())
    keep = np.array([cnt[e] >= n_splits for e in y_em])
    X_, y_ = X[keep], y_em[keep]
    if len(set(y_.tolist())) < 2: return None, None, None, len(X_)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    f1s_w, f1s_m = [], []
    for tr, te in skf.split(X_, y_):
        rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                     random_state=0, n_jobs=-1).fit(X_[tr], y_[tr])
        pr = rf.predict(X_[te])
        f1s_w.append(f1_score(y_[te], pr, average='weighted', zero_division=0))
        f1s_m.append(f1_score(y_[te], pr, average='macro', zero_division=0))
    return float(np.mean(f1s_w)), float(np.std(f1s_w)), float(np.mean(f1s_m)), len(X_)


print('\n=== Within-context caller ID ===')
results = []
for c in HP1_CTX:
    cn = CTX_NAMES[c]
    print(f'\nContext {c} ({cn}):', flush=True)
    # Берём токены/мел только этого контекста
    Xs = {}
    y_em_ref = None
    for name, t in toks.items():
        Xt, Xm, ye, _ = voc_features_for_context(t, c)
        Xs[name] = Xt
        if y_em_ref is None:
            y_em_ref = ye; X_mel_ref = Xm
    n_em = len(set(y_em_ref.tolist()))
    print(f'  vocs={len(y_em_ref)}, unique emitters={n_em}', flush=True)
    if n_em < 2:
        print(f'  skip — too few emitters', flush=True); continue

    for name in ['G', 'KM15', 'PC-orig', 'PC-share']:
        f1w, sd, f1m, n_kept = run_caller_id(name, Xs[name], y_em_ref)
        if f1w is None: continue
        print(f'  {name:>10s}: f1w={f1w:.3f} ± {sd:.3f}, f1m={f1m:.3f}  (n={n_kept})', flush=True)
        results.append({'context': cn, 'tokenization': name,
                        'f1_w': f1w, 'f1_w_std': sd, 'f1_m': f1m, 'n': n_kept})
    # raw mel
    f1w, sd, f1m, n_kept = run_caller_id('mel', X_mel_ref, y_em_ref)
    if f1w is not None:
        print(f'  {"raw-mel":>10s}: f1w={f1w:.3f} ± {sd:.3f}, f1m={f1m:.3f}  (n={n_kept})', flush=True)
        results.append({'context': cn, 'tokenization': 'raw-mel',
                        'f1_w': f1w, 'f1_w_std': sd, 'f1_m': f1m, 'n': n_kept})


df = pd.DataFrame(results)
df.to_csv('docs/thesis/figures/caller_id_within_context.csv', index=False)
print('\n=== Summary by tokenization (mean over contexts) ===')
agg = df.groupby('tokenization').agg(f1w_mean=('f1_w', 'mean'),
                                       f1w_std=('f1_w', 'std'),
                                       f1m_mean=('f1_m', 'mean'),
                                       n_ctx=('f1_w', 'count')).reset_index()
print(agg.to_string(index=False))
agg.to_csv('docs/thesis/figures/caller_id_summary.csv', index=False)
print('\nSaved: docs/thesis/figures/caller_id_*.csv')
