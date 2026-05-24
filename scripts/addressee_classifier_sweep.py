"""Addressee-prediction sweep: разные классификаторы на одних и тех же признаках.

Задача: внутри Mating-контекста предсказать addressee (7 классов после
фильтра >=74 vocs/class). Цель — выяснить, является ли отрицательный
результат RF-300 артефактом классификатора или содержательным фактом.

Классификаторы:
  RF-300        — Random Forest, n_estimators=300, class_weight=balanced
  LogReg        — Logistic Regression, L2, class_weight=balanced
  SVM-RBF       — Support Vector Machine, RBF kernel, class_weight=balanced
  GBC           — Gradient Boosting Classifier, n_estimators=200
  MLP           — Multi-Layer Perceptron, hidden=(64,32), early stopping
  DP-GMM ML     — per-addressee DP-GMM + max-likelihood (как для контекста
                   в основном эксперименте, но цель = адресат)

Признаки:
  Bag-of-syllables GLOBAL (V=11)
  Bag-of-syllables PER-CONTEXT Mating (V=15)
  Pooled UMAP-8D (mean+std)
  Pooled raw mel (mean+std, 672D)

Протокол: 5 сидов × 5-fold stratified-by-addressee.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib, time
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
MATING_CTX = 2
MIN_VOCS_PER_ADDR = 74  # → 7 классов, 1179 вокализаций
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
print('Per-context k-means on Mating...', flush=True)
mating_mask = ctx == MATING_CTX
km = KMeans(n_clusters=15, n_init=10, random_state=0).fit(emb[mating_mask])
pc_labels = np.full(len(emb), -1, dtype=np.int32)
pc_labels[mating_mask] = km.labels_


def build_voc_features(token_arr):
    mask = (ctx == MATING_CTX) & (em_arr != 0) & (addr_arr != 0) & (token_arr >= 0)
    if mask.sum() == 0: return None
    V = int(token_arr[mask].max()) + 1
    sub = pd.DataFrame({
        'idx': np.arange(len(seg_df))[mask],
        'file': file_arr[mask], 'pos': pos_arr[mask],
        'tok': token_arr[mask], 'em': em_arr[mask], 'addr': addr_arr[mask],
    })
    X_tok, X_mel, X_emb, y_addr, y_em, files, seg_groups = [], [], [], [], [], [], []
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
        seg_groups.append(idxs)
    return (np.array(X_tok, dtype=np.float32), np.array(X_mel, dtype=np.float32),
            np.array(X_emb, dtype=np.float32), np.array(y_addr), np.array(y_em),
            np.array(files), seg_groups)


print('Build feature matrices...', flush=True)
g_glob = build_voc_features(hdb_nca.astype(np.int32))
g_pc = build_voc_features(pc_labels)
X_tok_g, X_mel_, X_emb_, y_addr, y_em, files, seg_groups = g_glob
X_tok_p, _, _, _, _, _, _ = g_pc

# Filter >= 74 vocs/addressee
cnt = Counter(y_addr.tolist())
keep = np.array([cnt[a] >= MIN_VOCS_PER_ADDR for a in y_addr])
X_tok_g = X_tok_g[keep]; X_tok_p = X_tok_p[keep]
X_mel_ = X_mel_[keep]; X_emb_ = X_emb_[keep]
y_addr = y_addr[keep]; y_em = y_em[keep]; files = files[keep]
seg_groups = [s for s, k in zip(seg_groups, keep) if k]
n_classes = len(set(y_addr.tolist()))
print(f'  vocs={len(y_addr)}, classes={n_classes}, '
      f'random F1≈{1/n_classes:.3f}', flush=True)


CLASSIFIERS = {
    'RF-300': lambda seed: RandomForestClassifier(
        n_estimators=300, class_weight='balanced', random_state=seed, n_jobs=-1),
    'LogReg': lambda seed: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight='balanced',
                           random_state=seed)),
    'SVM-RBF': lambda seed: make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', class_weight='balanced', random_state=seed)),
    'GBC': lambda seed: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, random_state=seed),
    'MLP': lambda seed: make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                      early_stopping=True, random_state=seed)),
}


def stratified_eval(X, y, make_clf, n_folds=N_FOLDS, n_seeds=N_SEEDS):
    f1w_per_seed, f1m_per_seed = [], []
    for s in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
        f1w, f1m = [], []
        for tr, te in skf.split(X, y):
            clf = make_clf(s).fit(X[tr], y[tr])
            pr = clf.predict(X[te])
            f1w.append(f1_score(y[te], pr, average='weighted', zero_division=0))
            f1m.append(f1_score(y[te], pr, average='macro', zero_division=0))
        f1w_per_seed.append(np.mean(f1w)); f1m_per_seed.append(np.mean(f1m))
    return (float(np.mean(f1w_per_seed)), float(np.std(f1w_per_seed)),
            float(np.mean(f1m_per_seed)), float(np.std(f1m_per_seed)))


def dpgmm_ml_eval(seg_groups, y_addr, n_folds=N_FOLDS, n_seeds=N_SEEDS,
                  cov_type='diag'):
    """Per-addressee DP-GMM на UMAP-8D-эмбеддингах сегментов + max-likelihood
    классификация по правилу, как в основном эксперименте, но цель = addressee.

    Равномерный приор по классам.
    """
    addrs_sorted = sorted(set(y_addr.tolist()))
    log_prior = -np.log(len(addrs_sorted))
    f1w_per_seed, f1m_per_seed = [], []
    for s in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
        idx_arr = np.arange(len(y_addr))
        f1w, f1m = [], []
        for tr, te in skf.split(idx_arr, y_addr):
            # Fit DP-GMM на UMAP-8D сегментах каждого addressee из train
            toks = {}
            for a in addrs_sorted:
                seg_ids_a = []
                for i in tr:
                    if y_addr[i] == a:
                        seg_ids_a.extend(seg_groups[i])
                if len(seg_ids_a) < 30: continue
                X_train = emb[seg_ids_a]
                try:
                    toks[a] = BayesianGaussianMixture(
                        n_components=15,
                        weight_concentration_prior_type='dirichlet_process',
                        weight_concentration_prior=0.1, covariance_type=cov_type,
                        max_iter=150, random_state=s
                    ).fit(X_train)
                except Exception:
                    pass
            # Predict
            yt, yp = [], []
            for i in te:
                X_test = emb[seg_groups[i]]
                if len(X_test) == 0: continue
                best, bs = None, -np.inf
                for a, t in toks.items():
                    ll = t.score_samples(X_test).sum() + log_prior
                    if ll > bs: bs = ll; best = a
                if best is None: continue
                yt.append(y_addr[i]); yp.append(best)
            yt_a, yp_a = np.array(yt), np.array(yp)
            f1w.append(f1_score(yt_a, yp_a, average='weighted', zero_division=0))
            f1m.append(f1_score(yt_a, yp_a, average='macro', zero_division=0))
        f1w_per_seed.append(np.mean(f1w)); f1m_per_seed.append(np.mean(f1m))
    return (float(np.mean(f1w_per_seed)), float(np.std(f1w_per_seed)),
            float(np.mean(f1m_per_seed)), float(np.std(f1m_per_seed)))


feat_sets = [
    ('Bag-of-tokens GLOBAL (|V|=11)', X_tok_g),
    ('Bag-of-tokens PER-CONTEXT (|V|=15)', X_tok_p),
    ('Pooled UMAP-8D', X_emb_),
    ('Pooled raw mel', X_mel_),
]

results = []
print('\n=== Sweep: classifier × features (5 seeds × 5-fold stratified) ===',
      flush=True)
for clf_name, make_clf in CLASSIFIERS.items():
    for feat_name, X in feat_sets:
        t0 = time.time()
        f1w, f1w_sd, f1m, f1m_sd = stratified_eval(X, y_addr, make_clf)
        dt = time.time() - t0
        print(f'  {clf_name:10s} | {feat_name:38s}: '
              f'f1w={f1w:.3f}±{f1w_sd:.3f}, f1m={f1m:.3f}±{f1m_sd:.3f}  ({dt:.0f}s)',
              flush=True)
        results.append({
            'classifier': clf_name, 'features': feat_name,
            'f1_w': f1w, 'f1_w_std': f1w_sd, 'f1_m': f1m, 'f1_m_std': f1m_sd,
            'time_s': dt,
        })

# DP-GMM ML — отдельно
print('\n=== Per-addressee DP-GMM + max-likelihood ===', flush=True)
for cov in ['diag', 'spherical']:
    t0 = time.time()
    f1w, f1w_sd, f1m, f1m_sd = dpgmm_ml_eval(seg_groups, y_addr, cov_type=cov)
    dt = time.time() - t0
    name = f'DP-GMM ML ({cov})'
    feat_label = 'UMAP-8D сегментов'
    print(f'  {name:18s} | {feat_label:38s}: '
          f'f1w={f1w:.3f}±{f1w_sd:.3f}, f1m={f1m:.3f}±{f1m_sd:.3f}  ({dt:.0f}s)',
          flush=True)
    results.append({
        'classifier': name, 'features': feat_label,
        'f1_w': f1w, 'f1_w_std': f1w_sd, 'f1_m': f1m, 'f1_m_std': f1m_sd,
        'time_s': dt,
    })

df = pd.DataFrame(results)
out = Path('docs/thesis/figures/addressee_classifier_sweep.csv')
df.to_csv(out, index=False)
print(f'\nSaved: {out}', flush=True)
print('\n=== Best macro F1 ===')
print(df.sort_values('f1_m', ascending=False).head(10).to_string(index=False))
