"""Cross-species transfer test of the THESIS method on marmosets.

Mirrors the bat main experiment (scripts/per_class_f1_chart_uniform.py) exactly,
transferred to the InfantMarmosetsVox corpus:

  unit            = mel frame (32-D)           [bat: segment]
  sequence        = call (frames of one call)  [bat: vocalization]
  grouping label  = call-type (9 working)      [bat: behavioural context]
  cross-individual= caller (10), 7 train/3 test [bat: 30/11 emitters]

Proposed: per-call-type DP-GMM (n_components=15, full cov, Dirichlet-process
prior 0.1) on UMAP-8D + max-likelihood classification with UNIFORM prior.
Baseline: global k-means dictionary on UMAP-8D + RandomForest on per-call
bag-of-tokens features (the Assom-style global pipeline).

Metric: macro F1 over the 9 call-types, 5 cross-caller seeds, paired 95% CI.
If per-call-type > global, the per-context tokenization paradigm transfers
beyond bats — empirical (not only theoretical) evidence of generality.

Output: conference/results/marmoset_percontext_transfer.csv
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib, time
import umap
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

CACHE = Path('/Volumes/T7/cache/assom_paper_repro/marmoset_melbeats_feat.joblib')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
MIN_CALLS = 50            # keep call-types with >= 50 calls
N_TEST_CALLERS = 3        # of 10 -> ~30% held out (bat: 11/41)
N_SEEDS = 5
UMAP_FIT_N = 120_000      # subsample frames for UMAP fit (transform all)
FIT_CAP = 40_000          # cap training frames per call-type for DP-GMM fit
KM_V = 30                 # global dictionary size (baseline)
KM_V_BIG = 150            # matched-capacity global baseline (fairness check)


def ci95(diffs):
    d = np.asarray([x for x in diffs if x == x], float)
    if len(d) < 2: return (float(d.mean()) if len(d) else np.nan, np.nan, np.nan)
    m = d.mean(); h = 2.776 * d.std(ddof=1) / np.sqrt(len(d))  # t(4,.975)
    return m, m - h, m + h


print('Loading marmoset cache...', flush=True)
d = joblib.load(CACHE)
mel_list = d['mel_list']; ct = np.array(d['ct']); cl = np.array(d['cl'])
n_calls = len(mel_list)

# working call-types
cnt = Counter(ct.tolist())
WORK = sorted([t for t, c in cnt.items() if c >= MIN_CALLS])
print(f'  call-types kept (>= {MIN_CALLS} calls): {WORK}', flush=True)
keep_call = np.array([t in WORK for t in ct])
idx_calls = np.where(keep_call)[0]
print(f'  calls kept: {len(idx_calls)} / {n_calls}', flush=True)

# stack frames, remember per-frame call id and per-call labels
frames, frame_call, call_ct, call_cl, call_slices = [], [], [], [], []
pos = 0
for new_i, i in enumerate(idx_calls):
    m = mel_list[i].astype(np.float32)
    nfr = m.shape[0]
    frames.append(m)
    frame_call.append(np.full(nfr, new_i, dtype=np.int32))
    call_ct.append(int(ct[i])); call_cl.append(int(cl[i]))
    call_slices.append((pos, pos + nfr)); pos += nfr
X = np.vstack(frames).astype(np.float32)
frame_call = np.concatenate(frame_call)
call_ct = np.array(call_ct); call_cl = np.array(call_cl)
print(f'  frames: {X.shape}, calls: {len(call_ct)}', flush=True)

# ---- UMAP-8D (unsupervised; fit on subsample, transform all) ----
print('Fitting UMAP-8D (subsample) ...', flush=True)
t0 = time.time()
rng = np.random.default_rng(42)
fit_idx = rng.choice(len(X), size=min(UMAP_FIT_N, len(X)), replace=False)
reducer = umap.UMAP(n_components=8, n_neighbors=30, min_dist=1.0,
                    random_state=42, verbose=False).fit(X[fit_idx])
emb = reducer.transform(X).astype(np.float64)
print(f'  UMAP done in {time.time()-t0:.0f}s -> emb {emb.shape}', flush=True)

all_callers = sorted(set(call_cl.tolist()))
log_prior = -np.log(len(WORK))   # uniform prior over working call-types


def per_call_frames(call_id):
    a, b = call_slices[call_id]; return emb[a:b]


def call_features(tokens_of_frames, V):
    """per-call bag-of-tokens + simple aggregates (no leaky context features)."""
    n = len(tokens_of_frames)
    hist = np.bincount(tokens_of_frames, minlength=V).astype(float)
    p = hist / max(hist.sum(), 1)
    ent = -np.sum(p[p > 0] * np.log(p[p > 0]))
    uniq = (hist > 0).sum() / V
    return np.concatenate([p, [np.log1p(n), uniq, ent]])


rows = []
for seed in range(N_SEEDS):
    t0 = time.time()
    rs = np.random.default_rng(seed)
    ca = np.array(all_callers); rs.shuffle(ca)
    test_callers = set(ca[:N_TEST_CALLERS].tolist())
    train_call = np.array([c not in test_callers for c in call_cl])
    test_call = ~train_call
    # frame-level train mask
    train_frame = train_call[frame_call]

    # ===== Proposed: per-call-type DP-GMM + max-likelihood =====
    toks = {}
    for c in WORK:
        m = train_frame & (call_ct[frame_call] == c)
        fi = np.where(m)[0]
        if len(fi) < 50: continue
        if len(fi) > FIT_CAP:
            fi = rs.choice(fi, size=FIT_CAP, replace=False)
        toks[c] = BayesianGaussianMixture(
            n_components=15, weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.1, covariance_type='full',
            max_iter=120, random_state=seed).fit(emb[fi])
    yt, yp = [], []
    for ci_ in np.where(test_call)[0]:
        Xc = per_call_frames(ci_)
        if len(Xc) == 0: continue
        best_c, best = None, -np.inf
        for c, t in toks.items():
            s = t.score_samples(Xc).sum() + log_prior
            if s > best: best, best_c = s, c
        yt.append(call_ct[ci_]); yp.append(best_c)
    f1_pc = f1_score(yt, yp, average='macro', labels=WORK, zero_division=0)

    # ===== Baseline: global k-means dict + RF on bag-of-tokens =====
    def global_baseline(V):
        fi = np.where(train_frame)[0]
        if len(fi) > 200_000:
            fi = rs.choice(fi, size=200_000, replace=False)
        km = KMeans(n_clusters=V, n_init=4, random_state=seed).fit(emb[fi])
        tok_all = km.predict(emb)
        Xtr, ytr, Xte, yte = [], [], [], []
        for ci_ in np.where(train_call)[0]:
            a, b = call_slices[ci_]
            Xtr.append(call_features(tok_all[a:b], V)); ytr.append(call_ct[ci_])
        for ci_ in np.where(test_call)[0]:
            a, b = call_slices[ci_]
            Xte.append(call_features(tok_all[a:b], V)); yte.append(call_ct[ci_])
        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                    random_state=seed, n_jobs=-1).fit(Xtr, ytr)
        pred = rf.predict(Xte)
        return f1_score(yte, pred, average='macro', labels=WORK, zero_division=0)

    f1_glob = global_baseline(KM_V)
    f1_glob_big = global_baseline(KM_V_BIG)

    rows.append({'seed': seed, 'n_test_calls': len(yt),
                 'f1_percontext': f1_pc, 'f1_global_v30': f1_glob,
                 'f1_global_v150': f1_glob_big})
    print(f'  seed {seed}: per-context={f1_pc:.3f}  global(V30)={f1_glob:.3f}  '
          f'global(V150)={f1_glob_big:.3f}  ({time.time()-t0:.0f}s)', flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / 'marmoset_percontext_transfer.csv', index=False)
print('\n=== Marmoset transfer summary (5 cross-caller seeds) ===', flush=True)
pc = df['f1_percontext']; g30 = df['f1_global_v30']; g150 = df['f1_global_v150']
print(f'  per-call-type DP-GMM (proposed): {pc.mean():.3f} ± {pc.std(ddof=1):.3f}')
print(f'  global k-means V=30 + RF:        {g30.mean():.3f} ± {g30.std(ddof=1):.3f}')
print(f'  global k-means V=150 + RF:       {g150.mean():.3f} ± {g150.std(ddof=1):.3f}')
best_glob = g30 if g30.mean() >= g150.mean() else g150
m, lo, hi = ci95((pc - best_glob).values)
which = 'V30' if g30.mean() >= g150.mean() else 'V150'
print(f'  delta (proposed - best global[{which}]): {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]')
print(f'  -> {"TRANSFERS (CI excludes 0)" if lo > 0 else "not significant"}', flush=True)
print(f'\nSaved: {OUT}/marmoset_percontext_transfer.csv', flush=True)
