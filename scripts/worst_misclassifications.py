"""5 худших ошибочно классифицированных вокализаций — рекомендуемая конфигурация.

Для подготовки к Q&A: на предзащите гарантированно спросят
"покажите 5 худших примеров и объясните". Скрипт находит вокализации,
которые модель уверенно отнесла к НЕ-истинному контексту (max log-likelihood
для неверного контекста существенно выше, чем для истинного).

Метод: рекомендуемая конфигурация (per-context DP-GMM full на UMAP-8D,
равн.\ приор). Для каждой test-вокализации:
  - true context = c_true
  - predicted   = argmax_c (sum log p_c(x_i) + log p(c))
  - margin      = log p_{c_pred}(X) - log p_{c_true}(X)
  - "худший" = большой положительный margin при c_pred != c_true

Сохраняет 5 примеров с метаданными для устного ответа.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib
from sklearn.mixture import BayesianGaussianMixture

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAME = {2: 'Mating', 3: 'Feeding', 4: 'Fighting', 5: 'Grooming',
            6: 'Isolation', 7: 'Kissing', 9: 'Biting', 10: 'Threat'}
# Внимание: в seg_df коды контекстов могут отличаться от display name.
# Используем те же, что в per_class_f1_chart_uniform.py:
CTX_NAME = {2: 'Biting', 3: 'Feeding', 4: 'Fighting', 5: 'Grooming',
            6: 'Isolation', 7: 'Kissing', 9: 'Mating', 10: 'Threat'}

print('Loading state and embeddings...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
ctx_arr = seg_df['context'].to_numpy()

# Build vocs (как в per_class_f1_chart_uniform.py — seed=0 для устойчивого примера)
vocs = []
for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
    seg_ids = g.index.to_list()
    if not seg_ids: continue
    dom_em_signed = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
    if dom_em_signed == 0: continue
    dom_em_abs = abs(dom_em_signed)
    dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
    if dom_ctx not in HP1_CTX: continue
    vocs.append({'fname': fname, 'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em_abs})

all_bats = sorted(set(v['em'] for v in vocs))
print(f'  vocs: {len(vocs)}, bats: {len(all_bats)}', flush=True)

# seed=0 split
rng = np.random.default_rng(0); ba = np.array(all_bats); rng.shuffle(ba)
test_b = set(ba[:11].tolist())
train_v = [v for v in vocs if v['em'] not in test_b]
test_v = [v for v in vocs if v['em'] in test_b]
print(f'  seed=0: train={len(train_v)}, test={len(test_v)}', flush=True)

train_mask = np.zeros(len(emb), dtype=bool)
for v in train_v: train_mask[v['seg_ids']] = True

log_prior = {c: -np.log(len(HP1_CTX)) for c in HP1_CTX}

print('Fitting 8 per-context DP-GMM...', flush=True)
toks = {}
for c in HP1_CTX:
    m = train_mask & (ctx_arr == c)
    if m.sum() < 30: continue
    toks[c] = BayesianGaussianMixture(
        n_components=15,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.1, covariance_type='full',
        max_iter=150, random_state=0
    ).fit(emb[m])
print(f'  fitted: {list(toks.keys())}', flush=True)

print('\nScoring test set + finding worst misclassifications...', flush=True)
records = []
for v in test_v:
    X = emb[v['seg_ids']]
    if len(X) == 0: continue
    n_seg = len(X)
    lls = {}
    for c, t in toks.items():
        lls[c] = t.score_samples(X).sum() + log_prior[c]
    c_pred = max(lls, key=lls.get)
    c_true = v['ctx']
    if c_pred == c_true: continue
    # Margin: насколько уверенно предсказан неверный контекст
    margin = lls[c_pred] - lls[c_true]
    # Per-segment margin (нормированный) — чтобы длинные voc не доминировали
    margin_per_seg = margin / max(n_seg, 1)
    records.append({
        'file': v['fname'],
        'emitter': v['em'],
        'n_segments': n_seg,
        'true_context': CTX_NAME[c_true],
        'predicted_context': CTX_NAME[c_pred],
        'margin_total_logp': float(margin),
        'margin_per_segment': float(margin_per_seg),
        'logp_true': float(lls[c_true]),
        'logp_pred': float(lls[c_pred]),
    })

df_err = pd.DataFrame(records).sort_values('margin_per_segment', ascending=False)
print(f'  Total misclassifications: {len(df_err)} / {len(test_v)} '
      f'({100*len(df_err)/len(test_v):.1f}%)', flush=True)

# Top-5
top5 = df_err.head(5)
out = Path('docs/thesis/figures/worst_misclassifications_seed0.csv')
df_err.to_csv(out, index=False)
print(f'\nSaved full list: {out}', flush=True)

print(f'\n=== TOP-5 наиболее уверенных ошибочных предсказаний ===\n', flush=True)
for i, r in top5.iterrows():
    print(f'#{list(top5.index).index(i)+1}. {r["file"]}', flush=True)
    print(f'    Истинный контекст: {r["true_context"]}, '
          f'предсказание: {r["predicted_context"]}', flush=True)
    print(f'    {r["n_segments"]} сегментов, '
          f'log-likelihood разрыв {r["margin_per_segment"]:.2f} на сегмент '
          f'({r["margin_total_logp"]:.1f} суммарно)', flush=True)
    print(f'    Эмиттер: {r["emitter"]}\n', flush=True)

# Confusion patterns among top-K errors
print('=== Топ-50 самых "уверенных" ошибок: распределение по парам ===', flush=True)
top50 = df_err.head(50)
pairs = Counter([(r['true_context'], r['predicted_context']) for _, r in top50.iterrows()])
for (t, p), n in pairs.most_common(8):
    print(f'  {t:10s} -> {p:10s}: {n}', flush=True)
