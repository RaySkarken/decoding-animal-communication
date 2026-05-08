"""Re-run the main per-context experiment with extended metrics.

For each of 5 emitter-split seeds computes, on the held-out emitters:

    - weighted F1 (mirrors Table 3.1 main result)
    - macro F1 (each of 8 contexts contributes equally)
    - Matthews correlation coefficient (MCC, dysbalance-robust)
    - per-class F1 per context

for:

    - Assom baseline (HDBSCAN+NCA global + RF on bag-of-syllables)
    - per-context DP-GMM (empirical prior)
    - per-context DP-GMM (uniform prior)

Saves:
    docs/thesis/figures/extended_metrics_5seeds.csv     — one row per (method, seed)
    docs/thesis/figures/extended_metrics_per_class.csv  — one row per (method, seed, context)
    docs/thesis/figures/extended_metrics_summary.json   — aggregates per method

Requires the cached ablation state from the Assom reproduction pipeline:
    /Volumes/T7/cache/assom_paper_repro/ablation_state.joblib
If the external SSD is not mounted, this script will exit with a clear message.
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings('ignore')

from src.per_context_tokenizer import PerContextFamily  # noqa: E402

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAME = {2: 'Biting', 3: 'Feeding', 4: 'Fighting', 5: 'Grooming',
            6: 'Isolation', 7: 'Kissing', 9: 'Mating', 10: 'Threat'}
N_SEEDS = 5
TEST_EMITTERS_PER_SEED = 11
OUT_DIR = REPO / 'docs' / 'thesis' / 'figures'


def main() -> int:
    state_file = CKPT / 'ablation_state.joblib'
    if not state_file.exists():
        print(f'ERROR: {state_file} not found — mount the T7 SSD and retry.',
              file=sys.stderr)
        return 2

    st = joblib.load(state_file)
    emb = st['embedding']
    seg_df = st['seg_df']
    ctx_arr = seg_df['context'].to_numpy()
    hdb_nca = st['hdb_nca_labels']
    V_GLOBAL = int(hdb_nca.max()) + 1

    vocs = []
    for _, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
        seg_ids = g.index.to_list()
        if not seg_ids:
            continue
        dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
        if dom_ctx not in HP1_CTX:
            continue
        dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
        vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em})
    all_emitters = sorted(set(v['em'] for v in vocs))
    print(f'{len(vocs)} vocalizations over {len(all_emitters)} emitters in 8 contexts.')

    def emitter_split(seed: int):
        rng = np.random.default_rng(seed)
        em_arr = np.array(all_emitters)
        rng.shuffle(em_arr)
        test_em = set(em_arr[:TEST_EMITTERS_PER_SEED].tolist())
        return ([v for v in vocs if v['em'] not in test_em],
                [v for v in vocs if v['em'] in test_em])

    def bag_of_syll(seq, V=V_GLOBAL):
        c = Counter(seq); n = max(len(seq), 1)
        bos = np.zeros(V, dtype=np.float32)
        for k, cnt in c.items():
            if 0 <= k < V:
                bos[k] = cnt / n
        probs = np.array(list(c.values()), dtype=np.float32) / n
        ent = float(-(probs * np.log(probs + 1e-12)).sum())
        richness = len(c) / n
        rep = max(c.values()) / n if c else 0.0
        return np.concatenate([bos, [n, richness, ent, rep]]).astype(np.float32)

    def baseline_predict(train_vocs, test_vocs, seed):
        Xt, yt = [], []
        for v in train_vocs:
            labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
            if labs:
                Xt.append(bag_of_syll(labs)); yt.append(v['ctx'])
        Xe, ye = [], []
        for v in test_vocs:
            labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
            if labs:
                Xe.append(bag_of_syll(labs)); ye.append(v['ctx'])
        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                     random_state=seed, n_jobs=-1).fit(Xt, yt)
        return np.array(ye), rf.predict(Xe)

    VARIANT_KWARGS = {
        'dpgmm':   dict(n_components=15, max_iter=150),
        'hdbscan': dict(),
        'kmeans':  dict(n_clusters=15),
    }

    def fit_family_and_predict(train_vocs, test_vocs, seed, variant, prior):
        tr_seg = np.concatenate([v['seg_ids'] for v in train_vocs])
        fam = PerContextFamily(
            variant=variant, prior=prior,
            tokenizer_kwargs=VARIANT_KWARGS[variant])
        fam.fit(emb[tr_seg], ctx_arr[tr_seg], HP1_CTX, seed=seed,
                prior_counts=dict(Counter(v['ctx'] for v in train_vocs)))
        y_true, y_pred = [], []
        for v in test_vocs:
            X_seq = emb[v['seg_ids']]
            if len(X_seq) == 0:
                continue
            y_true.append(v['ctx'])
            y_pred.append(fam.predict_context(X_seq))
        return np.array(y_true), np.array(y_pred)

    rows, rows_perclass = [], []
    for seed in range(N_SEEDS):
        tr, te = emitter_split(seed)
        runs = [
            ('baseline',            *baseline_predict(tr, te, seed)),
            ('dpgmm_empirical',     *fit_family_and_predict(tr, te, seed, 'dpgmm',   'empirical')),
            ('dpgmm_uniform',       *fit_family_and_predict(tr, te, seed, 'dpgmm',   'uniform')),
            ('hdbscan_empirical',   *fit_family_and_predict(tr, te, seed, 'hdbscan', 'empirical')),
            ('hdbscan_uniform',     *fit_family_and_predict(tr, te, seed, 'hdbscan', 'uniform')),
            ('kmeans_empirical',    *fit_family_and_predict(tr, te, seed, 'kmeans',  'empirical')),
            ('kmeans_uniform',      *fit_family_and_predict(tr, te, seed, 'kmeans',  'uniform')),
        ]
        for method, yt, yp in runs:
            weighted = float(f1_score(yt, yp, average='weighted', labels=HP1_CTX, zero_division=0))
            macro = float(f1_score(yt, yp, average='macro', labels=HP1_CTX, zero_division=0))
            mcc = float(matthews_corrcoef(yt, yp)) if len(set(yt)) > 1 else float('nan')
            rows.append({'method': method, 'seed': seed, 'n_test': len(yt),
                         'weighted_f1': weighted, 'macro_f1': macro, 'mcc': mcc})
            per_cls = f1_score(yt, yp, labels=HP1_CTX, average=None, zero_division=0)
            for c, f in zip(HP1_CTX, per_cls):
                rows_perclass.append({'method': method, 'seed': seed,
                                      'context': CTX_NAME[c],
                                      'n_true': int((yt == c).sum()),
                                      'f1': float(f)})
            print(f'seed={seed} {method:20s}  weighted={weighted:.3f}  '
                  f'macro={macro:.3f}  mcc={mcc:.3f}')

    df = pd.DataFrame(rows)
    df_pc = pd.DataFrame(rows_perclass)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / 'extended_metrics_5seeds.csv', index=False)
    df_pc.to_csv(OUT_DIR / 'extended_metrics_per_class.csv', index=False)

    summary = {}
    for method, g in df.groupby('method'):
        summary[method] = {
            'weighted_f1_mean': float(g['weighted_f1'].mean()),
            'weighted_f1_std':  float(g['weighted_f1'].std()),
            'macro_f1_mean':    float(g['macro_f1'].mean()),
            'macro_f1_std':     float(g['macro_f1'].std()),
            'mcc_mean':         float(g['mcc'].mean()),
            'mcc_std':          float(g['mcc'].std()),
        }
    summary['per_class'] = (df_pc.groupby(['method', 'context'])['f1']
                            .agg(['mean', 'std']).reset_index()
                            .to_dict(orient='records'))
    with (OUT_DIR / 'extended_metrics_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    print('\nAggregates:')
    for method, s in summary.items():
        if method == 'per_class':
            continue
        print(f"  {method:20s}  weighted={s['weighted_f1_mean']:.3f}±{s['weighted_f1_std']:.3f}   "
              f"macro={s['macro_f1_mean']:.3f}±{s['macro_f1_std']:.3f}   "
              f"MCC={s['mcc_mean']:.3f}±{s['mcc_std']:.3f}")
    print(f'\nSaved to {OUT_DIR}/extended_metrics_*.csv/.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
