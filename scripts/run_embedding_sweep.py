"""Re-run the main per-context experiment on different embedding spaces.

Compares:
    - UMAP-2D  (baseline, matches current thesis results)
    - UMAP-16D (higher-dimensional UMAP on the same mel frontend)
    - BEATs-768D (audio foundation-model embeddings; computed once in
      beats_full_experiment.joblib). Full 768-D with diagonal covariance
      to avoid O(d^2) parameter blow-up.
    - BEATs-UMAP-16D (UMAP-16D on BEATs features, if available; falls back
      to running UMAP here if not cached)

For each embedding computes per-seed extended metrics (weighted F1, macro F1,
MCC, per-class F1) for baseline and six DP-GMM / HDBSCAN / k-means configurations
(empirical + uniform prior).

Writes:
    docs/thesis/figures/embedding_sweep_{5seeds,per_class,summary}.{csv,json}

Requires T7 mounted.
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


def build_vocs(seg_df, emb):
    """Group segments into vocalizations by file_name; keep only HP1 contexts."""
    vocs = []
    for _, g in seg_df.sort_values('pos_segment' if 'pos_segment' in seg_df.columns else g.index.name or seg_df.columns[0]).groupby('file_name', sort=False):
        seg_ids = g.index.to_list()
        if not seg_ids:
            continue
        dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
        if dom_ctx not in HP1_CTX:
            continue
        dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
        vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em})
    return vocs


def run_sweep_on(name: str, emb: np.ndarray, seg_df: pd.DataFrame,
                 hdb_nca: np.ndarray, V_global: int, covariance_type: str = 'full'):
    ctx_arr = seg_df['context'].to_numpy()

    # group into vocalizations (must not depend on pos_segment in simple case)
    if 'pos_segment' in seg_df.columns:
        ordered = seg_df.sort_values('pos_segment')
    else:
        ordered = seg_df
    vocs = []
    for _, g in ordered.groupby('file_name', sort=False):
        seg_ids = g.index.to_list()
        if not seg_ids:
            continue
        dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
        if dom_ctx not in HP1_CTX:
            continue
        dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
        vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em})
    all_emitters = sorted(set(v['em'] for v in vocs))
    print(f'[{name}] {len(vocs)} vocalizations over {len(all_emitters)} emitters')

    # Map seg_df index values to row indices in emb array.
    # seg_df may have a non-default integer index (it is the original
    # 0..N-1 for ablation_state.joblib, but for BEATs it is also 0..N-1
    # over its own 49604-segment slice). emb is indexed by the positional
    # row number in seg_df. We build a lookup table.
    idx_to_pos = {idx: pos for pos, idx in enumerate(seg_df.index.to_list())}

    def resolve(seg_ids):
        return np.array([idx_to_pos[i] for i in seg_ids], dtype=int)

    def emitter_split(seed: int):
        rng = np.random.default_rng(seed)
        em_arr = np.array(all_emitters)
        rng.shuffle(em_arr)
        test_em = set(em_arr[:TEST_EMITTERS_PER_SEED].tolist())
        return ([v for v in vocs if v['em'] not in test_em],
                [v for v in vocs if v['em'] in test_em])

    def bag_of_syll(seq, V=V_global):
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
            pos = resolve(v['seg_ids'])
            labs = [int(hdb_nca[p]) for p in pos if hdb_nca[p] >= 0]
            if labs:
                Xt.append(bag_of_syll(labs)); yt.append(v['ctx'])
        Xe, ye = [], []
        for v in test_vocs:
            pos = resolve(v['seg_ids'])
            labs = [int(hdb_nca[p]) for p in pos if hdb_nca[p] >= 0]
            if labs:
                Xe.append(bag_of_syll(labs)); ye.append(v['ctx'])
        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                     random_state=seed, n_jobs=-1).fit(Xt, yt)
        return np.array(ye), rf.predict(Xe)

    variant_kwargs = {
        'dpgmm':   dict(n_components=15, max_iter=150, covariance_type=covariance_type),
        'hdbscan': dict(),
        'kmeans':  dict(n_clusters=15),
    }
    # DPGMMTokenizer does not accept covariance_type; fall back:
    from sklearn.mixture import BayesianGaussianMixture  # noqa
    # monkey-patch our DPGMM to respect covariance_type
    from src import per_context_tokenizer as pct
    _orig_fit = pct.DPGMMTokenizer.fit
    def _patched_fit(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._bgm = BayesianGaussianMixture(
                n_components=self.n_components,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=self.weight_concentration_prior,
                covariance_type=covariance_type,
                max_iter=self.max_iter,
                random_state=self.random_state,
            ).fit(X)
        self.prototype_weights = np.asarray(self._bgm.weights_)
        self.prototype_centers = np.asarray(self._bgm.means_)
        self.n_prototypes = int((self.prototype_weights > 1e-3).sum())
        return self
    pct.DPGMMTokenizer.fit = _patched_fit

    variant_kwargs['dpgmm'] = dict(n_components=15, max_iter=150)  # covariance via monkey-patch

    def fit_family_and_predict(train_vocs, test_vocs, seed, variant, prior):
        tr_pos = np.concatenate([resolve(v['seg_ids']) for v in train_vocs])
        fam = PerContextFamily(
            variant=variant, prior=prior,
            tokenizer_kwargs=variant_kwargs[variant])
        fam.fit(emb[tr_pos], ctx_arr[tr_pos], HP1_CTX, seed=seed,
                prior_counts=dict(Counter(v['ctx'] for v in train_vocs)))
        y_true, y_pred = [], []
        for v in test_vocs:
            pos = resolve(v['seg_ids'])
            X_seq = emb[pos]
            if len(X_seq) == 0:
                continue
            y_true.append(v['ctx'])
            y_pred.append(fam.predict_context(X_seq))
        return np.array(y_true), np.array(y_pred)

    rows, rows_pc = [], []
    for seed in range(N_SEEDS):
        tr, te = emitter_split(seed)
        runs = [
            ('baseline',          *baseline_predict(tr, te, seed)),
            ('dpgmm_empirical',   *fit_family_and_predict(tr, te, seed, 'dpgmm',   'empirical')),
            ('dpgmm_uniform',     *fit_family_and_predict(tr, te, seed, 'dpgmm',   'uniform')),
            ('hdbscan_empirical', *fit_family_and_predict(tr, te, seed, 'hdbscan', 'empirical')),
            ('hdbscan_uniform',   *fit_family_and_predict(tr, te, seed, 'hdbscan', 'uniform')),
            ('kmeans_empirical',  *fit_family_and_predict(tr, te, seed, 'kmeans',  'empirical')),
            ('kmeans_uniform',    *fit_family_and_predict(tr, te, seed, 'kmeans',  'uniform')),
        ]
        for method, yt, yp in runs:
            w = float(f1_score(yt, yp, average='weighted', labels=HP1_CTX, zero_division=0))
            m = float(f1_score(yt, yp, average='macro', labels=HP1_CTX, zero_division=0))
            cc = float(matthews_corrcoef(yt, yp)) if len(set(yt)) > 1 else float('nan')
            rows.append({'embedding': name, 'method': method, 'seed': seed,
                         'n_test': len(yt), 'weighted_f1': w, 'macro_f1': m, 'mcc': cc})
            per_cls = f1_score(yt, yp, labels=HP1_CTX, average=None, zero_division=0)
            for c, f in zip(HP1_CTX, per_cls):
                rows_pc.append({'embedding': name, 'method': method, 'seed': seed,
                                'context': CTX_NAME[c],
                                'n_true': int((yt == c).sum()), 'f1': float(f)})
            print(f'[{name}] seed={seed} {method:20s}  w={w:.3f} m={m:.3f} mcc={cc:.3f}')
    return rows, rows_pc


def main() -> int:
    if not (CKPT / 'ablation_state.joblib').exists():
        print('ERROR: T7 SSD not mounted.', file=sys.stderr)
        return 2

    st = joblib.load(CKPT / 'ablation_state.joblib')
    seg_df = st['seg_df']
    hdb_nca = st['hdb_nca_labels']
    V_mel = int(hdb_nca.max()) + 1

    all_rows, all_pc = [], []

    # -- UMAP-2D (reference, should reproduce existing results) --
    emb_2d = st['embedding']
    print('\n=== UMAP-2D ===')
    r, pc = run_sweep_on('umap_2d', emb_2d, seg_df, hdb_nca, V_mel, covariance_type='full')
    all_rows += r; all_pc += pc

    # -- UMAP-16D --
    emb_16d = np.load(CKPT / 'umap_16d.npy')
    print('\n=== UMAP-16D ===')
    r, pc = run_sweep_on('umap_16d', emb_16d, seg_df, hdb_nca, V_mel, covariance_type='full')
    all_rows += r; all_pc += pc

    # -- BEATs-768D (diagonal covariance due to dimensionality) --
    be = joblib.load(CKPT / 'beats_full_experiment.joblib')
    seg_meta_be = be['seg_meta'].copy()
    X_be = be['X_beats']
    hdb_be = be['hdb_nca']
    V_be = int(hdb_be.max()) + 1
    # Re-introduce pos_segment for ordering within file via row order
    seg_meta_be = seg_meta_be.reset_index(drop=True)
    seg_meta_be['pos_segment'] = seg_meta_be.groupby('file_name').cumcount()
    print('\n=== BEATs-768D (diagonal cov) ===')
    r, pc = run_sweep_on('beats_768d', X_be, seg_meta_be, hdb_be, V_be,
                          covariance_type='diag')
    all_rows += r; all_pc += pc

    # -- BEATs UMAP-16D --
    X_be_umap2d = be['umap_beats']  # (49604, 2) — only 2D cached
    # Compute UMAP-16D from BEATs if not cached
    cache_be_umap16 = CKPT / 'beats_umap16.npy'
    if cache_be_umap16.exists():
        X_be_umap16 = np.load(cache_be_umap16)
    else:
        print('Computing UMAP-16D on BEATs (~few min)...')
        import umap
        reducer = umap.UMAP(n_components=16, n_neighbors=30, min_dist=0.3,
                             metric='euclidean', random_state=42)
        X_be_umap16 = reducer.fit_transform(X_be).astype(np.float32)
        np.save(cache_be_umap16, X_be_umap16)
    print('\n=== BEATs-UMAP-16D ===')
    r, pc = run_sweep_on('beats_umap_16d', X_be_umap16, seg_meta_be, hdb_be, V_be,
                          covariance_type='full')
    all_rows += r; all_pc += pc

    df = pd.DataFrame(all_rows)
    dfpc = pd.DataFrame(all_pc)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / 'embedding_sweep_5seeds.csv', index=False)
    dfpc.to_csv(OUT_DIR / 'embedding_sweep_per_class.csv', index=False)

    summary = {}
    for emb_name, g_emb in df.groupby('embedding'):
        summary[emb_name] = {}
        for method, g in g_emb.groupby('method'):
            summary[emb_name][method] = {
                'weighted_f1_mean': float(g['weighted_f1'].mean()),
                'weighted_f1_std':  float(g['weighted_f1'].std()),
                'macro_f1_mean':    float(g['macro_f1'].mean()),
                'macro_f1_std':     float(g['macro_f1'].std()),
                'mcc_mean':         float(g['mcc'].mean()),
                'mcc_std':          float(g['mcc'].std()),
            }
    with (OUT_DIR / 'embedding_sweep_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    print('\n=== AGGREGATES BY EMBEDDING ===')
    for emb_name in summary:
        print(f'\n--- {emb_name} ---')
        for method, s in summary[emb_name].items():
            print(f"  {method:18s}  w={s['weighted_f1_mean']:.3f}±{s['weighted_f1_std']:.3f}   "
                  f"m={s['macro_f1_mean']:.3f}±{s['macro_f1_std']:.3f}   "
                  f"mcc={s['mcc_mean']:.3f}±{s['mcc_std']:.3f}")
    print(f'\nSaved to {OUT_DIR}/embedding_sweep_*.{csv,json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
