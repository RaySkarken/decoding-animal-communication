"""Simplified embedding sweep with streaming output and per-embedding saves.

Runs ONLY the minimal set needed to answer: does higher dimensionality help?

Embedding 1: UMAP-16D on mel features (same as current thesis but 16D instead of 2D)
    — full 6 configurations (3 tokenizers × 2 priors) + baseline.

Embedding 2: BEATs-768D
    — only DP-GMM with diagonal covariance (2 priors) + baseline.
    (HDBSCAN/kmeans on 768D are pathologically slow due to post-hoc full cov)

Saves after EACH (embedding, seed) pair so partial progress is never lost.
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
from sklearn.mixture import BayesianGaussianMixture

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings('ignore')

from src.per_context_tokenizer import PerContextFamily  # noqa: E402
from src import per_context_tokenizer as pct           # noqa: E402

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAME = {2: 'Biting', 3: 'Feeding', 4: 'Fighting', 5: 'Grooming',
            6: 'Isolation', 7: 'Kissing', 9: 'Mating', 10: 'Threat'}
N_SEEDS = 5
TEST_EMITTERS_PER_SEED = 11
OUT_DIR = REPO / 'docs' / 'thesis' / 'figures'


def patch_dpgmm_cov(cov_type: str):
    """Patch DPGMMTokenizer.fit to use the requested covariance_type."""
    def _fit(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._bgm = BayesianGaussianMixture(
                n_components=self.n_components,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=self.weight_concentration_prior,
                covariance_type=cov_type,
                max_iter=self.max_iter,
                random_state=self.random_state,
            ).fit(X)
        self.prototype_weights = np.asarray(self._bgm.weights_)
        self.prototype_centers = np.asarray(self._bgm.means_)
        self.n_prototypes = int((self.prototype_weights > 1e-3).sum())
        return self
    pct.DPGMMTokenizer.fit = _fit


def build_vocs(seg_df):
    """Group into vocalizations keyed by file_name."""
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
    return vocs, sorted(set(v['em'] for v in vocs))


def run_one_config(name_emb: str, emb: np.ndarray, seg_df: pd.DataFrame,
                    hdb_nca: np.ndarray, V_global: int, configs: list,
                    intermediate_csv: Path):
    """Run configurations on a given embedding, saving progress after each seed."""
    ctx_arr = seg_df['context'].to_numpy()
    vocs, all_emitters = build_vocs(seg_df)
    print(f'[{name_emb}] {len(vocs)} vocs, {len(all_emitters)} emitters, '
          f'emb_dim={emb.shape[1]}', flush=True)

    idx_to_pos = {idx: pos for pos, idx in enumerate(seg_df.index.to_list())}
    def resolve(seg_ids):
        return np.array([idx_to_pos[i] for i in seg_ids], dtype=int)

    def emitter_split(seed: int):
        rng = np.random.default_rng(seed)
        em_arr = np.array(all_emitters); rng.shuffle(em_arr)
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
        Xt, yt, Xe, ye = [], [], [], []
        for v in train_vocs:
            pos = resolve(v['seg_ids'])
            labs = [int(hdb_nca[p]) for p in pos if hdb_nca[p] >= 0]
            if labs: Xt.append(bag_of_syll(labs)); yt.append(v['ctx'])
        for v in test_vocs:
            pos = resolve(v['seg_ids'])
            labs = [int(hdb_nca[p]) for p in pos if hdb_nca[p] >= 0]
            if labs: Xe.append(bag_of_syll(labs)); ye.append(v['ctx'])
        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                     random_state=seed, n_jobs=-1).fit(Xt, yt)
        return np.array(ye), rf.predict(Xe)

    def fit_family_and_predict(train_vocs, test_vocs, seed, variant, prior, cov_type):
        patch_dpgmm_cov(cov_type)
        tr_pos = np.concatenate([resolve(v['seg_ids']) for v in train_vocs])
        kwargs = {'dpgmm': dict(n_components=15, max_iter=150),
                  'hdbscan': dict(), 'kmeans': dict(n_clusters=15)}[variant]
        fam = PerContextFamily(variant=variant, prior=prior, tokenizer_kwargs=kwargs)
        fam.fit(emb[tr_pos], ctx_arr[tr_pos], HP1_CTX, seed=seed,
                prior_counts=dict(Counter(v['ctx'] for v in train_vocs)))
        y_true, y_pred = [], []
        for v in test_vocs:
            pos = resolve(v['seg_ids']); X_seq = emb[pos]
            if len(X_seq) == 0: continue
            y_true.append(v['ctx'])
            y_pred.append(fam.predict_context(X_seq))
        return np.array(y_true), np.array(y_pred)

    rows, rows_pc = [], []
    for seed in range(N_SEEDS):
        tr, te = emitter_split(seed)
        yt_bl, yp_bl = baseline_predict(tr, te, seed)
        for method, yt, yp in [('baseline', yt_bl, yp_bl)]:
            _append_metrics(rows, rows_pc, name_emb, method, seed, yt, yp)
            print(f'[{name_emb}] seed={seed} baseline done', flush=True)
        for cfg in configs:
            variant, prior, cov = cfg['variant'], cfg['prior'], cfg['cov']
            method = cfg['name']
            yt, yp = fit_family_and_predict(tr, te, seed, variant, prior, cov)
            _append_metrics(rows, rows_pc, name_emb, method, seed, yt, yp)
            print(f'[{name_emb}] seed={seed} {method} done', flush=True)
        # intermediate save after each seed (overwrite full CSV each time)
        pd.DataFrame(rows).to_csv(intermediate_csv, index=False)
    return rows, rows_pc


def _append_metrics(rows, rows_pc, name_emb, method, seed, yt, yp):
    w = float(f1_score(yt, yp, average='weighted', labels=HP1_CTX, zero_division=0))
    m = float(f1_score(yt, yp, average='macro', labels=HP1_CTX, zero_division=0))
    cc = float(matthews_corrcoef(yt, yp)) if len(set(yt)) > 1 else float('nan')
    rows.append({'embedding': name_emb, 'method': method, 'seed': seed,
                 'n_test': len(yt), 'weighted_f1': w, 'macro_f1': m, 'mcc': cc})
    per_cls = f1_score(yt, yp, labels=HP1_CTX, average=None, zero_division=0)
    for c, f in zip(HP1_CTX, per_cls):
        rows_pc.append({'embedding': name_emb, 'method': method, 'seed': seed,
                        'context': CTX_NAME[c], 'n_true': int((yt == c).sum()),
                        'f1': float(f)})


def main() -> int:
    if not (CKPT / 'ablation_state.joblib').exists():
        print('ERROR: T7 not mounted', file=sys.stderr); return 2
    st = joblib.load(CKPT / 'ablation_state.joblib')
    seg_df = st['seg_df']
    hdb_nca = st['hdb_nca_labels']
    V_mel = int(hdb_nca.max()) + 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows, all_pc = [], []

    # --- UMAP-16D (full cov), all 6 configs ---
    emb_16d = np.load(CKPT / 'umap_16d.npy')
    configs_full = [
        dict(name='dpgmm_empirical',   variant='dpgmm',   prior='empirical', cov='full'),
        dict(name='dpgmm_uniform',     variant='dpgmm',   prior='uniform',   cov='full'),
        dict(name='hdbscan_empirical', variant='hdbscan', prior='empirical', cov='full'),
        dict(name='hdbscan_uniform',   variant='hdbscan', prior='uniform',   cov='full'),
        dict(name='kmeans_empirical',  variant='kmeans',  prior='empirical', cov='full'),
        dict(name='kmeans_uniform',    variant='kmeans',  prior='uniform',   cov='full'),
    ]
    print('\n=== UMAP-16D ===', flush=True)
    r, pc = run_one_config('umap_16d', emb_16d, seg_df, hdb_nca, V_mel, configs_full,
                            OUT_DIR / 'embedding_sweep_v2_umap16d_partial.csv')
    all_rows += r; all_pc += pc
    # full save after this embedding
    pd.DataFrame(all_rows).to_csv(OUT_DIR / 'embedding_sweep_v2_5seeds.csv', index=False)
    pd.DataFrame(all_pc).to_csv(OUT_DIR / 'embedding_sweep_v2_per_class.csv', index=False)

    # --- BEATs-768D (diagonal cov), only DP-GMM ---
    be = joblib.load(CKPT / 'beats_full_experiment.joblib')
    seg_meta_be = be['seg_meta'].reset_index(drop=True)
    seg_meta_be['pos_segment'] = seg_meta_be.groupby('file_name').cumcount()
    X_be = be['X_beats']
    hdb_be = be['hdb_nca']
    V_be = int(hdb_be.max()) + 1
    configs_diag = [
        dict(name='dpgmm_empirical', variant='dpgmm', prior='empirical', cov='diag'),
        dict(name='dpgmm_uniform',   variant='dpgmm', prior='uniform',   cov='diag'),
    ]
    print('\n=== BEATs-768D (diag cov) ===', flush=True)
    r, pc = run_one_config('beats_768d', X_be, seg_meta_be, hdb_be, V_be, configs_diag,
                            OUT_DIR / 'embedding_sweep_v2_beats768_partial.csv')
    all_rows += r; all_pc += pc
    pd.DataFrame(all_rows).to_csv(OUT_DIR / 'embedding_sweep_v2_5seeds.csv', index=False)
    pd.DataFrame(all_pc).to_csv(OUT_DIR / 'embedding_sweep_v2_per_class.csv', index=False)

    # --- Final aggregates ---
    df = pd.DataFrame(all_rows)
    print('\n=== AGGREGATES ===', flush=True)
    for emb_name in df['embedding'].unique():
        print(f'\n--- {emb_name} ---', flush=True)
        for method in sorted(df[df['embedding'] == emb_name]['method'].unique()):
            g = df[(df['embedding'] == emb_name) & (df['method'] == method)]
            print(f"  {method:20s}  w={g['weighted_f1'].mean():.3f}±{g['weighted_f1'].std(ddof=1):.3f}   "
                  f"m={g['macro_f1'].mean():.3f}±{g['macro_f1'].std(ddof=1):.3f}   "
                  f"mcc={g['mcc'].mean():.3f}±{g['mcc'].std(ddof=1):.3f}", flush=True)
    print('DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
