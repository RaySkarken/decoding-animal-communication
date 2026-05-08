"""Dimension sweep: find the sweet spot UMAP dimensionality.

Tests DP-GMM diagonal + k-means (both priors) across UMAP-{2,8,16,32}D
to isolate the effect of embedding dimension while controlling for overfit
via diagonal covariance in DP-GMM.

Saves partial CSV after each (embedding, seed) pair.
"""
from __future__ import annotations

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

from src.per_context_tokenizer import PerContextFamily  # noqa
from src import per_context_tokenizer as pct           # noqa

CKPT = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAME = {2: 'Biting', 3: 'Feeding', 4: 'Fighting', 5: 'Grooming',
            6: 'Isolation', 7: 'Kissing', 9: 'Mating', 10: 'Threat'}
N_SEEDS = 5
TEST_EMITTERS_PER_SEED = 11
OUT_DIR = REPO / 'docs' / 'thesis' / 'figures'


def patch_dpgmm_cov(cov_type: str):
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


def main() -> int:
    if not (CKPT / 'ablation_state.joblib').exists():
        print('ERROR: T7 not mounted', file=sys.stderr); return 2
    st = joblib.load(CKPT / 'ablation_state.joblib')
    seg_df = st['seg_df']
    ctx_arr = seg_df['context'].to_numpy()
    hdb_nca = st['hdb_nca_labels']
    V_mel = int(hdb_nca.max()) + 1

    # Build vocs once (same across embeddings)
    vocs = []
    for _, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
        seg_ids = g.index.to_list()
        if not seg_ids: continue
        dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
        if dom_ctx not in HP1_CTX: continue
        dom_em = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
        vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em})
    all_emitters = sorted(set(v['em'] for v in vocs))
    print(f'{len(vocs)} vocalizations over {len(all_emitters)} emitters', flush=True)

    def split(seed):
        rng = np.random.default_rng(seed)
        em_arr = np.array(all_emitters); rng.shuffle(em_arr)
        test_em = set(em_arr[:TEST_EMITTERS_PER_SEED].tolist())
        return ([v for v in vocs if v['em'] not in test_em],
                [v for v in vocs if v['em'] in test_em])

    def bag_of_syll(seq, V=V_mel):
        c = Counter(seq); n = max(len(seq), 1)
        bos = np.zeros(V, dtype=np.float32)
        for k, cnt in c.items():
            if 0 <= k < V: bos[k] = cnt / n
        probs = np.array(list(c.values()), dtype=np.float32) / n
        ent = float(-(probs * np.log(probs + 1e-12)).sum())
        rich = len(c) / n; rep = max(c.values()) / n if c else 0.0
        return np.concatenate([bos, [n, rich, ent, rep]]).astype(np.float32)

    def baseline_predict(tr, te, seed):
        Xt, yt, Xe, ye = [], [], [], []
        for v in tr:
            labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
            if labs: Xt.append(bag_of_syll(labs)); yt.append(v['ctx'])
        for v in te:
            labs = [int(hdb_nca[i]) for i in v['seg_ids'] if hdb_nca[i] >= 0]
            if labs: Xe.append(bag_of_syll(labs)); ye.append(v['ctx'])
        rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                     random_state=seed, n_jobs=-1).fit(Xt, yt)
        return np.array(ye), rf.predict(Xe)

    def fit_and_predict(emb, tr, te, seed, variant, prior, cov):
        patch_dpgmm_cov(cov)
        tr_seg = np.concatenate([v['seg_ids'] for v in tr])
        kwargs = {'dpgmm': dict(n_components=15, max_iter=150),
                  'kmeans': dict(n_clusters=15)}[variant]
        fam = PerContextFamily(variant=variant, prior=prior, tokenizer_kwargs=kwargs)
        fam.fit(emb[tr_seg], ctx_arr[tr_seg], HP1_CTX, seed=seed,
                prior_counts=dict(Counter(v['ctx'] for v in tr)))
        y_true, y_pred = [], []
        for v in te:
            X_seq = emb[v['seg_ids']]
            if len(X_seq) == 0: continue
            y_true.append(v['ctx']); y_pred.append(fam.predict_context(X_seq))
        return np.array(y_true), np.array(y_pred)

    def metrics(yt, yp, emb_name, method, seed):
        w = float(f1_score(yt, yp, average='weighted', labels=HP1_CTX, zero_division=0))
        m = float(f1_score(yt, yp, average='macro', labels=HP1_CTX, zero_division=0))
        cc = float(matthews_corrcoef(yt, yp)) if len(set(yt)) > 1 else float('nan')
        return {'embedding': emb_name, 'method': method, 'seed': seed,
                'n_test': len(yt), 'weighted_f1': w, 'macro_f1': m, 'mcc': cc}

    embs = {
        'umap_2d':  np.load(CKPT / 'umap_2d.npy'),
        'umap_8d':  np.load(CKPT / 'umap_8d.npy'),
        'umap_16d': np.load(CKPT / 'umap_16d.npy'),
        'umap_32d': np.load(CKPT / 'umap_32d.npy'),
    }
    for n, e in embs.items():
        print(f'{n}: shape={e.shape}', flush=True)

    rows = []
    for seed in range(N_SEEDS):
        tr, te = split(seed)
        yt_bl, yp_bl = baseline_predict(tr, te, seed)
        row_bl = metrics(yt_bl, yp_bl, 'baseline', 'assom_rf', seed)
        rows.append(row_bl)
        print(f"seed={seed} baseline w={row_bl['weighted_f1']:.3f} m={row_bl['macro_f1']:.3f}", flush=True)
        for emb_name, emb in embs.items():
            for cfg in [
                ('dpgmm_diag_emp',  'dpgmm',  'empirical', 'diag'),
                ('dpgmm_diag_uni',  'dpgmm',  'uniform',   'diag'),
                ('dpgmm_spherical_emp',  'dpgmm',  'empirical', 'spherical'),
                ('dpgmm_spherical_uni',  'dpgmm',  'uniform',   'spherical'),
                ('kmeans_emp',      'kmeans', 'empirical', 'full'),
                ('kmeans_uni',      'kmeans', 'uniform',   'full'),
            ]:
                name, variant, prior, cov = cfg
                try:
                    yt, yp = fit_and_predict(emb, tr, te, seed, variant, prior, cov)
                    r = metrics(yt, yp, emb_name, name, seed)
                except Exception as e:
                    print(f"seed={seed} {emb_name} {name:20s} SKIP ({e.__class__.__name__}: {str(e)[:80]})", flush=True)
                    continue
                rows.append(r)
                print(f"seed={seed} {emb_name} {name:20s} w={r['weighted_f1']:.3f} m={r['macro_f1']:.3f} mcc={r['mcc']:.3f}", flush=True)
        # save incrementally
        pd.DataFrame(rows).to_csv(OUT_DIR / 'dim_sweep_partial.csv', index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'dim_sweep_5seeds.csv', index=False)
    print('\n=== AGGREGATES ===', flush=True)
    for (emb_name, method), g in df.groupby(['embedding', 'method']):
        print(f"{emb_name:10s} {method:18s}  w={g['weighted_f1'].mean():.3f}±{g['weighted_f1'].std(ddof=1):.3f}   "
              f"m={g['macro_f1'].mean():.3f}±{g['macro_f1'].std(ddof=1):.3f}   "
              f"mcc={g['mcc'].mean():.3f}±{g['mcc'].std(ddof=1):.3f}", flush=True)
    print('DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
