"""Confusion matrix для рекомендуемой конфигурации.

Рекомендуемая конфигурация: per-context DP-GMM с полной ковариационной
матрицей на UMAP-8D с равномерным априорным распределением, классификация
по правилу максимума правдоподобия. Агрегируем предсказания по 5
разбиениям по особям и строим матрицу ошибок (нормирование по строкам,
т.е. каждая строка показывает условное распределение предсказаний при
истинной метке контекста).

Output:
  docs/thesis/figures/confusion_matrix_recommended.{png,pdf}
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import confusion_matrix

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
CTX_NAME = {2: 'Biting', 3: 'Feeding', 4: 'Fighting', 5: 'Grooming',
            6: 'Isolation', 7: 'Kissing', 9: 'Mating', 10: 'Threat'}


def main():
    print('Loading state and embeddings...', flush=True)
    st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
    seg_df = st['seg_df'].reset_index(drop=True)
    emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
    ctx_arr = seg_df['context'].to_numpy()

    vocs = []
    for fname, g in seg_df.sort_values('pos_segment').groupby('file_name', sort=False):
        seg_ids = g.index.to_list()
        if not seg_ids: continue
        dom_em_signed = int(Counter(g['emitter'].to_numpy()).most_common(1)[0][0])
        if dom_em_signed == 0: continue
        dom_em_abs = abs(dom_em_signed)
        dom_ctx = int(np.bincount(g['context'].to_numpy()).argmax())
        if dom_ctx not in HP1_CTX: continue
        vocs.append({'seg_ids': seg_ids, 'ctx': dom_ctx, 'em': dom_em_abs})
    print(f'  vocs: {len(vocs)}', flush=True)

    all_bats = sorted(set(v['em'] for v in vocs))
    log_prior = {c: -np.log(len(HP1_CTX)) for c in HP1_CTX}

    yt_all, yp_all = [], []
    for s in range(5):
        rng = np.random.default_rng(s)
        ba = np.array(all_bats); rng.shuffle(ba)
        test_b = set(ba[:11].tolist())
        train_v = [v for v in vocs if v['em'] not in test_b]
        test_v = [v for v in vocs if v['em'] in test_b]
        train_mask = np.zeros(len(emb), dtype=bool)
        for v in train_v: train_mask[v['seg_ids']] = True

        toks = {}
        for c in HP1_CTX:
            m = train_mask & (ctx_arr == c)
            if m.sum() < 30: continue
            toks[c] = BayesianGaussianMixture(
                n_components=15,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.1, covariance_type='full',
                max_iter=150, random_state=s
            ).fit(emb[m])

        for v in test_v:
            X = emb[v['seg_ids']]
            if len(X) == 0: continue
            best, bs = None, -np.inf
            for c, t in toks.items():
                ll = t.score_samples(X).sum() + log_prior[c]
                if ll > bs: bs = ll; best = c
            if best is None: continue
            yt_all.append(v['ctx']); yp_all.append(best)

        print(f'  seed {s}: cumulative {len(yt_all)} predictions', flush=True)

    yt = np.array(yt_all); yp = np.array(yp_all)
    cm = confusion_matrix(yt, yp, labels=HP1_CTX)
    cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    # Plot
    labels = [CTX_NAME[c] for c in HP1_CTX]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel('Предсказанный контекст')
    ax.set_ylabel('Истинный контекст')

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            color = 'white' if v > 0.5 else 'black'
            txt = f'{v:.2f}' if v >= 0.01 else ''
            if txt:
                ax.text(j, i, txt, ha='center', va='center', color=color, fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Доля предсказаний')
    plt.tight_layout()

    out_png = Path('docs/thesis/figures/confusion_matrix_recommended.png')
    out_pdf = Path('docs/thesis/figures/confusion_matrix_recommended.pdf')
    plt.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f'\nSaved: {out_png} and {out_pdf}', flush=True)
    print(f'Total predictions across 5 seeds: {len(yt)}', flush=True)
    print(f'Confusion matrix (row-normalized):')
    print(pd.DataFrame(cm_norm, index=labels, columns=labels).round(3).to_string())


if __name__ == '__main__':
    main()
