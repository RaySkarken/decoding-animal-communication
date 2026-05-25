"""Generate publication figures from conference/results/*.csv."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('conference/results'); FIG = Path('conference/figures'); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150, 'axes.spgrid' if False else 'axes.grid': True,
                     'grid.alpha': 0.3})


def ci(d):
    d = np.array(d, float); m = d.mean(); h = 2.776 * d.std(ddof=1) / np.sqrt(len(d)); return m, h


# ---- Fig 1: order ablation forest plot (Δ real-shuf, CIs) ----
def fig_order():
    rows = []
    seq = pd.read_csv(R / 'seq_order_test.csv')
    for tk in ['kmeans30', 'agglo30']:
        s = seq[seq.tokenizer == tk]
        m, h = ci((s.bert_real_macro - s.bert_shuf_macro).tolist())
        rows.append((f'bat context · {tk} · BERT', m, h))
    knn = pd.read_csv(R / 'knn_levenshtein_order.csv')
    for tk in knn.tokenizer.unique():
        s = knn[knn.tokenizer == tk]
        m, h = ci((s.knn_real_macro - s.knn_shuf_macro).tolist())
        rows.append((f'bat context · {tk} · kNN-Lev', m, h))
    cid = pd.read_csv(R / 'caller_id_order.csv')
    m, h = ci((cid.bert_real_macro - cid.bert_shuf_macro).tolist())
    rows.append(('bat caller · kmeans30 · BERT', m, h))
    bf = pd.read_csv(R / 'bat_frame_order.csv'); bf = bf[bf.band == 'all']
    m, h = ci((bf.real_macro - bf.shuf_macro).tolist())
    rows.append(('bat context · frame-level · BERT', m, h))
    mo = pd.read_csv(R / 'marmoset_order.csv')
    for task in ['calltype', 'caller']:
        s = mo[mo.task == task]
        m, h = ci((s.bert_real_macro - s.bert_shuf_macro).tolist())
        rows.append((f'marmoset {task} · BERT', m, h))

    labels = [r[0] for r in rows]; means = [r[1] for r in rows]; errs = [r[2] for r in rows]
    colors = ['tab:red' if 'marmoset' in l else 'tab:blue' for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(rows))
    ax.errorbar(means, y, xerr=errs, fmt='o', color='none', ecolor='gray', capsize=3)
    for yi, mi, ei, c in zip(y, means, errs, colors):
        ax.plot(mi, yi, 'o', color=c)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel('Δ macro-F1 (real − shuffled order)  [95% CI]')
    ax.set_title('Token order ablation: bats (blue) vs marmosets (red)')
    fig.tight_layout(); fig.savefig(FIG / 'fig1_order_ablation.pdf'); fig.savefig(FIG / 'fig1_order_ablation.png')
    print('fig1 saved', flush=True)


# ---- Fig 2: length modulation ----
def fig_length():
    lc = pd.read_csv(R / 'length_control_marmoset.csv')
    bf = pd.read_csv(R / 'bat_frame_order.csv')
    fig, ax = plt.subplots(figsize=(6, 4))
    # marmoset bands
    xs, ys, es = [], [], []
    for band in lc.band.unique():
        s = lc[lc.band == band]; m, h = ci((s.real_macro - s.shuf_macro).tolist())
        xs.append(s.mean_len.iloc[0]); ys.append(m); es.append(h)
    ax.errorbar(xs, ys, yerr=es, fmt='o-', color='tab:red', capsize=3, label='marmoset call-type')
    # bat frame bands
    xb, yb, eb = [], [], []
    for band in bf.band.unique():
        if band == 'all': continue
        s = bf[bf.band == band]
        if len(s) < 2: continue
        ml = {'13-40': 21, '41-96': 78}.get(band, np.nan)
        m, h = ci((s.real_macro - s.shuf_macro).tolist())
        xb.append(ml); yb.append(m); eb.append(h)
    ax.errorbar(xb, yb, yerr=eb, fmt='s-', color='tab:blue', capsize=3, label='bat context (frame)')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('mean sequence length (sub-units)'); ax.set_ylabel('Δ macro-F1 (real − shuffled)')
    ax.set_title('Order effect vs sequence length'); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / 'fig2_length_modulation.pdf'); fig.savefig(FIG / 'fig2_length_modulation.png')
    print('fig2 saved', flush=True)


# ---- Fig 3: SSL tokenizer vs V ----
def fig_ssl_vocab():
    vs = pd.read_csv(R / 'vocab_sweep_tokenizers.csv')
    fig, ax = plt.subplots(figsize=(6, 4))
    for src, c in [('mel_umap', 'tab:gray'), ('ssl', 'tab:green')]:
        s = vs[vs.source == src].sort_values('V')
        ax.plot(s.V, s.macro_f1_mean, 'o-', color=c, label=src)
        ax.fill_between(s.V, s.ci_lo, s.ci_hi, color=c, alpha=0.2)
    ax.axhline(0.313, color='k', ls=':', lw=1, label='per-context DP-GMM (0.313)')
    ax.set_xscale('log'); ax.set_xlabel('vocabulary size V'); ax.set_ylabel('bat context macro-F1')
    ax.set_title('SSL vs mel-UMAP tokenizer across V'); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / 'fig3_ssl_vocab.pdf'); fig.savefig(FIG / 'fig3_ssl_vocab.png')
    print('fig3 saved', flush=True)


# ---- Fig 4: SSL positive-pair ablation crossover ----
def fig_ssl_ablation():
    ab = pd.read_csv(R / 'marmoset_ssl_positive_ablation.csv')
    piv = ab.pivot(index='positive_strategy', columns='task', values='macro_f1')
    order = ['mel', 'same_call', 'adjacent', 'augment']
    piv = piv.reindex([o for o in order if o in piv.index])
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(piv)); w = 0.38
    ax.bar(x - w/2, piv['caller'], w, label='caller', color='tab:purple')
    ax.bar(x + w/2, piv['calltype'], w, label='call-type', color='tab:orange')
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=15)
    ax.set_ylabel('marmoset macro-F1'); ax.set_title('SSL positive-pair choice selects the task')
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / 'fig4_ssl_positive_ablation.pdf'); fig.savefig(FIG / 'fig4_ssl_positive_ablation.png')
    print('fig4 saved', flush=True)


if __name__ == '__main__':
    for f in (fig_order, fig_length, fig_ssl_vocab, fig_ssl_ablation):
        try:
            f()
        except Exception as e:
            print(f'{f.__name__} failed: {e}', flush=True)
    print('figures ->', FIG, flush=True)
