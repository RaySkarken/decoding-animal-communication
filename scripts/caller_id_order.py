"""Does token ORDER encode individual identity (caller-ID), even if not context?

Dissociation test. The context experiments show order carries no behavioral-context
signal. Individual vocal signatures, however, often DO have sequential structure.
We test caller-ID (closed-set over the most-recorded bats) with the same shuffle
control. A positive Δ(real-shuf) here while context shows none would be a clean
dissociation: sequential structure encodes WHO, not WHAT-context.

Protocol differs from context (caller-ID cannot be cross-individual): random voc
split 80/20, stratified by emitter; restrict to top-K emitters by voc count.
5 random splits, macro F1 over K emitters, BERT real vs shuffled + bag-LR.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import time, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seqlib as S

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
TOPK_EM = 20          # most-recorded emitters -> stable closed-set classes
N_SEEDS = 5
V_TOK = 30

print('Loading state...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
emb = np.load(CACHE / 'umap_152k_21x32_md1.0_8d.npy')
ctx = seg_df['context'].to_numpy(); em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy(); pos_arr = seg_df['pos_segment'].to_numpy()


def stratified_split(vocs, seed, frac=0.2):
    rng = np.random.default_rng(seed)
    by = {}
    for i, v in enumerate(vocs):
        by.setdefault(v['em'], []).append(i)
    tr, te = [], []
    for em, idxs in by.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        ncut = max(1, int(len(idxs) * frac))
        te += idxs[:ncut].tolist(); tr += idxs[ncut:].tolist()
    return [vocs[i] for i in tr], [vocs[i] for i in te]


if __name__ == '__main__':
    tok = KMeans(n_clusters=V_TOK, n_init=10, random_state=0).fit_predict(emb).astype(np.int32)
    vocs_all = S.build_vocs(tok, file_arr, pos_arr, ctx, em_arr)
    # top-K emitters by voc count
    cnt = pd.Series([v['em'] for v in vocs_all]).value_counts()
    keep = set(cnt.index[:TOPK_EM].tolist())
    vocs = [v for v in vocs_all if v['em'] in keep]
    y2i = {em: i for i, em in enumerate(sorted(keep))}
    print(f'caller-ID: {len(vocs)} vocs, {len(keep)} emitters (chance={1/len(keep):.3f})', flush=True)

    rows = []
    for seed in range(N_SEEDS):
        t0 = time.time()
        tr, te = stratified_split(vocs, seed)
        rm = S.train_eval_bert(tr, te, V_TOK, y2i, 'em', seed, shuf=False)
        sm = S.train_eval_bert(tr, te, V_TOK, y2i, 'em', seed, shuf=True)
        bm = S.bag_lr(tr, te, V_TOK, y2i, 'em')
        print(f'  seed {seed}: caller BERT real={rm:.3f} shuf={sm:.3f} bag={bm:.3f} ({time.time()-t0:.0f}s)', flush=True)
        rows.append({'task': 'caller_id', 'tokenizer': 'kmeans30', 'seed': seed,
                     'bert_real_macro': rm, 'bert_shuf_macro': sm, 'bag_lr_macro': bm})
        pd.DataFrame(rows).to_csv(OUT / 'caller_id_order.csv', index=False)

    df = pd.DataFrame(rows)
    m, lo, hi = S.ci95((df.bert_real_macro - df.bert_shuf_macro).tolist())
    print(f'\nCaller-ID: real={df.bert_real_macro.mean():.3f}±{df.bert_real_macro.std():.3f} '
          f'shuf={df.bert_shuf_macro.mean():.3f}±{df.bert_shuf_macro.std():.3f} '
          f'bag={df.bag_lr_macro.mean():.3f} | Δ(real-shuf)={m:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] '
          f'{"ORDER HELPS for identity" if lo>0 else "order n.s."}', flush=True)
    print(f'Saved: {OUT/"caller_id_order.csv"}', flush=True)
