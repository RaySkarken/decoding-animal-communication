"""Decisive disentangling test: is the bat order-null fundamental, or just because
bat sub-unit (segment) sequences are short (~4)?

We re-tokenize bats at FRAME granularity: each segment's mel is 21 time-frames, so a
vocalization yields ~21 x (#segments) ~ 84 frame-tokens — long sequences comparable
to marmoset's long band. We then run the context order-test (real vs shuffled frame
order), overall and length-stratified.

- If bat order is STILL n.s. at long frame-sequences -> graded-system order-null is
  genuine (not a granularity artifact).
- If bat order HELPS at long frame-sequences -> the segment-level null was a
  granularity artifact; order does carry context once enough sub-units are exposed.

Cross-bat 30/11, 5 seeds, macro F1 + 95% CI.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys, time
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seqlib as S

CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]
V_TOK = 30
N_SEEDS = 5
MAX_FRAMES = 96
BANDS = [(13, 40), (41, 96)]

print('Loading state...', flush=True)
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
specs = st['tf_specs']
specs = np.asarray(specs).reshape(len(seg_df), 21, 32).astype(np.float32)  # (N,21,32)
ctx = seg_df['context'].to_numpy(); em_arr = seg_df['emitter'].to_numpy()
file_arr = seg_df['file_name'].to_numpy(); pos_arr = seg_df['pos_segment'].to_numpy()
print(f'specs: {specs.shape}', flush=True)


def build_frame_vocs():
    df = pd.DataFrame({'file': file_arr, 'pos': pos_arr, 'ctx': ctx, 'em': em_arr,
                       'idx': np.arange(len(file_arr))})
    df = df[(df['em'] != 0) & (df['ctx'].isin(HP1_CTX))]
    vocs = []
    for fn, g in df.sort_values('pos').groupby('file', sort=False):
        segids = g['idx'].to_numpy()
        frames = specs[segids].reshape(-1, 32)[:MAX_FRAMES]   # (n_seg*21, 32) in order
        if len(frames) < 2:
            continue
        vocs.append({'frames': frames,
                     'ctx': int(np.bincount(g['ctx'].to_numpy()).argmax()),
                     'em': abs(int(Counter(g['em'].to_numpy().tolist()).most_common(1)[0][0]))})
    return vocs


def split(vocs, seed):
    rng = np.random.default_rng(seed); ems = np.array(sorted(set(v['em'] for v in vocs))); rng.shuffle(ems)
    test = set(ems[:11].tolist())
    return [v for v in vocs if v['em'] not in test], [v for v in vocs if v['em'] in test]


if __name__ == '__main__':
    t0 = time.time()
    vocs = build_frame_vocs()
    lens = [len(v['frames']) for v in vocs]
    print(f'vocs: {len(vocs)}, frame-seq len: median={np.median(lens):.0f} '
          f'p90={np.percentile(lens,90):.0f} max={max(lens)} ({time.time()-t0:.0f}s)', flush=True)

    # k-means frame tokenizer (fit on sample, predict all)
    allf = np.concatenate([v['frames'] for v in vocs])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    rng = np.random.default_rng(0)
    samp = allf[rng.choice(len(allf), min(200000, len(allf)), replace=False)]
    print(f'fitting k-means V={V_TOK} on {len(samp):,} frames (total {len(allf):,})...', flush=True)
    km = KMeans(V_TOK, n_init=10, random_state=0).fit((samp - mu) / sd)
    for v in vocs:
        v['seq'] = km.predict((v['frames'] - mu) / sd).tolist()

    y2i = {c: i for i, c in enumerate(HP1_CTX)}
    rows = []
    # overall
    print('\n=== BAT frame-level, overall context order-test ===', flush=True)
    for seed in range(N_SEEDS):
        ts = time.time()
        tr, te = split(vocs, seed)
        rm = S.train_eval_bert(tr, te, V_TOK, y2i, 'ctx', seed, shuf=False, max_len_cap=MAX_FRAMES)
        sm = S.train_eval_bert(tr, te, V_TOK, y2i, 'ctx', seed, shuf=True, max_len_cap=MAX_FRAMES)
        bm = S.bag_lr(tr, te, V_TOK, y2i, 'ctx')
        print(f'  seed {seed}: real={rm:.3f} shuf={sm:.3f} bag={bm:.3f} ({time.time()-ts:.0f}s)', flush=True)
        rows.append({'band': 'all', 'seed': seed, 'real_macro': rm, 'shuf_macro': sm, 'bag_macro': bm})
    pd.DataFrame(rows).to_csv(OUT / 'bat_frame_order.csv', index=False)

    # length-stratified
    for lo, hi in BANDS:
        band = [v for v in vocs if lo <= len(v['seq']) <= hi]
        if len(band) < 500:
            print(f'band [{lo},{hi}]: too few ({len(band)}) — skip', flush=True); continue
        ml = np.mean([len(v['seq']) for v in band])
        print(f'\n=== band frames[{lo},{hi}]: {len(band)} vocs, mean_len={ml:.1f} ===', flush=True)
        for seed in range(N_SEEDS):
            ts = time.time()
            tr, te = split(band, seed)
            if len(set(v['ctx'] for v in te)) < 2 or len(te) < 50:
                continue
            rm = S.train_eval_bert(tr, te, V_TOK, y2i, 'ctx', seed, shuf=False, max_len_cap=MAX_FRAMES)
            sm = S.train_eval_bert(tr, te, V_TOK, y2i, 'ctx', seed, shuf=True, max_len_cap=MAX_FRAMES)
            print(f'  seed {seed}: real={rm:.3f} shuf={sm:.3f} Δ={rm-sm:+.3f} ({time.time()-ts:.0f}s)', flush=True)
            rows.append({'band': f'{lo}-{hi}', 'seed': seed, 'real_macro': rm, 'shuf_macro': sm, 'bag_macro': np.nan})
        pd.DataFrame(rows).to_csv(OUT / 'bat_frame_order.csv', index=False)

    df = pd.DataFrame(rows)
    print('\n=== BAT frame-level: does ORDER help? ===', flush=True)
    for band in df.band.unique():
        sub = df[df.band == band]
        if len(sub) < 2: continue
        m, lo, hi = S.ci95((sub.real_macro - sub.shuf_macro).tolist())
        print(f'{band:8s} real={sub.real_macro.mean():.3f} shuf={sub.shuf_macro.mean():.3f} '
              f'Δ={m:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] {"ORDER HELPS" if lo>0 else "n.s."}', flush=True)
    print(f'Saved: {OUT/"bat_frame_order.csv"}', flush=True)
