"""Generalization: does the order-null hold on a 2nd species (marmosets)?

Mirrors the bat order-test on InfantMarmosetsVox (Sarkar's own data). Each marmoset
call is segmented into mel-spectrogram FRAMES; frames are tokenized (k-means);
each call -> sequence of frame-tokens. We classify call-type and caller with BERT
(MLM+CLS) on real vs within-call SHUFFLED frame order, + bag-of-tokens baseline.

If marmosets ALSO show order n.s. -> the order-irrelevance generalizes. If order
helps for marmosets -> bats (graded system) are special. Either is significant.

Closed-set random split 80/20 stratified by label, 5 seeds, macro F1 + 95% CI.
Processes whichever twins are already extracted under data/twin_N/.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, librosa
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seqlib as S

DATA = Path('/Volumes/T7/datasets/InfantMarmosetsVox/InfantMarmosetsVox')
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path('/Volumes/T7/datasets/InfantMarmosetsVox/cache'); CACHE.mkdir(exist_ok=True)
SR = 44100
N_MELS = 32
N_FFT = 2048
HOP = 512
MAX_FRAMES = 48          # cap frames/call
MIN_FRAMES = 2           # need >=2 for an order test
V_TOK = 30
N_SEEDS = 5
CALLTYPE_NAMES = {0: 'Peep(Pre-Phee)', 1: 'Phee', 2: 'Twitter', 3: 'Trill', 4: 'Trillphee',
                  5: 'TsikTse', 6: 'Egg', 7: 'Pheecry', 8: 'TrllTwitter', 9: 'Pheetwitter', 10: 'Peep'}


def build_calls():
    """Return list of calls with mel-frame feature rows + labels, caching per twin."""
    df = pd.read_csv(DATA / 'labels.csv')
    df['twin'] = df.filename.apply(lambda x: x.split('_')[1][-1])
    df['indiv'] = df.filename.apply(lambda x: x.split('_')[2][-1])
    df['date'] = df.filename.apply(lambda x: x.split('_')[0])
    df['wav'] = df.apply(lambda r: DATA / f'data/twin_{r.twin}/{r.date}_Twin{r.twin}_marmoset{r.indiv}.wav', axis=1)
    df = df[(df.calltype != 11) & (df.calltype != 12)]
    df = df[df.wav.map(lambda p: Path(p).exists())]
    print(f'labelled calls with audio present: {len(df):,} across {df.wav.nunique()} wavs', flush=True)

    calls = []          # each: {'frames': (n,32), 'calltype', 'caller'}
    for wav, g in df.groupby('wav'):
        cpath = CACHE / (Path(wav).stem + '.npz')
        if cpath.exists() and not cpath.name.startswith('._'):
            z = np.load(cpath, allow_pickle=True)
            for fr, ct, cl in zip(z['frames'], z['calltype'], z['caller']):
                fr = np.asarray(fr, dtype=np.float32)
                if fr.ndim == 2 and len(fr) >= MIN_FRAMES:
                    calls.append({'frames': fr, 'calltype': int(ct), 'caller': int(cl)})
            continue
        try:
            y, _ = librosa.load(wav, sr=SR, mono=True)
        except Exception as e:
            print(f'  skip {wav}: {e}', flush=True); continue
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
        logmel = librosa.power_to_db(mel).T.astype(np.float32)        # (T, 32)
        frame_t = np.arange(logmel.shape[0]) * HOP / SR
        frames_list, cts, cls = [], [], []
        for _, r in g.iterrows():
            i0 = np.searchsorted(frame_t, r.start); i1 = np.searchsorted(frame_t, r.end)
            fr = logmel[i0:i1][:MAX_FRAMES]
            frames_list.append(fr); cts.append(int(r.calltype)); cls.append(int(r.caller))
            if len(fr) >= MIN_FRAMES:
                calls.append({'frames': fr, 'calltype': int(r.calltype), 'caller': int(r.caller)})
        np.savez(cpath, frames=np.array(frames_list, dtype=object),
                 calltype=np.array(cts), caller=np.array(cls))
        print(f'  processed {Path(wav).stem}: {len(g)} calls', flush=True)
    return calls


def split_strat(vocs, key, seed, frac=0.2):
    rng = np.random.default_rng(seed); by = {}
    for i, v in enumerate(vocs): by.setdefault(v[key], []).append(i)
    tr, te = [], []
    for k, idxs in by.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        n = max(1, int(len(idxs) * frac)); te += idxs[:n].tolist(); tr += idxs[n:].tolist()
    return [vocs[i] for i in tr], [vocs[i] for i in te]


if __name__ == '__main__':
    t0 = time.time()
    calls = build_calls()
    print(f'usable calls (>= {MIN_FRAMES} frames): {len(calls):,}  ({time.time()-t0:.0f}s)', flush=True)
    if len(calls) < 1000:
        print('Not enough calls yet (download still in progress?). Exiting.', flush=True); sys.exit(0)

    # tokenize frames (k-means over all frames)
    allf = np.concatenate([c['frames'] for c in calls])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    print(f'total frames: {len(allf):,}; fitting k-means V={V_TOK}...', flush=True)
    km = KMeans(V_TOK, n_init=10, random_state=0).fit((allf - mu) / sd)
    # assign each call's frames to tokens
    for c in calls:
        c['seq'] = km.predict((c['frames'] - mu) / sd).tolist()

    from collections import Counter
    print('calltype dist:', Counter(c['calltype'] for c in calls).most_common(), flush=True)
    print('caller dist:', Counter(c['caller'] for c in calls).most_common(), flush=True)

    rows = []
    for task in ['calltype', 'caller']:
        labs = sorted(set(c[task] for c in calls))
        y2i = {l: i for i, l in enumerate(labs)}
        print(f'\n=== task={task}: {len(labs)} classes, chance={1/len(labs):.3f} ===', flush=True)
        for seed in range(N_SEEDS):
            ts = time.time()
            tr, te = split_strat(calls, task, seed)
            rm = S.train_eval_bert(tr, te, V_TOK, y2i, task, seed, shuf=False)
            sm = S.train_eval_bert(tr, te, V_TOK, y2i, task, seed, shuf=True)
            bm = S.bag_lr(tr, te, V_TOK, y2i, task)
            print(f'  seed {seed}: BERT real={rm:.3f} shuf={sm:.3f} bag={bm:.3f} ({time.time()-ts:.0f}s)', flush=True)
            rows.append({'task': task, 'seed': seed, 'bert_real_macro': rm,
                         'bert_shuf_macro': sm, 'bag_lr_macro': bm})
        pd.DataFrame(rows).to_csv(OUT / 'marmoset_order.csv', index=False)

    df = pd.DataFrame(rows)
    print('\n=== Marmoset: does ORDER help? ===', flush=True)
    for task in ['calltype', 'caller']:
        sub = df[df.task == task]
        m, lo, hi = S.ci95((sub.bert_real_macro - sub.bert_shuf_macro).tolist())
        print(f'{task:9s} real={sub.bert_real_macro.mean():.3f} shuf={sub.bert_shuf_macro.mean():.3f} '
              f'bag={sub.bag_lr_macro.mean():.3f} Δ={m:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] '
              f'{"ORDER HELPS" if lo>0 else "order n.s."}', flush=True)
    print(f'Saved: {OUT/"marmoset_order.csv"}', flush=True)
