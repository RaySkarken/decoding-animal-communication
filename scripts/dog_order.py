"""3rd species (carnivore): does bark-sequence order encode breed / individual?

DogSpeak: 77k bark-sequence WAVs, 157 dogs, 5 breeds. Path encodes labels:
dogspeak_released/dog_N/{idx}_{breed}_{sex}_dog_N.wav. Each WAV -> mel frames ->
k-means tokens -> sequence. Order control (BERT real vs within-seq shuffled + bag),
tasks: breed (5) and individual (top-K dogs). Stratified split, 5 seeds, macro-F1, CI.

Extends the bat (graded) vs marmoset (structured primate) contrast to a phylogenetically
distant taxon.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import sys, time
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, librosa
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seqlib as S

ROOT = Path('/Volumes/T7/datasets/DogSpeak/dogspeak_released')
CACHE = Path('/Volumes/T7/datasets/DogSpeak/cache'); CACHE.mkdir(parents=True, exist_ok=True)
OUT = Path('conference/results'); OUT.mkdir(parents=True, exist_ok=True)
SR = 16000; N_MELS = 32; N_FFT = 1024; HOP = 256
MIN_FRAMES = 2; MAX_FRAMES = 48; V_TOK = 30; N_SEEDS = 5; TOPK_DOG = 20


def build_calls():
    cache_f = CACHE / 'dog_features.npz'
    if cache_f.exists():
        z = np.load(cache_f, allow_pickle=True)
        return [{'frames': np.asarray(fr, np.float32), 'breed': str(b), 'dog': str(d)}
                for fr, b, d in zip(z['frames'], z['breed'], z['dog'])
                if np.asarray(fr).ndim == 2 and len(fr) >= MIN_FRAMES]
    wavs = sorted(ROOT.glob('dog_*/*.wav'))
    print(f'wavs found: {len(wavs)}', flush=True)
    frames_list, breeds, dogs = [], [], []
    t0 = time.time()
    for i, w in enumerate(wavs):
        parts = w.name.split('_')
        if len(parts) < 4:
            continue
        breed = parts[1]; dog = w.parent.name
        try:
            y, _ = librosa.load(w, sr=SR, mono=True)
        except Exception:
            continue
        if len(y) < HOP * 2:
            continue
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
        lm = librosa.power_to_db(mel).T.astype(np.float32)[:MAX_FRAMES]
        if len(lm) < MIN_FRAMES:
            continue
        frames_list.append(lm); breeds.append(breed); dogs.append(dog)
        if (i + 1) % 10000 == 0:
            print(f'  {i+1}/{len(wavs)} ({time.time()-t0:.0f}s)', flush=True)
    np.savez(cache_f, frames=np.array(frames_list, dtype=object),
             breed=np.array(breeds), dog=np.array(dogs))
    return [{'frames': fr, 'breed': b, 'dog': d} for fr, b, d in zip(frames_list, breeds, dogs)]


def split_strat(vocs, key, seed, frac=0.2):
    rng = np.random.default_rng(seed); by = {}
    for i, v in enumerate(vocs): by.setdefault(v[key], []).append(i)
    tr, te = [], []
    for k, idxs in by.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        n = max(1, int(len(idxs) * frac)); te += idxs[:n].tolist(); tr += idxs[n:].tolist()
    return [vocs[i] for i in tr], [vocs[i] for i in te]


if __name__ == '__main__':
    calls = build_calls()
    print(f'usable bark-seqs: {len(calls):,}', flush=True)
    if len(calls) < 2000:
        print('not enough (download still running?) — exiting', flush=True); sys.exit(0)
    allf = np.concatenate([c['frames'] for c in calls])
    mu, sd = allf.mean(0), allf.std(0) + 1e-6
    print(f'total frames {len(allf):,}; k-means V={V_TOK}...', flush=True)
    km = KMeans(V_TOK, n_init=10, random_state=0).fit((allf - mu) / sd)
    for c in calls:
        c['seq'] = km.predict((c['frames'] - mu) / sd).tolist()

    print('breed dist:', Counter(c['breed'] for c in calls), flush=True)
    # restrict individual task to top-K dogs by count
    dogcnt = Counter(c['dog'] for c in calls)
    keep = {d for d, _ in dogcnt.most_common(TOPK_DOG)}

    rows = []
    for task, subset in [('breed', calls), ('dog', [c for c in calls if c['dog'] in keep])]:
        labs = sorted({c[task] for c in subset}); y2i = {l: i for i, l in enumerate(labs)}
        print(f'\n=== task={task}: {len(labs)} classes, {len(subset)} seqs, chance={1/len(labs):.3f} ===', flush=True)
        for seed in range(N_SEEDS):
            ts = time.time()
            tr, te = split_strat(subset, task, seed)
            rm = S.train_eval_bert(tr, te, V_TOK, y2i, task, seed, shuf=False)
            sm = S.train_eval_bert(tr, te, V_TOK, y2i, task, seed, shuf=True)
            bm = S.bag_lr(tr, te, V_TOK, y2i, task)
            print(f'  seed {seed}: real={rm:.3f} shuf={sm:.3f} bag={bm:.3f} ({time.time()-ts:.0f}s)', flush=True)
            rows.append({'task': task, 'seed': seed, 'bert_real_macro': rm,
                         'bert_shuf_macro': sm, 'bag_lr_macro': bm})
        pd.DataFrame(rows).to_csv(OUT / 'dog_order.csv', index=False)

    df = pd.DataFrame(rows)
    print('\n=== Dog: does ORDER help? ===', flush=True)
    for task in ['breed', 'dog']:
        s = df[df.task == task]
        m, lo, hi = S.ci95((s.bert_real_macro - s.bert_shuf_macro).tolist())
        print(f'{task:6s} real={s.bert_real_macro.mean():.3f} shuf={s.bert_shuf_macro.mean():.3f} '
              f'bag={s.bag_lr_macro.mean():.3f} Δ={m:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] '
              f'{"ORDER HELPS" if lo>0 else "n.s."}', flush=True)
    print(f'Saved: {OUT/"dog_order.csv"}', flush=True)
