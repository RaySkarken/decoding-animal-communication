"""AVES extraction на subset 10000 случайных сегментов из 153k.

10% корпуса достаточно для оценки feasibility и сравнения с BEATs (§3.3).
Если AVES даст лучше BEATs (0.307) — это аргумент в пользу domain-специфичного
претренинга для биоакустики; если хуже или сравним — substituting BEATs/AVES
на одном уровне не даёт разницы.

Subsample стратифицирован по контексту, чтобы покрыть все 8 рабочих контекстов.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, time
from pathlib import Path
import torch
import torchaudio
import torchaudio.models as tam
import soundfile as sf

import os
DEVICE = torch.device('cpu') if os.environ.get('FORCE_CPU') else (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
print(f'Device: {DEVICE}')
CACHE = Path('/Volumes/T7/cache/assom_paper_repro')
AVES_PATH = '/Volumes/T7/models/aves-base-bio.pt'
AUDIO_ROOT = Path('/Volumes/T7/data/raw/fruitbat/zip_contents')
TARGET_SR = 16000
SOURCE_SR = 250000

print('Loading state...')
st = joblib.load(CACHE / 'ablation_state_152k_21x32.joblib')
seg_df = st['seg_df'].reset_index(drop=True)
HP1_CTX = [2, 3, 4, 5, 6, 7, 9, 10]

# stratified subsample по контексту — берём по 1250 из каждого где можно
rng = np.random.default_rng(0)
subset_idx = []
for c in HP1_CTX:
    mask_c = (seg_df['context'].values == c) & (seg_df['emitter'].values != 0)
    avail = np.where(mask_c)[0]
    n_take = min(1250, len(avail))
    subset_idx.extend(rng.choice(avail, n_take, replace=False).tolist())
subset_idx = np.array(sorted(subset_idx))
print(f'Subset: {len(subset_idx)} segments')


print('\nLoading AVES base-bio...')
sd = torch.load(AVES_PATH, map_location='cpu', weights_only=False)
model = tam.hubert_base()
model.load_state_dict(sd, strict=False)
model.eval().to(DEVICE)
print('  loaded')


# Group by audio file (file_name = real WAV filename, e.g. '120601002132055008.WAV')
file_to_subset = {}
for i in subset_idx:
    fname = str(seg_df.iloc[i]['file_name'])
    file_to_subset.setdefault(fname, []).append(i)
print(f'Audio files: {len(file_to_subset)}')

# Pre-build dictionary {filename: full_path}
print('Indexing audio files on disk...')
file_paths = {}
for d in AUDIO_ROOT.glob('files*'):
    if d.name.startswith('.') or d.name.startswith('._'): continue
    if not d.is_dir(): continue
    for p in d.glob('*.WAV'):
        if p.name.startswith('.') or p.name.startswith('._'): continue
        file_paths[p.name] = p
print(f'  indexed {len(file_paths)} WAV files')


def find_audio(fname):
    return file_paths.get(fname)


def aves_embed(wav_16k):
    """wav_16k: 1D tensor at 16kHz; mean+std pool last layer (1536D)."""
    if len(wav_16k) < 400: return None
    with torch.no_grad():
        x = wav_16k.to(DEVICE).unsqueeze(0)
        feat, _ = model.extract_features(x)
        last = feat[-1].squeeze(0)
        agg = torch.cat([last.mean(0), last.std(0) + 1e-6])
    return agg.cpu().numpy()


emb_aves = np.full((len(subset_idx), 1536), np.nan, dtype=np.float32)
idx_map = {orig: pos for pos, orig in enumerate(subset_idx)}

t0 = time.time()
n_done = 0; n_skip = 0
print('\nExtracting AVES embeddings...')

for fname, seg_idxs in file_to_subset.items():
    audio_path = find_audio(fname)
    if audio_path is None: n_skip += len(seg_idxs); continue
    try:
        wav_np, sr = sf.read(str(audio_path), dtype='float32')
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)
        wav = torch.from_numpy(wav_np)
        if sr != SOURCE_SR:
            wav = torchaudio.functional.resample(wav, sr, SOURCE_SR)
        wav_16k = torchaudio.functional.resample(wav, SOURCE_SR, TARGET_SR)
    except Exception as e:
        print(f'  failed {fname}: {e}', flush=True); n_skip += len(seg_idxs); continue

    rate = TARGET_SR / SOURCE_SR
    for i in seg_idxs:
        seg = seg_df.iloc[i]
        s = int(seg['parent_start'] * rate)
        e = int(seg['parent_end'] * rate)
        if e <= s + 100: n_skip += 1; continue
        clip = wav_16k[s:e]
        agg = aves_embed(clip)
        if agg is not None:
            emb_aves[idx_map[i]] = agg; n_done += 1
        else:
            n_skip += 1
    if n_done % 1000 < 50:
        elapsed = time.time() - t0
        print(f'  {n_done}/{len(subset_idx)}, skip={n_skip}, '
              f'elapsed={elapsed/60:.1f}min, rate={n_done/max(elapsed,1):.0f}/s',
              flush=True)


print(f'\nDone: {n_done} / {len(subset_idx)} extracted, {n_skip} skipped, '
      f'time={(time.time()-t0)/60:.1f}min')
np.save(CACHE / 'aves_emb_subset10k.npy', emb_aves)
np.save(CACHE / 'aves_subset_idx.npy', subset_idx)
print(f'Saved to {CACHE}/aves_emb_subset10k.npy')


# --- Quick eval: классификация контекста на этом subset через RF ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

print('\n=== AVES + RF на subset (5 разбиений по особям) ===')
emitters = seg_df.iloc[subset_idx]['emitter'].values
ctx_labels = seg_df.iloc[subset_idx]['context'].values
file_ids = seg_df.iloc[subset_idx]['file_name'].values

# valid mask
valid = ~np.isnan(emb_aves[:, 0])
em_v = emitters[valid]; ctx_v = ctx_labels[valid]; X = emb_aves[valid]; files_v = file_ids[valid]
print(f'  valid embeddings: {valid.sum()}')

# group by vocalization
import pandas as pd
df = pd.DataFrame({'idx': np.arange(len(X)), 'file': files_v, 'em': em_v, 'ctx': ctx_v})
voc_X, voc_y, voc_em = [], [], []
for fname, g in df.groupby('file'):
    idxs = g['idx'].values
    voc_X.append(np.concatenate([X[idxs].mean(0), X[idxs].std(0) + 1e-6]))
    voc_y.append(int(g['ctx'].mode()[0]))
    voc_em.append(int(g['em'].mode()[0]))
voc_X = np.array(voc_X); voc_y = np.array(voc_y); voc_em = np.array(voc_em)
print(f'  vocalizations: {len(voc_X)}')

# 5 emitter splits
results = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    em_uniq = np.array(sorted(set(voc_em.tolist()))); rng.shuffle(em_uniq)
    test_em = set(em_uniq[:11].tolist())
    tr = np.array([i for i in range(len(voc_X)) if voc_em[i] not in test_em])
    te = np.array([i for i in range(len(voc_X)) if voc_em[i] in test_em])
    if len(te) < 10: continue
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                 random_state=seed, n_jobs=-1).fit(voc_X[tr], voc_y[tr])
    pred = rf.predict(voc_X[te])
    f1w = f1_score(voc_y[te], pred, average='weighted', zero_division=0)
    f1m = f1_score(voc_y[te], pred, average='macro', zero_division=0)
    results.append({'seed': seed, 'f1_w': f1w, 'f1_m': f1m, 'n_test': len(te)})
    print(f'  seed={seed}: f1w={f1w:.3f}, f1m={f1m:.3f}, n_test={len(te)}', flush=True)

dfr = pd.DataFrame(results)
if len(dfr) > 0 and 'f1_w' in dfr.columns:
    print(f'\nAVES + RF: weighted F1 = {dfr["f1_w"].mean():.3f} ± {dfr["f1_w"].std():.3f}')
    print(f'         : macro    F1 = {dfr["f1_m"].mean():.3f} ± {dfr["f1_m"].std():.3f}')
else:
    print('\nNo valid results — extraction failed entirely or no test emitters')
print(f'\nFor reference (на полном корпусе из §3.3):')
print(f'  BEATs-768D + DP-GMM diag: 0.307')
print(f'  UMAP-8D + per-context k-means + Bayes: 0.448')
print(f'  Raw mel + RF: 0.529')
dfr.to_csv('docs/thesis/figures/aves_subset_results.csv', index=False)
