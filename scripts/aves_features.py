"""AVES (Hagiwara 2023) — извлечение эмбеддингов для всех сегментов.

AVES base — HuBERT-style transformer, предобучен self-supervised на ~5000 часов
звуков животных (BirdAVES initial pretrain). Архитектура: 95M parameters,
12 transformer layers, output 768-D per 50ms frame.

Сегменты — короткие (10–500 ms), потенциально тысячи сегментов в секунду inference.
Чтобы помещалось в разумное время, обрабатываем батчами и используем mean+std pooling
по фреймам.

Resampling: AVES обучен на 16 kHz. Корпус Prat — 250 kHz. Применяем time-stretch
(такой же, как в BEATs-эксперименте): прямой ресэмпл с 250 → 16 kHz, что
эквивалентно time-stretch ×15.6 с сохранением спектральной структуры.

Выход: AVES-эмбеддинг (768D) на каждый сегмент → сохраняется в .npy.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, joblib, time
from pathlib import Path
import torch
import torchaudio

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
print(f'  segments: {len(seg_df)}')


# --- Load AVES ---
print('\nLoading AVES base-bio...')
# AVES uses fairseq HuBERT module; the saved file is a torchaudio.pipelines wrapper
sd = torch.load(AVES_PATH, map_location='cpu', weights_only=False)
print(f'  state dict keys: {list(sd.keys())[:5] if isinstance(sd, dict) else type(sd)}')

# torchaudio HuBERT model
import torchaudio.models as tam
# AVES base = HuBERT base config
model = tam.hubert_base()
# Load from torchaudio bundle-style state dict
if isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']
try:
    model.load_state_dict(sd, strict=False)
    print('  loaded with strict=False')
except Exception as e:
    print(f'  load failed: {e}')
    # Try alternative: state dict has different prefix
    sd2 = {k.replace('model.', '').replace('hubert.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd2, strict=False)
    print('  loaded with prefix fix')

model.eval().to(DEVICE)


# --- Build voc index ---
def find_audio(file_id):
    # File names like 120601000005102988 → search through zip_contents
    for d in AUDIO_ROOT.glob('files*'):
        p = d / f'{file_id}.WAV'
        if p.exists(): return p
        p2 = d / f'{file_id}.wav'
        if p2.exists(): return p2
    return None


# Group segments by source audio file
print('\nBuilding extraction plan...')
seg_df['file_id'] = seg_df['file_id'].astype(str)
file_groups = seg_df.groupby('file_id')
print(f'  unique audio files: {file_groups.ngroups}')


# --- Extract embeddings ---
def aves_embed(audio_16k):
    """audio_16k: 1D tensor at 16kHz; return mean+std pooled embedding (1536D)."""
    if len(audio_16k) < 400: return None
    with torch.no_grad():
        x = audio_16k.to(DEVICE).unsqueeze(0)  # (1, T)
        feat, _ = model.extract_features(x)
        # feat: list of layer outputs; take last
        last = feat[-1].squeeze(0)  # (T_frame, 768)
        agg = torch.cat([last.mean(0), last.std(0) + 1e-6])  # (1536,)
    return agg.cpu().numpy()


emb_aves = np.full((len(seg_df), 1536), np.nan, dtype=np.float32)
n_done = 0; n_skip = 0
t0 = time.time()
print('\nExtracting AVES embeddings (this may take ~1-2 hours)...')

# Resampler: 250kHz → 16kHz (~ time-stretch 15.6x)
resampler = torchaudio.transforms.Resample(SOURCE_SR, TARGET_SR).to(DEVICE)

for fid, grp in file_groups:
    audio_path = find_audio(fid)
    if audio_path is None: n_skip += len(grp); continue
    try:
        wav, sr = torchaudio.load(str(audio_path))
        wav = wav.mean(0)  # mono
        if sr != SOURCE_SR:
            # use torchaudio resample
            wav = torchaudio.functional.resample(wav, sr, SOURCE_SR)
        # Resample to 16kHz
        wav_16k = resampler(wav.to(DEVICE)).cpu()
    except Exception as e:
        print(f'  failed {fid}: {e}', flush=True); n_skip += len(grp); continue

    rate = TARGET_SR / SOURCE_SR
    for _, seg in grp.iterrows():
        s = int(seg['parent_start'] * rate)
        e = int(seg['parent_end'] * rate)
        if e <= s + 100: n_skip += 1; continue
        clip = wav_16k[s:e]
        emb = aves_embed(clip)
        if emb is not None:
            emb_aves[seg.name] = emb; n_done += 1
        else: n_skip += 1
    if n_done % 5000 < 50:
        elapsed = time.time() - t0
        rate_done = n_done / max(elapsed, 1)
        print(f'  {n_done}/{len(seg_df)} done, skip={n_skip}, '
              f'rate={rate_done:.0f}/s, elapsed={elapsed/60:.1f}min', flush=True)

np.save(CACHE / 'aves_emb_152k.npy', emb_aves)
print(f'\nDone: {n_done} embeddings extracted, {n_skip} skipped')
print(f'Total time: {(time.time()-t0)/60:.1f} min')
print(f'Saved to {CACHE}/aves_emb_152k.npy')
