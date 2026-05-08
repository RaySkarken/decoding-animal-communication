"""
BEATs on FULL 53k-segment dataset — validate subset finding.

Expected runtime:
  - Audio re-segmentation (already efficient): ~15 min
  - BEATs embeddings (170 seg/s from subset): ~5 min for 53k
  - UMAP + HDBSCAN + DP-GMM + eval: ~15 min
  Total: ~35 min

Key comparison vs mel on same dataset:
  Mel baseline: ctx_purity=0.318, Mel DP-GMM UMAP k=40: 0.420
  Target: BEATs DP-GMM UMAP should significantly exceed 0.420 to
  claim that better embedding resolves the acoustic-context trade-off.
"""
from __future__ import annotations
import io, sys, time, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
import soundfile as sf
from tqdm.auto import tqdm
from collections import Counter
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import hdbscan
import umap

from src.features import compute_beats_embeddings

DATA_DIR = Path('/Volumes/T7/data/raw/fruitbat')
BEATS_CKPT = '/Volumes/T7/models/beats/BEATs_iter3_plus_AS2M.pt'
NATURELM_DIR = '/Volumes/T7/models/NatureLM-audio'
CHECKPOINT_DIR = Path('/Volumes/T7/cache/assom_paper_repro')
SR = 250_000
RANDOM_STATE = 0

# ─── 1. Reload audio for ALL 41 bats ────────────────────────────────────────
ann = pd.read_csv(DATA_DIR / 'Annotations.csv', low_memory=False)
with open(DATA_DIR / 'FileInfo.csv') as f:
    max_cols = max(len(line.split(',')) for line in f)
fi = pd.read_csv(DATA_DIR / 'FileInfo.csv', header=None,
                  names=[f'c{i}' for i in range(max_cols)], low_memory=False)
fi.columns = fi.iloc[0].values
fi = fi.iloc[1:].reset_index(drop=True)
fi['FileID'] = fi['FileID'].astype(int)
fi = fi[['FileID', 'File name', 'File folder']].drop_duplicates('FileID')

ann['FileID'] = ann['FileID'].astype(int)
df = ann.merge(fi, on='FileID', how='inner')
df['Emitter'] = pd.to_numeric(df['Emitter'], errors='coerce')
df = df[df['Emitter'] > 0]
df['Context'] = pd.to_numeric(df['Context'], errors='coerce')
df = df[~df['Context'].isin([0, 11, 12])].dropna(subset=['Start sample', 'End sample'])
df['Emitter'] = df['Emitter'].astype(int)
df['Context'] = df['Context'].astype(int)
df['Start sample'] = df['Start sample'].astype(int)
df['End sample'] = df['End sample'].astype(int)
print(f'Annotations after filter: {len(df)}')

from scipy import signal as scipy_signal
import noisereduce as nr
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation
from scipy.signal import butter, filtfilt

nyq = 0.5 * SR
b, a = butter(4, [256 / nyq, 120000 / nyq], btype='band')
PRE_EMPHASIS = 0.97


def preprocess(y):
    if y.ndim > 1: y = y[:, 0]
    y = y.astype(np.float32)
    y = filtfilt(b, a, y).astype(np.float32)
    try:
        y = nr.reduce_noise(y=y, sr=SR, stationary=False,
                            time_constant_s=0.2, time_mask_smooth_ms=5,
                            freq_mask_smooth_hz=256).astype(np.float32)
    except Exception:
        pass
    return np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1]).astype(np.float32)


DYN_PARAMS = dict(n_fft=2048, hop_length_ms=1.024, win_length_ms=4.096,
                   ref_level_db=20, pre=PRE_EMPHASIS, min_level_db=-60,
                   silence_threshold=0.1, min_silence_for_spec=0.1,
                   max_vocal_for_spec=1.0, min_syllable_length_s=0.01,
                   spectral_range=[2000, 60000], min_level_db_floor=20, verbose=False)

zip_cache = {}
def get_zip(folder):
    if folder not in zip_cache:
        zp = DATA_DIR / f'{folder}.zip'
        zip_cache[folder] = zipfile.ZipFile(zp, 'r') if zp.exists() else None
    return zip_cache[folder]

sub_seg_rows = []
for file_id, gg in tqdm(df.groupby('FileID'), desc='Re-segmenting audio'):
    r0 = gg.iloc[0]
    folder = str(r0['File folder']).strip()
    fname = str(r0['File name']).strip()
    zf = get_zip(folder)
    if zf is None: continue
    try:
        wav_bytes = zf.read(fname)
        audio_full, sr_raw = sf.read(io.BytesIO(wav_bytes), dtype='float32')
    except Exception:
        continue
    for _, r in gg.iterrows():
        s, e = int(r['Start sample']), int(r['End sample'])
        if e > len(audio_full) or s >= e or (e - s) < 100: continue
        ann_audio = preprocess(audio_full[s:e])
        try:
            seg_res = dynamic_threshold_segmentation(ann_audio, SR, **DYN_PARAMS)
        except Exception:
            seg_res = None
        if seg_res and len(seg_res.get('onsets', [])) > 0:
            for on, off in zip(seg_res['onsets'], seg_res['offsets']):
                si, ei = int(on * SR), int(off * SR)
                sub = ann_audio[si:ei]
                if len(sub) < 50: continue
                sub_seg_rows.append({
                    'audio': sub, 'sr': SR,
                    'context': int(r['Context']),
                    'emitter': int(r['Emitter']),
                    'file_name': fname, 'file_id': int(file_id),
                })

for z in zip_cache.values():
    if z is not None: z.close()

seg = pd.DataFrame(sub_seg_rows)
print(f'Sub-segments: {len(seg)}')

# IQR filter
dur = seg['audio'].apply(len).values / SR
q1, q3 = np.percentile(dur, [25, 75]); span = q3 - q1
mask = (dur >= max(q1 - 1.5*span, 0.001)) & (dur <= q3 + 1.5*span)
seg = seg[mask].reset_index(drop=True)
print(f'After IQR: {len(seg)}')

# ─── 2. BEATs embeddings ───────────────────────────────────────────────────
print('\nComputing NatureBEATs embeddings...')
t0 = time.time()
X_beats = compute_beats_embeddings(
    seg, checkpoint_path=BEATS_CKPT,
    mode='pitch_shift', native_sr=SR, batch_size=8,
    encoder='naturelm', naturelm_dir=NATURELM_DIR,
)
print(f'BEATs: {X_beats.shape} | took {time.time()-t0:.0f}s')

# ─── 3. UMAP + HDBSCAN + DP-GMM ─────────────────────────────────────────────
print('\nUMAP → HDBSCAN + NCA noise reassign on BEATs...')
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                     metric='euclidean', random_state=0, n_jobs=-1)
umap_b = reducer.fit_transform(X_beats)
mcs = max(int(len(X_beats) * 0.01), 10)
hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=20,
                       cluster_selection_epsilon=0.1,
                       cluster_selection_method='leaf').fit(umap_b)
hdb_labels = hdb.labels_
noise = hdb_labels == -1
hdb_nca = hdb_labels.copy()
if noise.any() and (~noise).any():
    Xg, yg = umap_b[~noise], hdb_labels[~noise]
    if len(Xg) > 5000:
        idx = np.random.default_rng(0).choice(len(Xg), 5000, replace=False)
        Xg, yg = Xg[idx], yg[idx]
    nca = Pipeline([('nca', NeighborhoodComponentsAnalysis(random_state=0)),
                     ('knn', KNeighborsClassifier(30, n_jobs=-1))])
    try:
        nca.fit(Xg, yg)
        hdb_nca[noise] = nca.predict(umap_b[noise])
    except Exception:
        pass

print('DP-GMM on BEATs-UMAP...')
bgm = BayesianGaussianMixture(n_components=40, weight_concentration_prior_type='dirichlet_process',
                                weight_concentration_prior=0.1, covariance_type='full',
                                max_iter=100, random_state=0)
dp_u = bgm.fit_predict(umap_b)
cnt = Counter(int(x) for x in dp_u)
active = {k for k,v in cnt.items() if v >= 20}
if len(active) < len(cnt):
    am = np.isin(dp_u, list(active))
    if (~am).any():
        knn = NearestNeighbors(1).fit(umap_b[am])
        _, idx = knn.kneighbors(umap_b[~am])
        dp_u[~am] = dp_u[am][idx.ravel()]

print('DP-GMM on BEATs-PCA50...')
X_pca = PCA(n_components=50, random_state=0).fit_transform(X_beats)
bgm = BayesianGaussianMixture(n_components=40, weight_concentration_prior_type='dirichlet_process',
                                weight_concentration_prior=0.1, covariance_type='full',
                                max_iter=100, random_state=0)
dp_p = bgm.fit_predict(X_pca)
cnt = Counter(int(x) for x in dp_p)
active = {k for k,v in cnt.items() if v >= 20}
if len(active) < len(cnt):
    am = np.isin(dp_p, list(active))
    if (~am).any():
        knn = NearestNeighbors(1).fit(X_pca[am])
        _, idx = knn.kneighbors(X_pca[~am])
        dp_p[~am] = dp_p[am][idx.ravel()]

# ─── 4. Eval ────────────────────────────────────────────────────────────────
def ctx_metrics(labels, contexts):
    Hs=[]; pure=0; total=0
    for tid in sorted(set(labels)):
        if tid < 0: continue
        mem = np.where(labels == tid)[0]
        if not len(mem): continue
        cnt = Counter(contexts[mem])
        probs = np.array(list(cnt.values())); probs=probs/probs.sum(); probs=probs[probs>0]
        Hs.append(-np.sum(probs*np.log2(probs)))
        if max(cnt.values())/len(mem) >= 0.5: pure += len(mem)
        total += len(mem)
    return float(np.mean(Hs)), pure/total

ctx = seg['context'].to_numpy()
rows = []
for name, lbls, X_sil in [
    ('BEATs HDBSCAN+NCA', hdb_nca, umap_b),
    ('BEATs DP-GMM UMAP', dp_u, umap_b),
    ('BEATs DP-GMM PCA50', dp_p, X_pca),
]:
    mask = lbls >= 0
    n = min(8000, mask.sum())
    sil = silhouette_score(X_sil[mask][:n], lbls[mask][:n], random_state=0) if n > 100 else np.nan
    H, p = ctx_metrics(lbls, ctx)
    rows.append({'method': name, 'vocab': len(set(lbls)) - (1 if -1 in lbls else 0),
                  'silh_native': round(sil, 3), 'ctx_H': round(H, 2),
                  'ctx_purity': round(p, 3)})

df_res = pd.DataFrame(rows)
print('\n=== BEATs FULL DATASET ===')
print(df_res.to_string(index=False))
df_res.to_csv('docs/thesis/figures/beats_full.csv', index=False)
joblib.dump({
    'seg_meta': seg.drop(columns=['audio']),
    'X_beats': X_beats,
    'umap_beats': umap_b,
    'hdb_nca': hdb_nca,
    'dp_umap': dp_u,
    'dp_pca': dp_p,
    'rows': rows,
}, CHECKPOINT_DIR / 'beats_full_experiment.joblib', compress=3)
print('\nSaved.')
