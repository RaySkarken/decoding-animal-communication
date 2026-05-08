"""
BEATs embedding experiment — subset comparison.

Hypothesis: Mel-192D shared clusters are continua (no HDBSCAN-splittable
sub-structure, acoustic similarity decoupled from context). BEATs
embeddings — pretrained SSL on audio — may produce a different geometry
where:
  - acoustic similarity is better aligned with perceptual/semantic
    structure
  - cluster sub-structure is density-separable
  - adaptive operations may actually find signal

Subsetting to top-5 emitters (~20k segments) keeps BEATs compute
manageable (~30 min CPU). If results are promising, scale up.
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
from sklearn.metrics import silhouette_score
import hdbscan
import umap

from src.adaptive_tokenizer import TokenizerState, Token, full_evaluation
from src.features import compute_beats_embeddings, LogMelSpectrogram   # type: ignore

DATA_DIR = Path('/Volumes/T7/data/raw/fruitbat')
BEATS_CKPT = '/Volumes/T7/models/beats/BEATs_iter3_plus_AS2M.pt'
NATURELM_DIR = '/Volumes/T7/models/NatureLM-audio'
CHECKPOINT_DIR = Path('/Volumes/T7/cache/assom_paper_repro')
RANDOM_STATE = 0

# ─── 1. Select a SUBSET of seg_df (top-5 emitters) and re-load audio ────────
st = joblib.load(CHECKPOINT_DIR / 'ablation_state.joblib')
seg_df_all = st['seg_df']

# top 5 emitters by segment count
top_emitters = (seg_df_all['emitter'].value_counts().head(5).index.tolist())
print(f'Top-5 emitters by segment count: {top_emitters}')
seg_sub = seg_df_all[seg_df_all['emitter'].isin(top_emitters)].reset_index(drop=True)
print(f'Subset segments: {len(seg_sub)}')

# ─── 2. Load audio for these sub-segments ───────────────────────────────────
# seg_df has `file_name`, `parent_start`, `parent_end` — indices INTO the
# original annotation audio (not raw WAV). To reconstruct, we reload the
# annotation WAV slice via Start sample / End sample in Annotations.csv.

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
df = df[df['Start sample'].notna() & df['End sample'].notna()]
df['Start sample'] = df['Start sample'].astype(int)
df['End sample'] = df['End sample'].astype(int)

# For each sub-segment, we need (file_name, file_id, parent_start, parent_end)
# → the parent annotation's Start/End samples in the raw WAV.
# Build lookup: (file_name, file_id) → list of (start, end) annotations
# We don't actually know which annotation in the file produced the sub-seg;
# best effort: use pos_segment (sub-segs are contiguous within a file).
#
# Simpler: just re-extract by reading the full annotation and indexing.
# Our sub-seg's parent_start/parent_end are offsets within the parent
# annotation audio. So:
#   raw[file_start + parent_start : file_start + parent_end]
# where file_start is the annotation's Start sample in the WAV.

# Since each file has multiple annotations, we need to map sub-seg to its
# parent annotation. pos_segment is within-file order. A reasonable heuristic:
# sort sub-segs by (file_name, pos_segment); iterate annotations in order
# of Start sample within the same file; assign sub-segs in order.

zip_cache = {}
def get_zip(folder):
    if folder not in zip_cache:
        zp = DATA_DIR / f'{folder}.zip'
        zip_cache[folder] = zipfile.ZipFile(zp, 'r') if zp.exists() else None
    return zip_cache[folder]


def load_audio_cached(fid, fname, folder, cache={}):
    key = (fid, fname)
    if key not in cache:
        zf = get_zip(folder)
        if zf is None:
            cache[key] = None
            return None
        try:
            wav_bytes = zf.read(fname)
            audio, sr = sf.read(io.BytesIO(wav_bytes), dtype='float32')
            cache[key] = (audio, sr)
        except Exception:
            cache[key] = None
    return cache[key]


# Simpler strategy: re-run dynamic segmentation locally for subset to get audio.
# But we already have (parent_start, parent_end) + pos_segment. Let's try the
# simplest thing: find the parent annotation for each sub-seg by matching
# order, then extract.
# Actually the simplest way: each row in seg_df corresponds to ONE sub-syllable
# inside ONE annotation. If pos_segment=0 is the first sub-seg within a file,
# and multiple annotations within a file produce multiple sub-segs each, then
# the numbering crosses annotation boundaries. We need the ORIGINAL annotation
# for each sub-seg, which was lost in the previous pipeline.

# Fallback: re-run preprocessing + dynamic segmentation from scratch on the
# filtered annotation subset, producing sub-segs with audio in memory.

from scipy import signal as scipy_signal
import noisereduce as nr
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation

SR = 250_000
BP_LOW, BP_HIGH = 256, 120000
PRE_EMPHASIS = 0.97

from scipy.signal import butter, filtfilt
nyq = 0.5 * SR
b, a = butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype='band')


def preprocess(y):
    if y.ndim > 1: y = y[:, 0]
    y = y.astype(np.float32)
    y = filtfilt(b, a, y).astype(np.float32)
    try:
        y = nr.reduce_noise(
            y=y, sr=SR, stationary=False,
            time_constant_s=0.2, time_mask_smooth_ms=5, freq_mask_smooth_hz=256,
        ).astype(np.float32)
    except Exception:
        pass
    y = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1]).astype(np.float32)
    return y


DYN_PARAMS = dict(
    n_fft=2048, hop_length_ms=1000 * 256 / SR, win_length_ms=1000 * 1024 / SR,
    ref_level_db=20, pre=PRE_EMPHASIS, min_level_db=-60,
    silence_threshold=0.1, min_silence_for_spec=0.1,
    max_vocal_for_spec=1.0, min_syllable_length_s=0.01,
    spectral_range=[2000, 60000], min_level_db_floor=20, verbose=False,
)


# Extract annotation-level rows for our top-5 emitters' files
df_sub = df[df['Emitter'].isin([str(e) for e in top_emitters] + top_emitters)].copy()
df_sub['Emitter'] = pd.to_numeric(df_sub['Emitter'], errors='coerce')
df_sub = df_sub[df_sub['Emitter'].isin(top_emitters)]
df_sub = df_sub[~df_sub['Context'].isin(['0', '11', '12', 0, 11, 12])]
df_sub['Context'] = pd.to_numeric(df_sub['Context'], errors='coerce').astype(int)
df_sub = df_sub[~df_sub['Context'].isin([0, 11, 12])]
print(f'Annotations in subset: {len(df_sub)}')

# Re-run dynamic segmentation to get audio of each sub-seg
sub_seg_rows = []
for file_id, gg in tqdm(df_sub.groupby('FileID'), desc='Re-segmenting audio'):
    r0 = gg.iloc[0]
    folder = str(r0['File folder']).strip()
    fname = str(r0['File name']).strip()
    cached = load_audio_cached(file_id, fname, folder)
    if cached is None: continue
    audio_full, sr_raw = cached
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
                sub_audio = ann_audio[si:ei]
                if len(sub_audio) < 50: continue
                sub_seg_rows.append({
                    'audio': sub_audio, 'sr': SR,
                    'context': int(r['Context']),
                    'emitter': int(r['Emitter']),
                    'file_name': fname, 'file_id': int(file_id),
                })

for z in zip_cache.values():
    if z is not None: z.close()

seg_beats = pd.DataFrame(sub_seg_rows)
print(f'Sub-segments for BEATs: {len(seg_beats)}')

# IQR filter on duration
dur = seg_beats['audio'].apply(len).values / SR
q1, q3 = np.percentile(dur, [25, 75]); iqr_span = q3 - q1
lo, hi = q1 - 1.5 * iqr_span, q3 + 1.5 * iqr_span
mask = (dur >= max(lo, 0.001)) & (dur <= hi)
seg_beats = seg_beats[mask].reset_index(drop=True)
print(f'After IQR: {len(seg_beats)}')

# ─── 3. Compute BEATs naturelm-pitch embedding (best for ultrasonic) ────────
print('\nComputing NatureBEATs embeddings...')
t0 = time.time()
X_beats = compute_beats_embeddings(
    seg_beats, checkpoint_path=BEATS_CKPT,
    mode='pitch_shift', native_sr=SR, batch_size=8,
    encoder='naturelm', naturelm_dir=NATURELM_DIR,
)
print(f'BEATs embeddings: {X_beats.shape} | took {time.time()-t0:.0f}s')

# ─── 4. Baseline HDBSCAN + UMAP on BEATs ────────────────────────────────────
print('\nRunning UMAP → HDBSCAN on BEATs...')
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                     metric='euclidean', random_state=0, n_jobs=-1)
umap_beats = reducer.fit_transform(X_beats)

mcs = max(int(len(X_beats) * 0.01), 10)
hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=20,
                       cluster_selection_epsilon=0.1,
                       cluster_selection_method='leaf').fit(umap_beats)
hdb_labels = hdb.labels_
n_clust = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
print(f'HDBSCAN clusters on BEATs-UMAP: {n_clust}')

# NCA reassign noise
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, NeighborhoodComponentsAnalysis
noise = hdb_labels == -1
if noise.any() and (~noise).any():
    Xg, yg = umap_beats[~noise], hdb_labels[~noise]
    if len(Xg) > 5000:
        idx = np.random.default_rng(0).choice(len(Xg), 5000, replace=False)
        Xg, yg = Xg[idx], yg[idx]
    nca = Pipeline([('nca', NeighborhoodComponentsAnalysis(random_state=0)),
                     ('knn', KNeighborsClassifier(30, n_jobs=-1))])
    try:
        nca.fit(Xg, yg)
        hdb_labels_nca = hdb_labels.copy()
        hdb_labels_nca[noise] = nca.predict(umap_beats[noise])
    except Exception:
        hdb_labels_nca = hdb_labels
else:
    hdb_labels_nca = hdb_labels
print(f'After NCA: {len(set(hdb_labels_nca))}')

# ─── 5. DP-GMM on BEATs-UMAP ────────────────────────────────────────────────
print('\nRunning DP-GMM on BEATs-UMAP...')
bgm = BayesianGaussianMixture(
    n_components=40, weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=0.1, covariance_type='full',
    max_iter=100, random_state=0,
)
dp_labels_umap = bgm.fit_predict(umap_beats)
cnt = Counter(int(x) for x in dp_labels_umap)
active = {k for k, v in cnt.items() if v >= 20}
if len(active) < len(cnt):
    active_mask = np.isin(dp_labels_umap, list(active))
    tiny = ~active_mask
    if tiny.sum() > 0:
        knn = NearestNeighbors(n_neighbors=1).fit(umap_beats[active_mask])
        _, idx = knn.kneighbors(umap_beats[tiny])
        act_lbl = dp_labels_umap[active_mask]
        dp_labels_umap[tiny] = act_lbl[idx.ravel()]
print(f'DP-GMM on BEATs-UMAP vocab: {len(set(dp_labels_umap))}')

# ─── 6. DP-GMM on BEATs-768D directly ──────────────────────────────────────
print('\nRunning DP-GMM on BEATs-768D...')
# downsample for speed: PCA to 50D
from sklearn.decomposition import PCA
X_beats_pca = PCA(n_components=50, random_state=0).fit_transform(X_beats)
bgm = BayesianGaussianMixture(
    n_components=40, weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=0.1, covariance_type='full',
    max_iter=100, random_state=0,
)
dp_labels_hd = bgm.fit_predict(X_beats_pca)
cnt = Counter(int(x) for x in dp_labels_hd)
active = {k for k, v in cnt.items() if v >= 20}
if len(active) < len(cnt):
    active_mask = np.isin(dp_labels_hd, list(active))
    tiny = ~active_mask
    if tiny.sum() > 0:
        knn = NearestNeighbors(n_neighbors=1).fit(X_beats_pca[active_mask])
        _, idx = knn.kneighbors(X_beats_pca[tiny])
        act_lbl = dp_labels_hd[active_mask]
        dp_labels_hd[tiny] = act_lbl[idx.ravel()]
print(f'DP-GMM on BEATs-PCA50 vocab: {len(set(dp_labels_hd))}')

# ─── 7. Evaluation: silhouettes, ARI vs context, ctx_purity ────────────────
def ctx_metrics(labels, contexts):
    Hs = []; pure_seg = 0; total = 0
    for tid in sorted(set(labels)):
        if tid < 0: continue
        members = np.where(labels == tid)[0]
        if len(members) == 0: continue
        ctx_cnt = Counter(contexts[members])
        probs = np.array(list(ctx_cnt.values())); probs = probs / probs.sum(); probs = probs[probs > 0]
        Hs.append(-np.sum(probs * np.log2(probs)))
        if max(ctx_cnt.values()) / len(members) >= 0.5:
            pure_seg += len(members)
        total += len(members)
    return dict(mean_H=float(np.mean(Hs)), context_purity=pure_seg/total if total else 0)


def ari_nmi(a, b):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    mask = (a >= 0) & (b >= 0)
    return adjusted_rand_score(a[mask], b[mask]), normalized_mutual_info_score(a[mask], b[mask])


contexts = seg_beats['context'].to_numpy()
rows = []

for lbl_name, labels in [('HDBSCAN+NCA', hdb_labels_nca),
                          ('DP-GMM on BEATs-UMAP', dp_labels_umap),
                          ('DP-GMM on BEATs-PCA50', dp_labels_hd)]:
    mask = labels >= 0
    n = min(8000, mask.sum())
    sil_umap = float(silhouette_score(umap_beats[mask][:n], labels[mask][:n], random_state=0)) if n > 100 else np.nan
    sil_beats = float(silhouette_score(X_beats_pca[mask][:n], labels[mask][:n], random_state=0)) if n > 100 else np.nan
    ari_c, nmi_c = ari_nmi(labels, contexts)
    cm = ctx_metrics(labels, contexts)
    rows.append({
        'method': lbl_name,
        'vocab': len(set(labels)) - (1 if -1 in labels else 0),
        'silh_UMAP_beats': round(sil_umap, 3),
        'silh_PCA50_beats': round(sil_beats, 3),
        'ari_ctx': round(ari_c, 3),
        'nmi_ctx': round(nmi_c, 3),
        'ctx_H': round(cm['mean_H'], 2),
        'ctx_purity': round(cm['context_purity'], 3),
    })

df_res = pd.DataFrame(rows)
print('\n=== BEATs EXPERIMENT (subset: top-5 bats) ===')
print(df_res.to_string(index=False))
Path('docs/thesis/figures').mkdir(parents=True, exist_ok=True)
df_res.to_csv('docs/thesis/figures/beats_subset.csv', index=False)
joblib.dump({
    'seg_beats_meta': seg_beats.drop(columns=['audio']),
    'X_beats': X_beats,
    'umap_beats': umap_beats,
    'hdb_labels_nca': hdb_labels_nca,
    'dp_labels_umap': dp_labels_umap,
    'dp_labels_hd': dp_labels_hd,
    'rows': rows,
}, CHECKPOINT_DIR / 'beats_subset_experiment.joblib', compress=3)
print('\nSaved.')
