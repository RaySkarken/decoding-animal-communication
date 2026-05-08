"""Segment loading, padding, resampling, and filtering utilities."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal as scipy_signal
from tqdm.auto import tqdm

CONTEXT_DICT = {
    0: "Unknown", 1: "Separation", 2: "Biting", 3: "Feeding",
    4: "Fighting", 5: "Grooming", 6: "Isolation", 7: "Kissing",
    8: "Landing", 9: "Mating protest", 10: "Threat-like",
    11: "General", 12: "Sleeping",
}
CONTEXT_DICT_INV = {v: k for k, v in CONTEXT_DICT.items()}

DEFAULT_EXCLUDED_CONTEXTS: Set[str] = {"General", "Sleeping", "Unknown"}
DEFAULT_MIN_EMITTER_ID = 200
DEFAULT_IQR_K = 1.5


def load_annotations(
    data_dir: Path,
    min_emitter_id: int = DEFAULT_MIN_EMITTER_ID,
    excluded_contexts: Set[str] = DEFAULT_EXCLUDED_CONTEXTS,
) -> pd.DataFrame:
    """Load and filter Annotations.csv + FileInfo.csv, return merged DataFrame."""
    ann = pd.read_csv(data_dir / "Annotations.csv", low_memory=False)

    with open(data_dir / "FileInfo.csv") as f:
        max_cols = max(len(line.split(",")) for line in f)
    fi = pd.read_csv(
        data_dir / "FileInfo.csv",
        header=None,
        names=[f"c{i}" for i in range(max_cols)],
        low_memory=False,
    )
    fi.columns = fi.iloc[0].values
    fi = fi.iloc[1:].reset_index(drop=True)
    fi["FileID"] = fi["FileID"].astype(int)
    fi = fi[["FileID", "File name", "File folder"]].drop_duplicates("FileID")

    ann["FileID"] = ann["FileID"].astype(int)
    df = ann.merge(fi, on="FileID", how="inner")

    df["Emitter"] = pd.to_numeric(df["Emitter"], errors="coerce")
    df = df.dropna(subset=["Emitter"])
    df["Emitter"] = df["Emitter"].astype(int)
    df = df[df["Emitter"] >= min_emitter_id]

    df["Context_name"] = df["Context"].map(CONTEXT_DICT)
    df = df[~df["Context_name"].isin(excluded_contexts)]
    df = df.dropna(subset=["Start sample", "End sample"])
    df["Start sample"] = df["Start sample"].astype(int)
    df["End sample"] = df["End sample"].astype(int)
    return df.reset_index(drop=True)


def load_segments(
    df: pd.DataFrame,
    data_dir: Path,
    min_audio_len: int = 100,
) -> pd.DataFrame:
    """Load audio segments from zip files referenced in *df*."""
    grouped = df.groupby("FileID")
    zip_cache: Dict[str, Optional[zipfile.ZipFile]] = {}
    segments: List[dict] = []
    skipped = 0

    def _get_zip(folder: str) -> Optional[zipfile.ZipFile]:
        if folder not in zip_cache:
            zp = data_dir / f"{folder}.zip"
            zip_cache[folder] = zipfile.ZipFile(zp, "r") if zp.exists() else None
        return zip_cache[folder]

    for file_id, group in tqdm(grouped, total=grouped.ngroups, desc="Loading WAVs"):
        row0 = group.iloc[0]
        folder = str(row0["File folder"]).strip()
        fname = str(row0["File name"]).strip()

        zf = _get_zip(folder)
        if zf is None:
            skipped += len(group)
            continue

        try:
            wav_bytes = zf.read(fname)
        except (KeyError, Exception):
            skipped += len(group)
            continue

        try:
            audio_full, file_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        except Exception:
            skipped += len(group)
            continue

        for _, r in group.iterrows():
            s, e = int(r["Start sample"]), int(r["End sample"])
            if e > len(audio_full) or s >= e:
                skipped += 1
                continue
            seg_audio = audio_full[s:e]
            if len(seg_audio) < min_audio_len:
                skipped += 1
                continue
            segments.append({
                "audio": seg_audio,
                "sr": file_sr,
                "duration_s": len(seg_audio) / file_sr,
                "context": int(r["Context"]),
                "context_name": r["Context_name"],
                "emitter": int(r["Emitter"]),
                "file_name": fname,
                "file_id": file_id,
            })

    for z in zip_cache.values():
        if z is not None:
            z.close()

    seg_df = pd.DataFrame(segments)
    print(f"Segments loaded: {len(seg_df)} | skipped: {skipped}")
    return seg_df


def dynamic_segmentation(seg_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply vocalseg dynamic threshold segmentation to split segments."""
    from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation

    sub_segments: List[dict] = []
    dyn_fail = 0

    for idx in tqdm(range(len(seg_df)), desc="Dynamic segmentation"):
        row = seg_df.iloc[idx]
        audio = row["audio"]
        rate = int(row["sr"])

        try:
            results = dynamic_threshold_segmentation(audio, rate, **params)
        except Exception:
            results = None

        if results is not None and len(results.get("onsets", [])) > 0:
            for onset_s, offset_s in zip(results["onsets"], results["offsets"]):
                s_idx = int(onset_s * rate)
                e_idx = int(offset_s * rate)
                sub_audio = audio[s_idx:e_idx]
                if len(sub_audio) < 50:
                    continue
                sub_segments.append({
                    "audio": sub_audio,
                    "sr": rate,
                    "duration_s": len(sub_audio) / rate,
                    "context": row["context"],
                    "context_name": row["context_name"],
                    "emitter": row["emitter"],
                    "file_name": row["file_name"],
                    "file_id": row["file_id"],
                })
        else:
            dyn_fail += 1
            sub_segments.append({
                "audio": audio,
                "sr": rate,
                "duration_s": len(audio) / rate,
                "context": row["context"],
                "context_name": row["context_name"],
                "emitter": row["emitter"],
                "file_name": row["file_name"],
                "file_id": row["file_id"],
            })

    result = pd.DataFrame(sub_segments)
    print(f"After dynamic segmentation: {len(result)} sub-segments | failures: {dyn_fail}")
    return result


def iqr_filter(seg_df: pd.DataFrame, k: float = DEFAULT_IQR_K) -> pd.DataFrame:
    """Remove temporal outliers using IQR on segment durations."""
    durations = seg_df["duration_s"].values
    q1, q3 = np.percentile(durations, [25, 75])
    iqr = q3 - q1
    lo = max(q1 - k * iqr, 0.001)
    hi = q3 + k * iqr
    mask = (seg_df["duration_s"] >= lo) & (seg_df["duration_s"] <= hi)
    n_before = len(seg_df)
    result = seg_df[mask].reset_index(drop=True)
    print(f"IQR filter: kept {len(result)} / {n_before} (bounds [{lo:.4f}s, {hi:.4f}s])")
    return result


def resample_audio(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Resample waveform using scipy."""
    y = np.asarray(y, dtype=np.float32)
    if sr_from == sr_to:
        return y
    n_out = max(1, int(len(y) * sr_to / sr_from))
    return scipy_signal.resample(y, n_out).astype(np.float32)


def pad_mean(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Right-pad with the segment mean (Assom-style)."""
    if len(audio) >= target_len:
        return audio[:target_len]
    pad_val = audio.mean() if len(audio) > 0 else 0.0
    return np.pad(audio, (0, target_len - len(audio)), constant_values=pad_val)
