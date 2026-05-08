#!/usr/bin/env python3
"""
Download the Egyptian fruit bat vocalizations dataset from FigShare.

- Downloads WAV zip files (files101.zip ... files224.zip) to data/raw/fruitbat/.
- Unzips into data/raw/fruitbat/zip_contents/files101, files102, ... so the
  baseline pipeline can use USE_S3=False and read from disk.

Metadata (Annotations.csv, FileInfo.csv) is not downloaded here; the notebook
fetches them from FigShare if missing. Run from project root:

  python scripts/download_figshare_fruitbat.py
  python scripts/download_figshare_fruitbat.py --max-zips 2   # quick test
  python scripts/download_figshare_fruitbat.py --data-dir /Volumes/T7/data --unzip-only   # unzip on external drive
"""

from __future__ import annotations

import argparse
import ssl
import sys
import zipfile
from pathlib import Path

# Project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# FigShare URLs for WAV zips (from decodingNonHumanCommunication/0.0 - Download-fruitbat-data.ipynb)
FIGSHARE_ZIPS = [
    ("https://ndownloader.figshare.com/files/8879545", "files101.zip"),
    ("https://ndownloader.figshare.com/files/8879548", "files102.zip"),
    ("https://ndownloader.figshare.com/files/8879596", "files103.zip"),
    ("https://ndownloader.figshare.com/files/8879572", "files104.zip"),
    ("https://ndownloader.figshare.com/files/8879554", "files105.zip"),
    ("https://ndownloader.figshare.com/files/8879578", "files106.zip"),
    ("https://ndownloader.figshare.com/files/8879431", "files201.zip"),
    ("https://ndownloader.figshare.com/files/8879536", "files202.zip"),
    ("https://ndownloader.figshare.com/files/8879521", "files203.zip"),
    ("https://ndownloader.figshare.com/files/8879428", "files204.zip"),
    ("https://ndownloader.figshare.com/files/8879533", "files205.zip"),
    ("https://ndownloader.figshare.com/files/8879425", "files206.zip"),
    ("https://ndownloader.figshare.com/files/8879392", "files207.zip"),
    ("https://ndownloader.figshare.com/files/8879404", "files208.zip"),
    ("https://ndownloader.figshare.com/files/8879338", "files209.zip"),
    ("https://ndownloader.figshare.com/files/8879683", "files210.zip"),
    ("https://ndownloader.figshare.com/files/8879179", "files211.zip"),
    ("https://ndownloader.figshare.com/files/8879287", "files212.zip"),
    ("https://ndownloader.figshare.com/files/8879659", "files213.zip"),
    ("https://ndownloader.figshare.com/files/8879674", "files214.zip"),
    ("https://ndownloader.figshare.com/files/8879662", "files215.zip"),
    ("https://ndownloader.figshare.com/files/8879641", "files216.zip"),
    ("https://ndownloader.figshare.com/files/8879632", "files217.zip"),
    ("https://ndownloader.figshare.com/files/8879653", "files218.zip"),
    ("https://ndownloader.figshare.com/files/8879617", "files219.zip"),
    ("https://ndownloader.figshare.com/files/8879623", "files220.zip"),
    ("https://ndownloader.figshare.com/files/8879611", "files221.zip"),
    ("https://ndownloader.figshare.com/files/8879608", "files222.zip"),
    ("https://ndownloader.figshare.com/files/8879599", "files223.zip"),
    ("https://ndownloader.figshare.com/files/8879602", "files224.zip"),
]


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_url(url: str, path: Path, ctx: ssl.SSLContext) -> None:
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    req = urllib.request.Request(url, headers={"User-Agent": "AnimalCommunication-download/1.0"})
    with opener.open(req) as resp:
        path.write_bytes(resp.read())


def download_with_progress(url: str, path: Path, ctx: ssl.SSLContext) -> None:
    try:
        import urllib.request
        from tqdm import tqdm

        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        req = urllib.request.Request(url, headers={"User-Agent": "AnimalCommunication-download/1.0"})
        with opener.open(req) as resp:
            total = int(resp.headers.get("Content-Length", 0)) or None
            with tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as pbar:
                chunk_size = 1 << 20  # 1 MiB
                with path.open("wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
    except ImportError:
        download_url(url, path, ctx)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download fruit bat dataset from FigShare")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root data directory (e.g. /Volumes/T7/data). Default: PROJECT_ROOT/data.",
    )
    parser.add_argument(
        "--max-zips",
        type=int,
        default=None,
        help="Only download this many zips (default: all 24). Use 1–2 for a quick test.",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Only download zips; do not unzip.",
    )
    parser.add_argument(
        "--unzip-only",
        action="store_true",
        help="Only unzip existing zips; do not download.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    RAW_DIR = data_root / "raw" / "fruitbat"
    ZIP_CONTENTS = RAW_DIR / "zip_contents"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_CONTENTS.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {data_root}")

    to_do = FIGSHARE_ZIPS
    if args.max_zips is not None:
        to_do = to_do[: args.max_zips]
        print(f"Limiting to first {args.max_zips} zip(s).")

    ctx = ssl_context()

    if not args.unzip_only:
        for url, name in to_do:
            zip_path = RAW_DIR / name
            if zip_path.exists():
                print(f"Skip download (exists): {name}")
                continue
            print(f"Downloading {name} ...")
            try:
                download_with_progress(url, zip_path, ctx)
            except Exception as e:
                print(f"Error downloading {name}: {e}", file=sys.stderr)
                if zip_path.exists():
                    zip_path.unlink()
                return 1

    if not args.no_unzip:
        for _url, name in to_do:
            zip_path = RAW_DIR / name
            out_dir = ZIP_CONTENTS / name.replace(".zip", "")
            if not zip_path.exists():
                print(f"Skip unzip (no zip): {name}")
                continue
            if out_dir.exists() and next(out_dir.iterdir(), None) is not None:
                print(f"Skip unzip (exists): {name}")
                continue
            print(f"Unzipping {name} -> {out_dir.name}/")
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(out_dir)

    print("Done. Set USE_S3=False in the baseline notebook; DATA_ROOT =", repr(str(data_root)), "if data is on external drive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
