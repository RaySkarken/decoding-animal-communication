#!/usr/bin/env python3
"""
Unzip FigShare fruit bat archives with progress bars.

Examples:
  python scripts/unzip_figshare_fruitbat.py --data-dir /Volumes/T7/data
  python scripts/unzip_figshare_fruitbat.py --data-dir /Volumes/T7/data --only files103,files104
  python scripts/unzip_figshare_fruitbat.py --data-dir /Volumes/T7/data --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def zip_sort_key(path: Path) -> int:
    stem = path.stem  # files101
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def parse_only(value: str | None) -> set[str]:
    if not value:
        return set()
    items = [x.strip() for x in value.split(",") if x.strip()]
    normalized = set()
    for item in items:
        normalized.add(item.replace(".zip", ""))
    return normalized


def extract_zip_with_progress(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        total_bytes = sum(m.file_size for m in members)

        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc=zip_path.name,
            leave=True,
        ) as pbar:
            for member in members:
                zf.extract(member, out_dir)
                pbar.update(member.file_size)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unzip fruit bat files*.zip archives with progress bars."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root data dir (e.g. /Volumes/T7/data). Default: PROJECT_ROOT/data",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated subset (e.g. files103,files104 or files103.zip,files104.zip).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if target folder already exists and is non-empty.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per archive on extraction error (default: 3).",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=2.0,
        help="Seconds to wait between retries (default: 2.0).",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    raw_dir = data_root / "raw" / "fruitbat"
    zip_contents = raw_dir / "zip_contents"
    zip_contents.mkdir(parents=True, exist_ok=True)

    only = parse_only(args.only)

    zip_paths = sorted(raw_dir.glob("files*.zip"), key=zip_sort_key)
    if only:
        zip_paths = [p for p in zip_paths if p.stem in only]

    if not zip_paths:
        print(f"No archives found in {raw_dir} matching selection.")
        return 1

    print(f"Data root: {data_root}")
    print(f"Archives to process: {len(zip_paths)}")

    ok: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for zip_path in tqdm(zip_paths, desc="Overall", unit="zip"):
        stem = zip_path.stem
        out_dir = zip_contents / stem

        if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
            skipped.append(stem)
            continue

        success = False
        for attempt in range(1, args.retries + 1):
            try:
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                extract_zip_with_progress(zip_path, out_dir)
                ok.append(stem)
                success = True
                break
            except Exception as exc:
                print(f"[WARN] {stem} attempt {attempt}/{args.retries}: {exc}")
                if attempt < args.retries:
                    time.sleep(args.retry_sleep)
                else:
                    failed.append(stem)
                    shutil.rmtree(out_dir, ignore_errors=True)

        if not success:
            continue

    print("\nDone.")
    print(f"  Extracted: {len(ok)}")
    print(f"  Skipped:   {len(skipped)}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        print("  Failed list:", ",".join(failed))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
