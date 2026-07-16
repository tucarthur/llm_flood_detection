"""Download the exemplar bank's images that the image-RAG retriever needs but that
scripts/build_test_set.py never fetched.

The bank (knowledge_base/image_bank_backup/metadata.parquet, built in
knowledge_base/build_image_bank.ipynb) has 1,592 rows: 796 "rare" (medium/high/flood)
+ 796 "low_diverse" (FPS-selected). The "rare" rows are identical to the test set's
rare rows (both keep every rare image), so they're already sitting in
data/raw/test_images/. The "low_diverse" rows are FPS-selected -- disjoint from the
test set's randomly-sampled lows -- and were never downloaded, which meant
ImageBankRetriever.dispatch() silently returned 0 image payloads for any retrieved
low exemplar (see agent/image_retrieval.py; confirmed live: 0/5 payloads attached).

Downloads land in --images-dir using the same flattened-name convention as
data.enoe_images.local_image_path, so ImageBankRetriever's default images_dir can
point straight at it. Re-running skips already-downloaded files.

Usage:
    python -m scripts.download_bank_images
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from data.enoe_images import KAGGLE_PREFIX
from scripts.build_test_set import download_one

BANK_METADATA = Path("knowledge_base/image_bank_backup/metadata.parquet")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="knowledge_base/image_bank_images")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    bank = pd.read_parquet(BANK_METADATA)
    bank["kaggle_path"] = KAGGLE_PREFIX + bank["path"]
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for _, row in bank.iterrows()]
    print(f"downloading {len(rows)} bank images to {images_dir} ({args.workers} workers)")
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, row, images_dir) for row in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            msg, ok = fut.result()
            if not ok:
                failures.append(msg)
            if i % 100 == 0 or i == len(rows):
                print(f"downloaded {i}/{len(rows)} ({len(failures)} failures)", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} downloads FAILED (rerun to retry):", file=sys.stderr)
        for msg in failures[:20]:
            print(f"  {msg}", file=sys.stderr)
        sys.exit(1)
    print("all downloads complete")


if __name__ == "__main__":
    main()
