"""CLI entrypoint to scrape articles with body, sapo, and publication time."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import NEWS_JSONL_PATH
from crawler.config import TARGET_PER_CELL
from crawler.pipeline import scrape_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Thu thập bài báo từ VnExpress, Tuổi Trẻ và Tiền Phong ra file JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=NEWS_JSONL_PATH,
        help=f"Đường dẫn file JSONL đầu ra (mặc định: {NEWS_JSONL_PATH}).",
    )
    parser.add_argument(
        "--target-per-cell",
        type=int,
        default=TARGET_PER_CELL,
        help=f"Số bài mục tiêu cho mỗi cặp báo/chuyên mục (mặc định: {TARGET_PER_CELL}).",
    )
    args = parser.parse_args()
    scrape_dataset(output=args.output, target_per_cell=args.target_per_cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
