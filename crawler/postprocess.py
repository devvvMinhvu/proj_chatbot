"""CLI entrypoint to backfill missing article bodies (and optionally dates) in JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import NEWS_JSONL_PATH
from crawler.pipeline import fill_missing_bodies


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bổ sung toàn văn bài báo (và tùy chọn ngày đăng) bị thiếu trong file JSONL.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=NEWS_JSONL_PATH,
        help=f"Đường dẫn file JSONL cần xử lý (mặc định: {NEWS_JSONL_PATH}).",
    )
    parser.add_argument(
        "--fill-dates",
        action="store_true",
        help="Bổ sung thêm trường published_at cho các bài chưa có ngày đăng.",
    )
    args = parser.parse_args()
    counts = fill_missing_bodies(path=args.input, fill_dates=args.fill_dates)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
