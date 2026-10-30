"""Backfill published_at from article-page metadata.

Run from the project root:
    python -m database.backfill_published_at --limit 20

After checking the sample, process the remaining rows with:
    python -m database.backfill_published_at --limit 0

Use --dry-run to inspect without saving any dates.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from sqlalchemy import or_

from database.database_15k import SessionLocal, Article


USER_AGENT = (
    "Mozilla/5.0 (compatible; NewsReaderResearchBot/1.0; "
    "+https://example.org/bot-info)"
)
TIMEOUT_SECONDS = 15

# Highest priority first. Avoid dateModified: it is not the publication date.
META_DATE_SELECTORS = [
    ("property", "article:published_time"),
    ("property", "og:article:published_time"),
    ("name", "pubdate"),
    ("name", "publishdate"),
    ("name", "publish_date"),
    ("name", "publication_date"),
    ("name", "datepublished"),
    ("name", "date"),
    ("name", "dc.date.issued"),
    ("name", "dc.date"),
    ("itemprop", "datePublished"),
]


def parse_published_datetime(value: Any) -> datetime | None:
    """Parse an explicit publication date; return UTC-naive if timezone exists."""
    if not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    try:
        parsed = date_parser.parse(value, fuzzy=False)
    except (ValueError, TypeError, OverflowError):
        return None

    # Normalize timezone-aware values to UTC. Store as naive datetime because
    # SQLite's ordinary DATETIME column does not retain timezone information.
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)

    # Reject implausible dates rather than persisting accidental metadata.
    if parsed.year < 1990 or parsed > datetime.now(timezone.utc).replace(tzinfo=None):
        return None

    return parsed


def iter_jsonld_date_values(node: Any):
    """Yield datePublished values from nested JSON-LD objects and arrays."""
    if isinstance(node, dict):
        # datePublished is preferred. dateCreated is a fallback when publishers
        # omit datePublished. Do not use dateModified as a publication date.
        for key in ("datePublished", "dateCreated"):
            value = node.get(key)
            if isinstance(value, str):
                yield key, value

        for value in node.values():
            if isinstance(value, (dict, list)):
                yield from iter_jsonld_date_values(value)

    elif isinstance(node, list):
        for value in node:
            yield from iter_jsonld_date_values(value)


def extract_published_at(html: str) -> tuple[datetime | None, str | None]:
    """Extract the best publication date from common article-page metadata."""
    soup = BeautifulSoup(html, "html.parser")

    # 1. Well-known meta tags.
    for attr, expected_value in META_DATE_SELECTORS:
        tag = soup.find("meta", attrs={attr: expected_value})
        if tag:
            raw_date = tag.get("content") or tag.get("value")
            parsed = parse_published_datetime(raw_date)
            if parsed:
                return parsed, f"meta:{attr}={expected_value}"

    # 2. Schema.org JSON-LD. Prefer all datePublished values before dateCreated.
    jsonld_candidates: list[tuple[str, str]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw_json = script.string or script.get_text(strip=True)
        if not raw_json:
            continue
        try:
            data = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            continue
        jsonld_candidates.extend(iter_jsonld_date_values(data))

    for preferred_key in ("datePublished", "dateCreated"):
        for key, raw_date in jsonld_candidates:
            if key == preferred_key:
                parsed = parse_published_datetime(raw_date)
                if parsed:
                    return parsed, f"jsonld:{key}"

    # 3. A semantic <time> element explicitly marked as publication date.
    time_tag = soup.find("time", attrs={"itemprop": "datePublished"})
    if time_tag:
        parsed = parse_published_datetime(time_tag.get("datetime") or time_tag.get_text(" ", strip=True))
        if parsed:
            return parsed, "time:itemprop=datePublished"

    # Some publishers have a generic <time datetime="...">. Use this only
    # after explicit publication-date metadata was not found.
    for time_tag in soup.find_all("time", limit=10):
        raw_date = time_tag.get("datetime")
        parsed = parse_published_datetime(raw_date)
        if parsed:
            return parsed, "time:datetime"

    return None, None


def fetch_published_at(url: str, http: requests.Session) -> tuple[datetime | None, str | None, str]:
    """Fetch one article URL and extract its publication date."""
    try:
        response = http.get(
            url,
            timeout=TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return None, None, f"HTTP error: {exc.__class__.__name__}: {exc}"

    published_at, method = extract_published_at(response.text)
    if published_at is None:
        return None, None, f"No publication date found (HTTP {response.status_code}; {response.url})"

    return published_at, method, f"OK ({response.status_code}; {response.url})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Lấy ngày xuất bản từ URL bài báo để cập nhật SQLite.")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Số bài xử lý tối đa; dùng 0 để xử lý tất cả bài chưa có ngày (mặc định: 20).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Số giây nghỉ giữa các request (mặc định: 1.0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ xem kết quả, không ghi vào database.",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit phải >= 0")
    if args.delay < 0:
        parser.error("--delay phải >= 0")

    db = SessionLocal()
    http = requests.Session()
    http.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "vi,en-US;q=0.8,en;q=0.6",
    })

    try:
        query = (
            db.query(Article)
            .filter(
                Article.published_at.is_(None),
                Article.url.isnot(None),
                Article.url != "",
            )
            .order_by(Article.id)
        )
        if args.limit > 0:
            query = query.limit(args.limit)

        articles = query.all()
        print(f"Số bài cần kiểm tra: {len(articles)}")
        print("Chế độ:", "DRY RUN (không ghi DB)" if args.dry_run else "GHI DATABASE")

        success = 0
        failed = 0

        for number, article in enumerate(articles, start=1):
            title = article.title or "(không có tiêu đề)"
            print(f"\n[{number}/{len(articles)}] {title}")
            print("URL:", article.url)

            published_at, method, status = fetch_published_at(article.url, http)
            print("Kết quả:", status)

            if published_at is None:
                failed += 1
            else:
                success += 1
                print("Ngày xuất bản:", published_at.isoformat(sep=" "))
                print("Nguồn metadata:", method)
                if not args.dry_run:
                    article.published_at = published_at
                    db.commit()

            if number < len(articles) and args.delay:
                time.sleep(args.delay)

        print("\n" + "=" * 60)
        print(f"Đã lấy được ngày: {success}")
        print(f"Chưa lấy được ngày: {failed}")
        print(f"Chế độ dry-run: {args.dry_run}")
        print("=" * 60)

        if args.dry_run:
            print("Không có thay đổi nào được ghi vào database.")

    finally:
        http.close()
        db.close()


if __name__ == "__main__":
    main()
