"""Dataset scraping and post-processing pipelines."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
import time as time_module

from bs4 import BeautifulSoup

from config import NEWS_JSONL_PATH
from crawler.config import ARTICLE_PAUSE, CATEGORIES, PREFIX, SOURCES, TARGET_PER_CELL
from crawler.extractor import extract_article, extract_body, extract_published_at, parse_published_at
from crawler.http_client import fetch_html, site_for
from crawler.sources import candidate_stream


def process_cell(
    source: str,
    category: str,
    source_seen: set[str],
    destination,
    write_lock,
    target_per_cell: int = TARGET_PER_CELL,
) -> int:
    """Write the requested number of usable articles for one publisher/category pair."""
    count = 0
    skipped = 0
    for listing_title, url, listing_description in candidate_stream(source, category):
        if count >= target_per_cell:
            break
        if not url or url in source_seen:
            continue
        source_seen.add(url)
        try:
            html = fetch_html(url)
            title, body, sapo, published_at = extract_article(html, url, listing_title, listing_description)
            if not all(value.strip() for value in (title, body, sapo)):
                raise ValueError("Empty article field")
            if not published_at or parse_published_at(published_at) != published_at:
                raise ValueError("Invalid publication timestamp")
            if "\ufffd" in sapo or "\ufffd" in body:
                raise ValueError("Decoding replacement character in article")
        except Exception as error:
            skipped += 1
            if skipped <= 20 or skipped % 50 == 0:
                print(f"Skipped {source}/{category} {url}: {error}", flush=True)
            time_module.sleep(ARTICLE_PAUSE)
            continue

        row = {
            "id": f"{PREFIX[source]}_{category}_{count + 1:04d}",
            "source": source,
            "category": category,
            "split": "",
            "title": title,
            "url": url,
            "body": body,
            "sapo": sapo,
            "published_at": published_at,
        }
        with write_lock:
            # Only one publisher can write at a time, keeping each JSON line intact.
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1
        if count % 100 == 0 or count == target_per_cell:
            print(f"{source}/{category}: {count}/{target_per_cell}; latest={title[:90]}", flush=True)
        time_module.sleep(ARTICLE_PAUSE)

    if count != target_per_cell:
        raise RuntimeError(f"Only sourced {count}/{target_per_cell} valid records for {source}/{category}")
    return count


def process_source(
    source: str,
    destination,
    write_lock,
    target_per_cell: int = TARGET_PER_CELL,
) -> int:
    """Scrape all configured categories for one publisher."""
    # Share seen URLs across categories so one article is not saved twice.
    source_seen: set[str] = set()
    count = 0
    for category in CATEGORIES:
        count += process_cell(
            source,
            category,
            source_seen,
            destination,
            write_lock,
            target_per_cell=target_per_cell,
        )
    return count


def scrape_dataset(
    output: Path = NEWS_JSONL_PATH,
    target_per_cell: int = TARGET_PER_CELL,
) -> int:
    """Scrape articles from all configured publishers into a new JSONL dataset."""
    output = Path(output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_total = 0
    write_lock = Lock()
    # Save each article immediately; a failed run leaves a partial output.
    with output.open("x", encoding="utf-8", newline="\n", buffering=1) as destination:
        with ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
            futures = [
                pool.submit(process_source, source, destination, write_lock, target_per_cell)
                for source in SOURCES
            ]
            for future in as_completed(futures):
                rows_total += future.result()
    print(f"BUILD_COMPLETE {output} rows={rows_total}", flush=True)
    return rows_total


def fill_missing_bodies(
    path: Path = NEWS_JSONL_PATH,
    fill_dates: bool = False,
) -> dict[str, int]:
    """Fill missing article bodies (and optionally publication dates) in-place safely."""
    path = Path(path)
    initial_stat = path.stat()
    counts = {"extracted": 0, "dates_filled": 0, "skipped": 0, "failed": 0}
    # Keep the temporary file beside the original so replacement stays on one filesystem.
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as destination, path.open(encoding="utf-8", newline="") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    destination.write(line)
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Line {number} must be a JSON object")

                needs_body = not record.get("body")
                needs_date = fill_dates and not parse_published_at(record.get("published_at"))
                if not needs_body and not needs_date:
                    counts["skipped"] += 1
                    destination.write(line)
                    continue

                try:
                    url = record.get("url")
                    if not isinstance(url, str):
                        raise ValueError("Missing or invalid url")
                    html = fetch_html(url)
                    new_body = extract_body(html, url) if needs_body else None
                    new_published_at = (
                        extract_published_at(BeautifulSoup(html, "html.parser"), site_for(url))
                        if needs_date
                        else None
                    )
                except Exception as error:
                    # Keep failed rows unchanged so they can be retried later.
                    counts["failed"] += 1
                    print(f"Failed {record.get('url')}: {error}")
                    destination.write(line)
                else:
                    if new_body is not None:
                        record["body"] = new_body
                        counts["extracted"] += 1
                    if new_published_at is not None:
                        record["published_at"] = new_published_at
                        counts["dates_filled"] += 1
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                time_module.sleep(0.25)
            destination.flush()
            os.fsync(destination.fileno())

        if counts["extracted"] or counts["dates_filled"]:
            current_stat = path.stat()
            if (current_stat.st_size, current_stat.st_mtime_ns) != (initial_stat.st_size, initial_stat.st_mtime_ns):
                raise ValueError("Input changed during processing; refusing to overwrite it")
            # Replace only after writing succeeds; failures leave the original intact.
            os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

    print(f"Finished: {counts}")
    return counts
