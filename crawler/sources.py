"""Publisher-specific archive candidate streams for VnExpress, Tuổi Trẻ, and Tiền Phong."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import time as time_module
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from crawler.config import (
    MAX_ARCHIVE_PAGES,
    MAX_LOOKBACK_DAYS,
    PAGE_PAUSE,
    TIMEZONE,
    TP_CATEGORY_SLUGS,
    TP_ZONE_IDS,
    TTO_CATEGORY_SLUGS,
    TTO_ZONE_IDS,
    VNE_CATEGORY_IDS,
)
from crawler.http_client import clean, fetch_listing, fetch_tienphong_api, normalize_article_url


def make_vne_day_url(category: str, day: date) -> str:
    """Build a VnExpress daily archive URL for a given category and date."""
    category_id = VNE_CATEGORY_IDS[category]
    start = int(datetime.combine(day, time.min, tzinfo=TIMEZONE).timestamp())
    end = int(datetime.combine(day, time(23, 59, 59), tzinfo=TIMEZONE).timestamp())
    return (
        f"https://vnexpress.net/category/day/cateid/{category_id}"
        f"/fromdate/{start}/todate/{end}/allcate/{category_id}"
    )


def vne_listing_page(html: str, page_url: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Parse article links and pagination URLs from a VnExpress archive page."""
    soup = BeautifulSoup(html, "html.parser")
    records = []
    for item in soup.select("article.item-news"):
        link = item.select_one("h3.title-news a[href], .title-news a[href]")
        if link is None:
            continue
        url = normalize_article_url(link.get("href", ""), page_url, "vnexpress")
        if not url:
            continue
        title = clean(link.get_text(" ", strip=True))
        records.append((title, url, ""))
    pages = []
    for link in soup.select("#pagination a[href]"):
        href = link.get("href", "")
        if href and not href.startswith("javascript:"):
            pages.append(urljoin(page_url, href))
    return records, list(dict.fromkeys(pages))


def iter_vne_candidates(category: str):
    """Walk VnExpress archives backwards by day until enough articles are found."""
    day = datetime.now(TIMEZONE).date()
    for days_back in range(MAX_LOOKBACK_DAYS):
        current_day = day - timedelta(days=days_back)
        first_url = make_vne_day_url(category, current_day)
        try:
            html = fetch_listing(first_url)
        except Exception as error:
            print(f"VnExpress archive error {category} {current_day}: {error}", flush=True)
            continue
        records, page_urls = vne_listing_page(html, first_url)
        yield from records
        for page_url in page_urls:
            if "/page/" not in page_url:
                continue
            try:
                page_html = fetch_listing(page_url, first_url)
                extra, _ = vne_listing_page(page_html, page_url)
                yield from extra
            except Exception as error:
                print(f"VnExpress page error {category} {current_day}: {error}", flush=True)
            time_module.sleep(PAGE_PAUSE)
        if days_back and days_back % 10 == 0:
            print(f"VnExpress {category}: archived through {current_day}", flush=True)
        time_module.sleep(PAGE_PAUSE)
    raise RuntimeError(f"VnExpress archive exhausted {MAX_LOOKBACK_DAYS} days for {category}")


def iter_tto_candidates(category: str):
    """Read Tuổi Trẻ's numbered category pages until the listing is empty."""
    landing = f"https://tuoitre.vn/{TTO_CATEGORY_SLUGS[category]}.htm"
    for page in range(1, MAX_ARCHIVE_PAGES + 1):
        page_url = f"https://tuoitre.vn/timeline/{TTO_ZONE_IDS[category]}/trang-{page}.htm"
        try:
            html = fetch_listing(page_url, landing)
        except Exception as error:
            print(f"Tuổi Trẻ page error {category} page {page}: {error}", flush=True)
            if page > 1:
                return
            raise
        soup = BeautifulSoup(html, "html.parser")
        page_records = []
        for link in soup.select(".box-category-item .box-category-link-title[href]"):
            url = normalize_article_url(link.get("href", ""), page_url, "tuoitre")
            if url:
                page_records.append((clean(link.get_text(" ", strip=True)), url, ""))
        if not page_records:
            return
        yield from page_records
        if page % 10 == 0:
            print(f"Tuổi Trẻ {category}: archive page {page}", flush=True)
        time_module.sleep(PAGE_PAUSE)
    raise RuntimeError(f"Tuổi Trẻ archive exceeded {MAX_ARCHIVE_PAGES} pages for {category}")


def iter_tp_candidates(category: str):
    """Read Tiền Phong's numbered JSON listings, including their article summaries."""
    landing = f"https://tienphong.vn/{TP_CATEGORY_SLUGS[category]}/"
    zone = TP_ZONE_IDS[category]
    for page in range(1, MAX_ARCHIVE_PAGES + 1):
        api_url = f"https://api.tienphong.vn/api/morenews-zone-{zone}-{page}.html?phrase=&page_size=20"
        data = fetch_tienphong_api(api_url, landing)
        rows = data.get("data", {}).get("contents", [])
        if not rows:
            return
        for item in rows:
            url = normalize_article_url(item.get("url", ""), landing, "tienphong")
            if not url:
                continue
            yield (clean(item.get("title")), url, item.get("description", ""))
        if page % 10 == 0:
            print(f"Tiền Phong {category}: archive page {page}", flush=True)
        time_module.sleep(PAGE_PAUSE)
    raise RuntimeError(f"Tiền Phong archive exceeded {MAX_ARCHIVE_PAGES} pages for {category}")


def candidate_stream(source: str, category: str):
    """Dispatch candidate article stream for a publisher/category pair."""
    if source == "vnexpress":
        return iter_vne_candidates(category)
    if source == "tuoitre":
        return iter_tto_candidates(category)
    if source == "tienphong":
        return iter_tp_candidates(category)
    raise ValueError(f"Unknown source: {source}")
