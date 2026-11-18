"""HTTP fetching, URL normalization, and publisher redirect validation."""

from __future__ import annotations

import gzip
from http.client import HTTPException
import json
import time as time_module
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen
import zlib

from crawler.config import HOST, SITES


def clean(value: str | None) -> str:
    """Collapse whitespace and strip surrounding spaces."""
    return " ".join((value or "").split()).strip()


def site_for(url: str) -> str:
    """Validate an article URL against supported publisher domains and return its host."""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").removeprefix("www.")
    if (
        parsed.scheme not in ("http", "https")
        or host not in SITES
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 80, 443)
    ):
        raise ValueError(f"Unsupported article URL: {url}")
    return host


def normalize_article_url(raw_url: str, base_url: str, source: str) -> str | None:
    """Resolve a candidate link and ensure it belongs to the expected publisher."""
    url = urljoin(base_url, clean(raw_url))
    try:
        host = site_for(url)
    except (TypeError, ValueError):
        return None
    return url if host == HOST[source] else None


class NewsRedirectHandler(HTTPRedirectHandler):
    """Reject cross-publisher redirects while downloading articles."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if site_for(newurl) != site_for(request.full_url):
            raise ValueError("Article redirected to a different publisher")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def decode_response(response, errors: str = "strict") -> str:
    """Decompress gzip/deflate payloads and decode text using response charset."""
    data = response.read()
    encoding = response.headers.get("Content-Encoding", "").lower()
    if encoding == "gzip" or data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    elif encoding == "deflate":
        data = zlib.decompress(data)
    elif encoding not in ("", "identity"):
        raise ValueError(f"Unsupported content encoding: {encoding}")
    return data.decode(response.headers.get_content_charset() or "utf-8", errors)


def fetch_html(url: str) -> str:
    """Download an article, retrying temporary network and server failures."""
    retries = 3
    site_for(url)
    opener = build_opener(NewsRedirectHandler())
    for attempt in range(retries):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; NewsDataset/1.0)",
                    "Accept": "text/html",
                    "Accept-Encoding": "gzip, deflate",
                },
            )
            with opener.open(request, timeout=30) as response:
                if response.headers.get_content_type() not in ("text/html", "application/xhtml+xml"):
                    raise ValueError("Response is not HTML")
                return decode_response(response)
        except (OSError, HTTPException, EOFError) as error:
            if isinstance(error, HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == retries - 1:
                raise
            delay = min(2**attempt, 30)
            if isinstance(error, HTTPError):
                retry_after = error.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    delay = min(int(retry_after), 60)
            time_module.sleep(delay)
    raise RuntimeError(f"Failed to fetch HTML: {url}")


def fetch_listing(url: str, referer: str | None = None) -> str:
    """Fetch an archive or timeline listing page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; NewsDataset/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/json,*/*",
        "Accept-Encoding": "gzip, deflate",
    }
    if referer:
        headers["Referer"] = referer
    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise ValueError(f"Archive returned HTTP {response.status}: {url}")
        return decode_response(response, errors="replace")


def fetch_tienphong_api(url: str, referer: str) -> dict:
    """Query Tiền Phong's JSON archive endpoint with retries."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Referer": referer,
        "Origin": "https://tienphong.vn",
        "X-Requested-With": "XMLHttpRequest",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }
    for attempt in range(3):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=30) as response:
                data = json.loads(decode_response(response, errors="replace"))
            if data.get("error_code"):
                raise ValueError(f"Tiền Phong API error {data['error_code']}: {data.get('error_message', '')}")
            return data
        except HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise
            time_module.sleep(2**attempt)
    raise RuntimeError(f"Could not read Tiền Phong archive: {url}")
