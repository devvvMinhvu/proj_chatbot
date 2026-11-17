"""HTML parsing and extraction of article title, body, sapo, and publication time."""

from __future__ import annotations

from datetime import datetime
import html as html_lib
import json
import re

from bs4 import BeautifulSoup, Comment
from bs4.element import NavigableString

from crawler.config import BLOCKS, NOISE, REMOVE, SITES, TIMEZONE
from crawler.http_client import clean, site_for


def text_blocks(node) -> str:
    """Read HTML text without joining separate paragraphs or splitting inline words."""
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return str(node)
    if node.name == "br":
        return "\n"
    text = "".join(text_blocks(child) for child in node.children)
    if node.name in ("td", "th"):
        return text + "\t"
    return "\n" + text + "\n" if node.name in BLOCKS else text


def clean_text(node) -> str:
    """Format extracted HTML blocks into clean paragraph-separated text."""
    return "\n\n".join(" ".join(line.split()) for line in text_blocks(node).splitlines() if line.strip())


def strip_publisher_suffix(value: str) -> str:
    """Remove trailing publisher branding from metadata descriptions."""
    return re.sub(r"\s+-\s+(?:VnExpress|Tuổi Trẻ|Tiền Phong)(?: Online)?$", "", value).strip()


def generic_description(value: str) -> bool:
    """Recognize site slogans that describe the newspaper instead of the article."""
    value = value.casefold()
    return any(
        phrase in value
        for phrase in (
            "vnexpress tin tức mới nhất",
            "tin tức mới nhất trên các lĩnh vực chính trị, xã hội, văn hóa, thể thao",
            "báo điện tử tiền phong",
            "tuổi trẻ online - tin tức mới nhất",
        )
    )


def valid_exact_sapo(value: str) -> bool:
    """Validate an introduction extracted from the dedicated sapo DOM element."""
    return len(value) >= 25 and len(value.split()) >= 5 and not generic_description(value)


def valid_fallback_sapo(value: str) -> bool:
    """Validate a fallback introduction extracted from metadata or leading paragraph."""
    return (
        45 <= len(value) <= 1200
        and len(value.split()) >= 8
        and any(mark in value for mark in ".!?;:")
        and not generic_description(value)
    )


def parse_published_at(value: str | None) -> str | None:
    """Return a full timestamp in Vietnam time, or None when no time is available."""
    value = html_lib.unescape(clean(value))
    # A date alone does not tell us the publication time; do not invent midnight.
    if not re.search(r"\d{1,2}:\d{2}", value):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        patterns = (
            r"(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>\d{4})\D+(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?",
            r"(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})[T ,]+(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?",
        )
        for pattern in patterns:
            match = re.search(pattern, value)
            if not match:
                continue
            try:
                parsed = datetime(**{key: int(val) for key, val in match.groupdict(default="0").items()})
            except ValueError:
                return None
            break
        else:
            return None
    if parsed.tzinfo is None:
        # These publishers display local Vietnam time when no offset is included.
        parsed = parsed.replace(tzinfo=TIMEZONE)
    return parsed.astimezone(TIMEZONE).isoformat(timespec="seconds")


def extract_published_at(soup: BeautifulSoup, host: str) -> str:
    """Try publication metadata first, then dates displayed on the article page."""
    selectors = (
        'meta[property="article:published_time"]',
        'meta[itemprop="datePublished"]',
        'meta[name="pubdate"]',
        'meta[name="datepublished"]',
        'meta[name="publishdate"]',
        'meta[name="date"]',
    )
    for selector in selectors:
        for node in soup.select(selector):
            value = parse_published_at(node.get("content"))
            if value:
                return value

    def dates_in_json(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.casefold() == "datepublished" and isinstance(child, str):
                    parsed = parse_published_at(child)
                    if parsed:
                        yield parsed
                yield from dates_in_json(child)
        elif isinstance(value, list):
            for child in value:
                yield from dates_in_json(child)

    for node in soup.select('script[type="application/ld+json"]'):
        try:
            for value in dates_in_json(json.loads(node.string or node.get_text())):
                return value
        except (json.JSONDecodeError, TypeError):
            continue

    for node in soup.select("time[datetime]"):
        value = parse_published_at(node.get("datetime")) or parse_published_at(node.get_text(" ", strip=True))
        if value:
            return value

    site_selectors = {
        "vnexpress.net": ("span.date", ".date", ".time-public"),
        "tuoitre.vn": (".date-time", ".detail-time", "time"),
        "tienphong.vn": ("time", ".article__date", ".date"),
    }
    for selector in site_selectors[host]:
        for node in soup.select(selector):
            value = parse_published_at(node.get_text(" ", strip=True))
            if value:
                return value
    raise ValueError(f"No publication time found for {host} article")


def extract_body(html: str, url: str) -> str:
    """Keep article text and its introduction; remove ads, navigation, and related links."""
    host = site_for(url)
    soup = BeautifulSoup(html, "html.parser")
    selector, lead_selector = SITES[host]
    body = soup.select_one(selector)
    if body is None and host == "tienphong.vn":
        body = soup.select_one(".article .video-player.cms-body")
    if body is None and host == "vnexpress.net":
        body = soup.select_one(".section-content .wrapper-content")
        if body:
            for node in body.select(".people, .d-md-none"):
                node.decompose()
    if body is None:
        raise ValueError(f"Missing article container {selector}")
    if host == "vnexpress.net":
        for location in body.select(".location-stamp"):
            location.insert_after(" ")
        end = body.select_one("#article-end")
        if end:
            # Everything after this marker belongs outside the article.
            tail = end
            while tail is not body:
                for sibling in list(tail.next_siblings):
                    sibling.extract()
                tail = tail.parent
            end.decompose()
        for author in body.select("p[style]"):
            name = author.find("strong")
            if (
                "text-align:right" in author["style"].replace(" ", "")
                and name is not None
                and author.get_text(strip=True) == name.get_text(strip=True)
            ):
                author.decompose()
    lead = soup.select_one(lead_selector)
    media_article = host == "tienphong.vn" and body.select_one("video, iframe, figure") is not None
    for node in body.select(NOISE + ", " + REMOVE[host] + ", h1"):
        node.decompose()
    if not body.select_one("p, h2, h3, li, figcaption, td, blockquote") and not (media_article and lead):
        raise ValueError("Article container has no readable article blocks")
    content = clean_text(body)
    if lead is not None and lead is not body and body not in lead.parents:
        for node in lead.select(NOISE):
            node.decompose()
        introduction = clean_text(lead)
        if introduction and not content.startswith(introduction):
            content = introduction + ("\n\n" + content if content else "")
    if not content:
        raise ValueError("Article body is empty")
    if "\ufffd" in content:
        raise ValueError("Article text contains decoding replacement characters")
    return content


def extract_sapo(soup: BeautifulSoup, host: str, body: str, listing_description: str) -> str:
    """Find the article introduction, using page metadata or body text as fallbacks."""
    lead_selector = SITES[host][1]
    lead = soup.select_one(lead_selector)
    if lead is not None:
        value = strip_publisher_suffix(clean(clean_text(lead)))
        if valid_exact_sapo(value):
            return value

    for node in soup.find_all(True):
        attrs = " ".join([str(node.get("id", "")), " ".join(node.get("class", []))]).casefold()
        if not any(word in attrs for word in ("sapo", "standfirst", "lead", "summary", "description")):
            continue
        if any(word in attrs for word in ("related", "recommend", "advert", "sidebar", "footer")):
            continue
        value = strip_publisher_suffix(clean(clean_text(node)))
        if valid_fallback_sapo(value):
            return value

    for selector in ('meta[name="description"]', 'meta[property="og:description"]'):
        for node in soup.select(selector):
            value = strip_publisher_suffix(clean(node.get("content")))
            if valid_fallback_sapo(value):
                return value

    if listing_description:
        value = clean(BeautifulSoup(listing_description, "html.parser").get_text(" ", strip=True))
        if valid_fallback_sapo(value):
            return value

    # As a last resort, use the first article paragraph if it looks like a summary.
    first_paragraph = clean(body.split("\n\n", 1)[0])
    if valid_fallback_sapo(first_paragraph):
        return first_paragraph
    raise ValueError(f"No article-specific sapo found (expected {lead_selector})")


def extract_article(
    html: str,
    url: str,
    listing_title: str,
    listing_description: str = "",
) -> tuple[str, str, str, str]:
    """Return title, body, sapo, and publication time from the same downloaded page."""
    host = site_for(url)
    soup = BeautifulSoup(html, "html.parser")
    published_at = extract_published_at(soup, host)
    heading = soup.select_one("h1")
    title = clean(heading.get_text(" ", strip=True)) if heading else ""
    if not title:
        title_meta = soup.select_one('meta[property="og:title"], meta[name="twitter:title"]')
        title = clean(title_meta.get("content")) if title_meta else clean(listing_title)
    if not title:
        raise ValueError("Article page has no title")

    body = extract_body(html, url)
    sapo = extract_sapo(soup, host, body, listing_description)
    return title, body, sapo, published_at
