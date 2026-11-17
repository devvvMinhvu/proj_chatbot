"""Publisher, category, and HTML extraction configuration for the news crawler."""

from __future__ import annotations

from datetime import timedelta, timezone

TARGET_PER_CELL = 1000
SOURCES = ("vnexpress", "tuoitre", "tienphong")
CATEGORIES = ("news", "economy", "tech", "entertainment", "sports")
PREFIX = {"vnexpress": "vne", "tuoitre": "tt", "tienphong": "tp"}
HOST = {"vnexpress": "vnexpress.net", "tuoitre": "tuoitre.vn", "tienphong": "tienphong.vn"}
TIMEZONE = timezone(timedelta(hours=7))
PAGE_PAUSE = 0.15
ARTICLE_PAUSE = 0.35
MAX_ARCHIVE_PAGES = 500
MAX_LOOKBACK_DAYS = 3650

VNE_CATEGORY_IDS = {
    "news": 1001005,
    "economy": 1003159,
    "tech": 1006219,
    "entertainment": 1002691,
    "sports": 1002565,
}
TTO_CATEGORY_SLUGS = {
    "news": "thoi-su",
    "economy": "kinh-doanh",
    "tech": "nhip-song-so",
    "entertainment": "giai-tri",
    "sports": "the-thao",
}
TTO_ZONE_IDS = {
    "news": 3,
    "economy": 11,
    "tech": 16,
    "entertainment": 10,
    "sports": 1209,
}

TP_CATEGORY_SLUGS = {
    "news": "xa-hoi-tin-tuc",
    "economy": "kinh-te",
    "tech": "khoa-hoc",
    "entertainment": "giai-tri",
    "sports": "the-thao",
}
TP_ZONE_IDS = {
    "news": 104,
    "economy": 3,
    "tech": 45,
    "entertainment": 36,
    "sports": 11,
}

SITES = {
    "vnexpress.net": ("article.fck_detail", "p.description"),
    "tuoitre.vn": (".detail-content", ".detail-sapo"),
    "tienphong.vn": (".article__body", ".article__sapo"),
}
REMOVE = {
    "vnexpress.net": (
        '.list_link, .list-news, [data-component-type="tin_xemthem"], '
        ".box-tinlienquan, .box-related, .widget_soccer, .topbar-sticky, "
        ".social-com, .box_embed_video, .author_mail"
    ),
    "tuoitre.vn": '[type="RelatedNewsBox"], [type="RelatedOneNews"], .readmore-body-box, .kbwscwl-relatedbox',
    "tienphong.vn": '.article-relate, [data-source="related-news"], .sda_middle, .rennab, .article__author, .article__source',
}
NOISE = (
    'script, style, noscript, iframe, video, audio, svg, form, button, nav, aside, '
    '[hidden], [aria-hidden="true"], .ads, .advertisement, ins.adsbygoogle'
)
BLOCKS = {
    "p",
    "div",
    "section",
    "article",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ul",
    "ol",
    "blockquote",
    "figure",
    "figcaption",
    "tr",
}
