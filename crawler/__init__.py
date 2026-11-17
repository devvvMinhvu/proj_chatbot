"""News crawler package for VnExpress, Tuổi Trẻ, and Tiền Phong."""

from __future__ import annotations

from typing import Any

__all__ = [
    "extract_article",
    "extract_body",
    "fill_missing_bodies",
    "parse_published_at",
    "scrape_dataset",
]

_EXTRACTOR_EXPORTS = {"extract_article", "extract_body", "parse_published_at"}
_PIPELINE_EXPORTS = {"fill_missing_bodies", "scrape_dataset"}


def __getattr__(name: str) -> Any:
    if name in _EXTRACTOR_EXPORTS:
        from crawler import extractor

        return getattr(extractor, name)
    if name in _PIPELINE_EXPORTS:
        from crawler import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
