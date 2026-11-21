"""Create a separate, speech-friendly script from the assistant's display text.

The original answer is never modified. This module prepares concise text for
Edge TTS: strips URLs, UI tips, and redundant timestamps in multi-article lists,
and caps list narration to the top headlines so voice responses stay fast.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_SOURCE_NAMES = {
    "vnexpress": "VnExpress",
    "tienphong": "Tiền Phong",
    "tuoitre": "Tuổi Trẻ",
}

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)]+\)")
_NUMBERED_ITEM_RE = re.compile(r"^\s*\d{1,2}[.)]\s+")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•])\s+")
_INLINE_LIST_NUMBER_RE = re.compile(
    # Handles flattened output such as "... 4. Title ... 5. Next title".
    # Requires sentence-ending punctuation before the list counter to keep numeric facts.
    r"(?<=[.!?])\s+\d{1,2}[.)]\s+(?=[A-ZÀ-Ỹ])"
)
_SOURCE_SUFFIX_RE = re.compile(
    r"\s+[—–-]\s+(vnexpress|tienphong|tuoitre)\b",
    re.IGNORECASE,
)
_PUBLISHED_AT_RE = re.compile(
    r"Ngày đăng(?:\s+theo dữ liệu lưu)?\s*:\s*(\d{1,2})/(\d{1,2})/(\d{4})"
    r"(?:\s+(\d{1,2}):(\d{2}))?",
    re.IGNORECASE,
)
_UI_TIP_RE = re.compile(r"^\s*💡")


def _format_publication_date(match: re.Match[str]) -> str:
    day, month, year, hour, minute = match.groups()
    result = f"Ngày đăng theo dữ liệu lưu là ngày {int(day)} tháng {int(month)} năm {year}"
    if hour is not None:
        h = int(hour)
        m = int(minute or 0)
        if m:
            result += f", lúc {h} giờ {m} phút"
        else:
            result += f", lúc {h} giờ"
    return result


def _clean_line(line: str) -> str:
    line = line.strip()
    if not line or _UI_TIP_RE.match(line):
        return ""

    # Remove numbering only at a true line/list boundary.
    line = _LIST_PREFIX_RE.sub("", line)

    # Links are displayed in the UI but are not useful to read aloud.
    line = _MARKDOWN_LINK_RE.sub(r"\1", line)
    line = _URL_RE.sub("", line)

    def source_label(match: re.Match[str]) -> str:
        slug = match.group(1).lower()
        return f". Báo {_SOURCE_NAMES.get(slug, slug)}"

    line = _SOURCE_SUFFIX_RE.sub(source_label, line)
    line = _PUBLISHED_AT_RE.sub(_format_publication_date, line)

    # Remove Markdown decoration while preserving punctuation and numbers.
    line = re.sub(r"[`*_#>|]", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _finalize_sentences(spoken_lines: list[str]) -> str:
    sentences: list[str] = []
    for line in spoken_lines:
        line = line.strip()
        if not line:
            continue
        if line[-1] not in ".!?;:…":
            line += "."
        sentences.append(line)

    result = " ".join(sentences)
    result = re.sub(r"\s+([,.!?;:])", r"\1", result)
    result = re.sub(r"([.!?]){2,}", r"\1", result)
    return result.strip()


def _format_top_sources_intro(sources: list[dict[str, Any]], max_list_items: int) -> list[str]:
    ordinals = ("Tin thứ nhất", "Tin thứ hai", "Tin thứ ba", "Tin thứ tư", "Tin thứ năm")
    lines: list[str] = []
    for idx, src in enumerate(sources[:max_list_items]):
        title = (src.get("title") or "").strip()
        if not title:
            continue
        prefix = ordinals[idx] if idx < len(ordinals) else f"Tin thứ {idx + 1}"
        source_slug = (src.get("source") or "").strip().lower()
        source_name = _SOURCE_NAMES.get(source_slug, source_slug)
        if source_name:
            lines.append(f"{prefix}: {title}, theo báo {source_name}")
        else:
            lines.append(f"{prefix}: {title}")
    if len(sources) > max_list_items:
        lines.append(
            "Các bài báo khác đang được hiển thị trên màn hình. "
            "Bạn có thể chọn số thứ tự bài để nghe tóm tắt hoặc chi tiết"
        )
    return lines


def build_reading_text(
    display_text: str,
    *,
    sources: Optional[list[dict[str, Any]]] = None,
    max_list_items: int = 3,
    max_spoken_chars: int = 1200,
) -> str:
    """Build a concise, natural spoken script from assistant display text and optional sources."""
    if not isinstance(display_text, str) or not display_text.strip():
        return ""

    text = display_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = _INLINE_LIST_NUMBER_RE.sub(" ", text)

    raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
    numbered_count = sum(1 for line in raw_lines if _NUMBERED_ITEM_RE.match(line))
    is_concise_list = bool(sources and len(sources) > 1 and numbered_count == 0)

    if is_concise_list:
        intro_lines: list[str] = []
        for raw_line in raw_lines:
            if _UI_TIP_RE.match(raw_line) or raw_line.lower().startswith("**lưu ý:**"):
                continue
            cleaned = _clean_line(raw_line)
            if not cleaned:
                continue
            cleaned = re.sub(
                r"^(\d{1,2})\s+bài báo mới nhất theo ngày đăng đã lưu\.?$",
                lambda m: f"Sau đây là {m.group(1)} bài báo mới nhất theo ngày đăng đã lưu.",
                cleaned,
                flags=re.IGNORECASE,
            )
            intro_lines.append(cleaned)
        spoken = intro_lines[:1] + _format_top_sources_intro(sources or [], max_list_items)
        return _finalize_sentences(spoken)

    is_long_numbered_list = numbered_count > max_list_items
    spoken_lines: list[str] = []
    seen_list_items = 0
    truncated_list = False

    for raw_line in raw_lines:
        if _UI_TIP_RE.match(raw_line):
            continue

        if is_long_numbered_list:
            if _NUMBERED_ITEM_RE.match(raw_line):
                seen_list_items += 1
                if seen_list_items > max_list_items:
                    truncated_list = True
                    continue
            elif seen_list_items > max_list_items:
                continue
            elif _PUBLISHED_AT_RE.match(raw_line):
                # Skip repetitive timestamps in spoken multi-article lists.
                continue

        line = _clean_line(raw_line)
        if not line:
            continue

        line = re.sub(
            r"^(\d{1,2})\s+bài báo mới nhất theo ngày đăng đã lưu\.?$",
            lambda m: f"Sau đây là {m.group(1)} bài báo mới nhất theo ngày đăng đã lưu.",
            line,
            flags=re.IGNORECASE,
        )
        if line.lower() in {"nguồn bài báo", "danh sách nguồn bài báo"}:
            continue
        line = re.sub(r"\[Nguồn\s+\d+\]\s*", "", line, flags=re.IGNORECASE)
        spoken_lines.append(line)

    if truncated_list:
        spoken_lines.append(
            "Các bài báo khác đang được hiển thị trên màn hình. "
            "Bạn có thể chọn số thứ tự bài để nghe tóm tắt hoặc chi tiết"
        )

    result = _finalize_sentences(spoken_lines)
    if max_spoken_chars > 0 and len(result) > max_spoken_chars and is_long_numbered_list:
        cut = result.rfind(".", 0, max_spoken_chars)
        if cut > 80:
            result = result[: cut + 1].strip()
    return result
