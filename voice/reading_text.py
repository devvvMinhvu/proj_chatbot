"""Create a separate, speech-friendly script from the assistant's display text.

The original answer is never modified. This module only prepares the text sent
into TTS: removes list numbering and URLs, speaks publication sources and
stored publication timestamps naturally, and preserves meaningful numbers
inside article titles and facts.
"""

from __future__ import annotations

import re
from typing import Iterable


_SOURCE_NAMES = {
    "vnexpress": "VnExpress",
    "tienphong": "Tiền Phong",
    "tuoitre": "Tuổi Trẻ",
    "nhandan": "Nhân Dân",
}

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)]+\)")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•])\s+")
_INLINE_LIST_NUMBER_RE = re.compile(
    # Handles flattened output such as "... 4. Title ... 5. Next title".
    # Requires a preceding sentence boundary or multiple whitespace, to avoid
    # stripping numeric facts like 60.000 or dates such as 27/10/2026.
    r"(?<=[.!?])\s+\d{1,2}[.)]\s+(?=[A-ZÀ-Ỹ])"
)
_SOURCE_SUFFIX_RE = re.compile(
    r"\s+[—–]\s+(vnexpress|tienphong|tuoitre|nhandan)\b",
    re.IGNORECASE,
)
_PUBLISHED_AT_RE = re.compile(
    r"Ngày đăng theo dữ liệu lưu\s*:\s*(\d{1,2})/(\d{1,2})/(\d{4})"
    r"(?:\s+(\d{1,2}):(\d{2}))?",
    re.IGNORECASE,
)


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
    if not line:
        return ""

    # Remove numbering only at a true line/list boundary.
    line = _LIST_PREFIX_RE.sub("", line)

    # Links are displayed in the UI but are not useful to read aloud.
    line = _MARKDOWN_LINK_RE.sub(r"\1", line)
    line = _URL_RE.sub("", line)

    # Convert the source slug after an em dash into a speakable source label.
    def source_label(match: re.Match[str]) -> str:
        slug = match.group(1).lower()
        return f". Báo {_SOURCE_NAMES.get(slug, slug)}"

    line = _SOURCE_SUFFIX_RE.sub(source_label, line)
    line = _PUBLISHED_AT_RE.sub(_format_publication_date, line)

    # Remove Markdown decoration while preserving punctuation and numbers.
    line = re.sub(r"[`*_#>|]", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def build_reading_text(display_text: str) -> str:
    """Build a distinct spoken version without changing the visible answer.

    - Removes list markers such as ``1.`` and ``2)`` at line starts.
    - Removes URLs from narration (the links remain visible in the chat).
    - Converts known source slugs and stored date/time fields to natural speech.
    - Keeps factual numbers in titles and article content.
    """
    if not isinstance(display_text, str) or not display_text.strip():
        return ""

    text = display_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

    # If a numbered list has been flattened to one line, remove only list
    # counters that follow sentence-ending punctuation. Then process lines.
    text = _INLINE_LIST_NUMBER_RE.sub(" ", text)

    spoken_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = _clean_line(raw_line)
        if not line:
            continue

        # Turn common result headings into a natural introduction.
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

    # Preserve paragraph order and add pauses between separate display lines.
    sentences: list[str] = []
    for line in spoken_lines:
        if not line:
            continue
        if line[-1] not in ".!?;:…":
            line += "."
        sentences.append(line)

    result = " ".join(sentences)
    result = re.sub(r"\s+([,.!?;:])", r"\1", result)
    result = re.sub(r"([.!?]){2,}", r"\1", result)
    return result.strip()
