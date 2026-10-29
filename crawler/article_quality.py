import re
from urllib.parse import urlsplit, urlunsplit


# Các định dạng không nhận vào bộ dữ liệu bài báo dạng văn bản
MEDIA_TERMS = (
    "bộ ảnh",
    "chùm ảnh",
    "ảnh",
    "video",
    "clip",
    "infographic",
    "đồ họa",
    "đồ hoạ",
    "podcast",
    "audio",
    "photostory",
    "photo story",
    "livestream",
)

MEDIA_TITLE_RE = re.compile(
    r"^\s*(?:"
    r"\[\s*)?(?:"
    r"bộ\s+ảnh|chùm\s+ảnh|ảnh|video|clip|infographic|"
    r"đồ\s*họa|đồ\s*hoạ|podcast|audio|photostory|"
    r"photo\s+story|livestream|trực\s+tiếp"
    r")\s*(?:\])?\s*(?=[:：\-–—|]|\s|$)",
    re.IGNORECASE,
)

MEDIA_URL_RE = re.compile(
    r"^(?:anh|bo-anh|chum-anh|video|clip|infographic|"
    r"podcast|audio|photostory|photo-story|livestream)-",
    re.IGNORECASE,
)


# Các trường bắt buộc phải có
REQUIRED_FIELDS = (
    "title",
    "sapo",
    "content",
    "category",
    "research_topic",
    "author",
    "published_at",
    "source",
    "url",
)


def get_value(record, field):
    """Đọc giá trị từ dictionary hoặc đối tượng Article."""
    if isinstance(record, dict):
        return record.get(field)

    return getattr(record, field, None)


def normalize_article_url(url):
    """Chuẩn hóa URL để phát hiện URL trùng."""
    if not url:
        return ""

    try:
        parsed = urlsplit(str(url).strip())

        if parsed.scheme.lower() not in ("http", "https"):
            return ""

        if not parsed.hostname:
            return ""

        host = parsed.hostname.lower()

        if host.startswith("www."):
            host = host[4:]

        port = parsed.port
        if port and port not in (80, 443):
            host = f"{host}:{port}"

        path = parsed.path.rstrip("/") or "/"
        
        # Bỏ query và fragment.
        return urlunsplit((
            "https",
            host,
            path,
            "",
            "",
        ))

    except (ValueError, TypeError):
        return ""


def missing_required_fields(record, include_source=True):
    """Trả về danh sách trường bị thiếu hoặc chỉ chứa khoảng trắng."""
    fields = REQUIRED_FIELDS

    if not include_source:
        fields = tuple(
            field for field in fields
            if field != "source"
        )

    missing = []

    for field in fields:
        value = get_value(record, field)

        if value is None or not str(value).strip():
            missing.append(field)

    return missing


def is_non_text_article(title, url="", category=""):
    """Nhận diện các bài có định dạng ảnh/video/infographic."""
    title = (title or "").strip()
    category = (category or "").strip()

    # Không nhầm "Ảnh hưởng..." với bài ảnh.
    if re.match(
        r"^\s*(?:\[\s*)?ảnh\s+hưởng\b",
        title,
        re.IGNORECASE,
    ):
        return False

    if MEDIA_TITLE_RE.match(title):
        return True

    # Kiểm tra chuyên mục có tên định dạng media hay không.
    category_lower = category.casefold()

    if any(
        category_lower == term.casefold()
        for term in MEDIA_TERMS
    ):
        return True

    normalized_url = normalize_article_url(url)

    if normalized_url:
        path = urlsplit(normalized_url).path
        slug = path.rsplit("/", 1)[-1]

        if MEDIA_URL_RE.match(slug):
            return True

    return False


def has_paragraph_content(content):
    """
    Kiểm tra nội dung có đủ dấu hiệu là văn bản bài báo.

    Ngưỡng ban đầu:
    - Ít nhất 100 ký tự không tính khoảng trắng.
    - Ít nhất 12 từ.
    """
    if not content or not isinstance(content, str):
        return False

    text = content.strip()

    if len(re.sub(r"\s+", "", text)) < 100:
        return False

    words = re.findall(
        r"[A-Za-zÀ-ỹĐđ]+",
        text,
    )

    return len(words) >= 12