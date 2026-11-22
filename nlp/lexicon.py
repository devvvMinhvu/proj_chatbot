"""Closed-class Vietnamese words the turn parser needs.

Everything that can be learned from data is learned elsewhere (keyword
statistics, publisher and category names from the database). What remains here
is grammar that a parser without a trained model cannot infer: number words,
the ordinal marker, a few direction words, demonstratives and question words.
Words are matched accent-aware (see ``processing.text_matching``), so "sau"
(after) and "sáu" (six) stay distinct when typed with accents.
"""

from __future__ import annotations

from processing.text_matching import accented_lower, fold_text, needs_exact_accents

# --- Numbers --------------------------------------------------------------------

DIGIT_WORDS = {
    "không": 0, "linh": 0, "lẻ": 0,
    "một": 1, "mốt": 1, "nhất": 1,
    "hai": 2, "nhì": 2,
    "ba": 3,
    "bốn": 4, "tư": 4,
    "năm": 5, "lăm": 5,
    "sáu": 6,
    "bảy": 7, "bẩy": 7,
    "tám": 8,
    "chín": 9,
}
TEN_WORDS = {"mười": 10}
TENS_MARKERS = {"mươi"}
HUNDRED_WORDS = {"trăm": 100}

#: Words that turn the following number into a position ("thứ hai", "số 3").
ORDINAL_MARKERS = ("thứ", "số")

# --- Positions relative to the open article ------------------------------------

RELATIVE_WORDS = {
    "next": ("kế tiếp", "tiếp theo", "kế", "sau"),
    "previous": ("trước", "trước đó"),
    "first": ("đầu tiên", "đầu"),
    "last": ("cuối", "cuối cùng"),
    "current": ("này", "đó", "ấy", "kia", "vừa rồi", "vừa xong", "trên"),
    #: "bài còn lại": of the articles just compared, the one not just named.
    "remaining": ("còn lại",),
}
PLURAL_MARKERS = ("những", "các")
#: "vừa" + verb after a noun points back like "vừa rồi": "bài vừa đọc", "hai bài vừa nêu".
RECENT_MARKERS = ("vừa",)
#: After a list item, asks for items not shown yet: "ba bài khác".
OTHER_WORDS = ("khác",)

# --- Negation, exclusion and scope ---------------------------------------------

#: Negate what follows in the same clause ("không phải bài 2", "đừng mở bài 1").
NEGATORS = ("không phải", "chứ không", "đừng", "chớ")
#: Negators that are also question particles ("… không?"); they only negate in statements.
STATEMENT_NEGATORS = ("không", "chẳng", "chưa")
#: Leave out what follows ("trừ VnExpress", "thay cho bài 1").
EXCLUSION_WORDS = ("trừ", "ngoại trừ", "ngoài", "thay cho", "thay vì")
#: Start a new clause, so an earlier negation no longer applies ("không phải bài 1 mà là bài 2").
CONTRAST_WORDS = ("mà", "nhưng", "chứ")
#: "tất cả / mọi" + a publisher noun lifts the publisher filter ("tìm trên tất cả các báo").
ALL_WORDS = ("tất cả", "mọi")
PUBLISHER_NOUNS = ("báo", "tờ báo", "nguồn", "trang")

# --- Dates ----------------------------------------------------------------------

#: A date match with digits must contain a separator ("21/10") or one of these units;
#: "9 trong" is not a date.
DATE_UNITS = ("ngày", "tháng", "năm", "tuần", "hôm", "giờ")

# --- Questions and summary length ----------------------------------------------

QUESTION_WORDS = (
    "ai", "gì", "nào", "đâu", "sao", "mấy", "bao nhiêu", "bao giờ", "khi nào",
    "thế nào", "như thế nào", "ra sao", "vì sao", "tại sao", "làm sao", "có phải",
)
#: Question particles that only ask when they end the sentence ("… không?").
FINAL_QUESTION_PARTICLES = ("không", "chưa", "hả", "à")
#: Ask whether a statement is true; the rest of the message is a claim to check
#: ("Đúng hay sai: …", "Có phải … không?", "… đúng không?").
VERIFICATION_PHRASES = (
    "đúng hay sai", "đúng hay không", "đúng không", "phải không", "có phải", "có đúng",
    "đúng là", "có thật", "phải chăng", "đúng chứ",
)
#: Nouns that frame a question instead of naming its subject ("con số dự kiến hay thực tế?").
QUESTION_NOUNS = (
    "con số", "số liệu", "thông tin", "chi tiết", "nội dung", "điều", "việc",
    "tỷ lệ", "tỉ lệ", "phần trăm", "số lượng",
)
#: Ask for a quantity; the noun after them is what is counted ("bao nhiêu đại biểu").
QUANTITY_WORDS = ("bao nhiêu", "mấy")
#: Ask for a share ("bao nhiêu phần trăm nhân lực", "tỷ lệ").
PERCENT_WORDS = ("phần trăm", "tỷ lệ", "tỉ lệ", "%")
#: Ask for a person, a place, a time.
PERSON_QUESTION_WORDS = ("ai",)
PLACE_QUESTION_WORDS = ("ở đâu", "tại đâu", "đâu", "nơi nào", "tỉnh nào", "nước nào", "thành phố nào")
TIME_QUESTION_WORDS = (
    "khi nào", "bao giờ", "lúc nào", "ngày nào", "hôm nào", "tháng nào", "năm nào", "mấy giờ",
)

# --- What a statement commits to -------------------------------------------------

#: Words that make a statement a plan, an offer or a prediction rather than a fact.
#: An answer has to keep them: "dự kiến quy tụ 500 đại biểu" is not "500 đại biểu
#: tham dự", "đề xuất với 60.000 nhân viên" is not "60.000 nhân viên nghỉ việc".
MODALITY_MARKERS = {
    "expected": ("dự kiến", "dự tính", "kế hoạch", "mục tiêu", "ước tính", "sẽ"),
    "proposed": ("đề xuất", "đề nghị", "kiến nghị"),
    "forecast": ("dự đoán", "dự báo", "nhận định", "kỳ vọng", "có thể", "đề cử", "ứng viên"),
}
#: Words that ask for, or state, what came of a plan, offer or prediction. A question
#: about a number that uses a completed-action word ("đã", see ASPECT_MARKERS) or a
#: precision word asks for the outcome too, whatever the article planned or offered.
OUTCOME_MARKERS = {
    "expected": ("thực tế", "thực sự"),
    "proposed": ("chấp nhận", "đồng ý", "nhận lời", "được duyệt", "thông qua", "đăng ký"),
    "forecast": ("kết quả chính thức", "chung cuộc", "chính thức"),
}
#: Ask for an exact figure; a planned or approximate one does not answer them.
PRECISION_WORDS = ("chính xác", "cụ thể")
#: The reply when an article states only the plan, offer or prediction that was asked about.
MODALITY_REPLIES = {
    "expected": "Bài viết chỉ nêu con số dự kiến; bài không nêu con số thực tế.",
    "proposed": (
        "Bài viết chỉ ghi nhận lời đề xuất; bài không nêu bao nhiêu người hay bao nhiêu "
        "trường hợp đã chấp nhận."
    ),
    "forecast": "Bài viết chỉ nêu dự đoán; bài không nêu kết quả chính thức.",
}
#: Claims that an article leaves something out; only the passages can establish them.
ABSENCE_PHRASES = (
    "không nêu", "không đề cập", "không nhắc", "không có thông tin", "không cung cấp",
    "chưa nêu", "chưa có thông tin", "không được nêu",
)
#: Ask for an amount of money; numbers in these units answer it.
MONEY_WORDS = ("số tiền", "kinh phí", "bao nhiêu tiền", "tổng tài trợ", "giá trị", "chi phí")
MONEY_UNITS = ("tỷ", "triệu", "nghìn", "ngàn", "đồng", "usd", "đô", "euro")
#: Nouns that ask for a figure without "bao nhiêu" ("trích câu nêu số tiền").
#: ("số" + a noun, "số khách", is read from the tagger; see rag.evidence.question_need.)
QUANTITY_NOUNS = ("số tiền", "con số", "số liệu", "kinh phí", "số lượng")
#: Ask for the article's own sentence ("Hãy trích câu nêu số tiền").
QUOTE_WORDS = ("trích", "trích dẫn", "dẫn lại", "nguyên văn", "câu nào nêu", "câu nêu")
#: Words that make a request polite or imperative ("Hãy ghi …"); never a name or a subject.
IMPERATIVE_WORDS = ("hãy", "xin", "vui lòng", "làm ơn")
#: Words of a request that say what to do with an article, not what it is about.
REQUEST_WORDS = ("trích", "trích dẫn", "dẫn", "dẫn lại", "nguyên văn", "câu", "nêu", "bài", "đoạn", "hãy")
#: Words before a number that make it approximate ("gần 3 tỷ", "hơn 25%", "khoảng 60.000").
APPROXIMATE_MARKERS = ("gần", "khoảng", "hơn", "trên", "dưới", "ít nhất", "tối đa", "chừng", "xấp xỉ", "gần như")
#: Whether something is done, under way or still to come; an answer keeps it
#: ("đã hoàn tất bàn giao" is not "đang xem xét bàn giao").
ASPECT_MARKERS = {
    "done": ("đã", "vừa", "xong", "rồi"),
    "ongoing": ("đang",),
    "future": ("sẽ", "sắp", "chưa"),
}
#: Verbs that carry no claim of their own ("là", "có").
LIGHT_VERBS = ("là", "có", "được", "bị", "cho")
#: Which way an approximation points; an answer has to keep it ("hơn 25%" is not "khoảng 25%").
APPROXIMATE_DIRECTIONS = {
    "about": ("khoảng", "chừng", "xấp xỉ"),
    "near": ("gần", "gần như"),
    "over": ("hơn", "trên", "ít nhất"),
    "under": ("dưới", "tối đa"),
}
#: A sentence that starts with these continues the previous one ("Con số này tương đương
#: hơn 25% nhân lực của hãng"); "còn lại" anywhere does too ("Công trình còn lại do BW tài trợ").
CONTINUATION_STARTS = (
    "con số này", "con số đó", "việc này", "điều này", "đây", "đó", "trong đó", "ngoài ra",
    "anh", "ông", "bà", "chị", "họ", "hãng", "công ty", "đội bóng", "sự kiện",
)
CONTINUATION_WORDS = ("còn lại",)
#: A sentence that starts with these breaks the previous total down ("Trong đó, 10 công trình …").
BREAKDOWN_STARTS = ("trong đó", "cụ thể", "bao gồm", "gồm")
#: Words that stand for a name mentioned in the previous sentence.
COREFERENCE_WORDS = ("hãng", "công ty", "họ", "anh", "ông", "bà", "chị", "đội", "này", "đó")
#: "cả hai bài đều …": the claim has to hold in every article it cites.
EVERY_WORDS = ("cả hai", "cả ba", "đều")
#: Join names into one subject ("Opella Việt Nam và BW Industrial bàn giao 11 công trình").
JOINING_WORDS = ("và", "cùng", "với", ",")
#: Units that make a number a summary length ("một câu").
SENTENCE_UNITS = ("câu",)


def has_accents(text: str) -> bool:
    """True when any letter of ``text`` carries a Vietnamese accent or diacritic."""
    lowered = accented_lower(text)
    return fold_text(lowered) != lowered


def same_word(typed: str, reference: str, exact: bool | None = None) -> bool:
    """Accent-aware equality.

    ``exact`` should say whether the user's message was typed with accents; then
    words must match exactly, so "sau" (after) never equals "sáu" (six). In an
    unaccented message words match in accent-free form. When ``exact`` is None it
    is inferred from the word alone.
    """
    typed_lower = accented_lower(typed).replace("_", " ").strip()
    reference_lower = accented_lower(reference)
    if exact is None:
        exact = has_accents(typed_lower) or needs_exact_accents(typed_lower)
    if exact:
        return typed_lower == reference_lower
    return fold_text(typed_lower) == fold_text(reference_lower)


def is_one_of(typed: str, words, exact: bool | None = None) -> bool:
    return any(same_word(typed, word, exact) for word in words)


def number_word_value(word: str, exact: bool | None = None) -> int | None:
    for table in (DIGIT_WORDS, TEN_WORDS, HUNDRED_WORDS):
        for key, value in table.items():
            if same_word(word, key, exact):
                return value
    return None


def parse_number(words: list[str], exact: bool | None = None) -> int | None:
    """Value of a Vietnamese number written in digits or words.

    Handles "12", "mười hai", "hai mươi mốt", "một trăm linh năm" and ordinal
    forms after "thứ" ("nhất", "nhì", "tư"). Returns None when the words are
    not a number.
    """
    if not words:
        return None
    if len(words) == 1 and words[0].isdigit():
        return int(words[0])
    total = 0
    current = 0
    seen = False
    for word in words:
        if word.isdigit():
            current = current * 10 + int(word) if seen and current < 10 else int(word)
            seen = True
            continue
        if is_one_of(word, TENS_MARKERS, exact):
            current = (current or 1) * 10
            seen = True
            continue
        value = number_word_value(word, exact)
        if value is None:
            return None
        seen = True
        if value == 100:
            total += (current or 1) * 100
            current = 0
        elif value == 10:
            current += 10
        else:
            current += value
    return total + current if seen else None
