"""Replays the conversations of the follow-up conversation audit (22 Nov 2026).

Each test is one audit conversation. The database, search, summarizer and LLM
are stubbed, so the tests check what the assistant decides: which article a
message points to, which action runs, and which filters, count and length it
uses. Factual answer quality needs the live models and is checked separately
(``python -m evaluation.conversation_audit``).

The parser needs the corpus keyword statistics (``python -m nlp.keyword_stats``);
the tests are skipped when they are not available.
"""

from __future__ import annotations

from datetime import date
import unittest
from unittest.mock import patch

from nlp import intent_handler as handler
from nlp.conversation_state import ConversationState
from nlp.keyword_stats import KEYWORD_STATS_PATH

ARCHIVE = (date(2022, 11, 5), date(2026, 10, 27))
SUMMARY = "Đây là câu tóm tắt thứ nhất. Đây là câu tóm tắt thứ hai. Đây là câu tóm tắt thứ ba."
TODAY = date(2026, 11, 22)


def article(index: int, title: str, source: str = "vnexpress", category: str = "sports", body: str = "") -> dict:
    return {
        "article_id": f"a{index}",
        "title": title,
        "sapo": "",
        "source": source,
        "category": category,
        "url": f"https://example.vn/{index}.html",
        "published_at": "2026-10-27 10:00:00",
        # The stub summary is taken from the article, so its sentences are in the text.
        "body": f"{body or title} {SUMMARY}",
    }


LATEST = [
    article(1, "Tổng biên tập France Football: 'Mbappe đã có lúc dẫn đầu cuộc đua'"),
    article(2, "Harry Kane - từ người không danh hiệu đến Quả Bóng Vàng",
            body="Harry Kane đoạt Quả Bóng Vàng 2026 ở tuổi 33. Kane gia nhập Tottenham ở tuổi 11."),
    article(3, "Nhận định Fleetwood Town vs Arsenal, 03h00 ngày 28/10", source="tienphong"),
    article(4, "Trang Moon gây cười với vai cô bán bánh rán", category="entertainment"),
    article(5, "Taylor Swift, Bruno Mars có lép vế tại Grammy 2027?", source="tuoitre", category="entertainment"),
    article(6, "Xuất hiện đoạn video 'minh oan' cho Mbappe", source="tuoitre"),
]
SON_LA = [
    article(11, "Ngày hội vệ sinh học đường của học sinh Sơn La", category="news",
            body="Ngày hội vệ sinh học đường tại Sơn La, khánh thành 11 cụm công trình vệ sinh."),
    article(12, "Khánh thành cụm 11 công trình vệ sinh đạt chuẩn ở Sơn La", category="news",
            body="Quỹ Hy vọng bàn giao 11 công trình vệ sinh học đường tại 6 xã của Sơn La, tổng tài trợ gần 3 tỷ đồng."),
]
MBAPPE = [article(21, "Mbappe chạm mốc 100 bàn cho Real Madrid"), article(22, "Mbappe tăng tốc trong cuộc đua")]
# Shortened from the stored articles of the premise and role audit conversations.
DIGITAL_WEEK = article(
    31, "Những diễn giả nổi bật tại Tuần lễ Số 2026", category="tech",
    body=(
        "Nhiều chuyên gia sẽ tham dự Tuần lễ Số 2026. "
        "Tuần lễ Số Quốc tế Việt Nam 2026 (VIDW 2026) diễn ra từ ngày 28 đến 30/10 tại Quảng Ninh. "
        "Sự kiện do Bộ Khoa học và Công nghệ chủ trì tổ chức, dự kiến quy tụ 500 đại biểu trong và ngoài nước."
    ),
)
LONG_AN = article(
    32, "CLB hạng nhất đầu tiên thay HLV", source="tuoitre",
    body=(
        "CLB Long An đã thay HLV Trịnh Văn Hậu sau thành tích không tốt của đội qua 4 vòng đấu đầu tiên. "
        "Sau 4 trận, CLB Long An chỉ giành có 1 điểm. "
        "Cụ thể, HLV Huỳnh Ngọc San đã được chọn và đã chính thức bắt tay vào công việc vào chiều 26-10. "
        "Đội bóng xứ chè còn có chân sút Victor Mansaray đang dẫn đầu danh sách Vua phá lưới với 8 bàn thắng."
    ),
)
HUE_RAIN = article(33, "Huế cảnh báo mưa lớn, tăng nguy cơ lũ quét", source="tienphong", category="news",
                   body="Huế cảnh báo mưa lớn trong tuần này, 700 hộ dân cần sơ tán.")
MICROSOFT = article(
    34, "Microsoft đề xuất 60.000 nhân viên tự nguyện nghỉ việc", category="tech",
    body=(
        "Ngày 26/10, Microsoft cho biết đang đề xuất với khoảng 60.000 nhân viên nghỉ việc tự nguyện kèm trợ cấp. "
        "Con số này tương đương hơn 25% nhân lực toàn cầu của hãng."
    ),
)
FUNDING = article(
    35, "Khánh thành cụm 11 công trình vệ sinh đạt chuẩn ở Sơn La", category="news",
    body=(
        "Sáng 22/10, Quỹ Hy vọng cùng Opella Việt Nam và BW Industrial bàn giao cụm 11 công trình vệ sinh "
        "học đường tại 6 xã, tổng tài trợ gần 3 tỷ đồng. "
        "Trong đó, 10 công trình do Opella Việt Nam đồng hành. "
        "Công trình còn lại do BW Industrial tài trợ."
    ),
)
BY_ID = {
    item["article_id"]: item
    for item in LATEST + SON_LA + MBAPPE + [DIGITAL_WEEK, LONG_AN, HUE_RAIN, MICROSOFT, FUNDING]
}


class Backend:
    """Stubs for everything below the handler, recording what was asked for."""

    def __init__(self, test: unittest.TestCase) -> None:
        self.test = test
        self.calls: list[tuple[str, tuple, dict]] = []
        self.search_results: list[dict] = MBAPPE
        self.retrieve_articles: list[dict] = []
        self.latest_articles: list[dict] = LATEST

    def _record(self, name, result):
        def fake(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return result(*args, **kwargs) if callable(result) else result
        return fake

    def start(self) -> None:
        def content(article_id=None, url=None):
            item = BY_ID.get(article_id)
            return dict(item) if item else None

        def latest(limit=10, **filters):
            items = [a for a in self.latest_articles if filters.get("source") in (None, a["source"])
                     and filters.get("category") in (None, a["category"])
                     and a["source"] not in (filters.get("exclude_sources") or ())]
            return items[:limit], len(items), len(items)

        def retrieve(question, **kwargs):
            return [{**a, "text": a["body"], "chunk_id": 0, "similarity": 0.9} for a in self.retrieve_articles]

        def group(chunks):
            return [{**c, "texts": [c["text"]]} for c in chunks]

        patches = [
            patch.object(handler, "get_article_content", side_effect=self._record("content", content)),
            patch.object(handler, "_get_latest_news", side_effect=self._record("latest", latest)),
            patch.object(handler, "_search_sources", side_effect=self._record("search", lambda *a, **k: list(self.search_results))),
            patch.object(handler, "_search_latest_topic_articles", side_effect=self._record(
                "latest_topic", lambda words, **k: (list(self.search_results), len(self.search_results), len(self.search_results)))),
            patch.object(handler, "_search_topic_articles", side_effect=self._record(
                "category", lambda cat, **k: ([a for a in LATEST if a["category"] == cat], 3000, True, []))),
            patch.object(handler, "retrieve", side_effect=self._record("retrieve", retrieve)),
            patch.object(handler, "group_chunks_by_article", side_effect=group),
            patch.object(handler, "stream_answer", side_effect=self._record(
                "llm", lambda q, articles, **k: iter([f"<answer from {len(articles)} articles>"]))),
            patch.object(handler, "generate_answer", side_effect=self._record("llm", lambda q, a, **k: "<answer>")),
            patch.object(handler, "_load_summarizer", return_value=self._record(
                "summarize", lambda body, article_id=None: SUMMARY)),
            patch.object(handler, "get_archive_date_range", return_value=ARCHIVE),
            patch("builtins.print"),
        ]
        for item in patches:
            item.start()
            self.test.addCleanup(item.stop)

    def called(self, name: str) -> list[tuple[tuple, dict]]:
        return [(args, kwargs) for call_name, args, kwargs in self.calls if call_name == name]


class ConversationTestCase(unittest.TestCase):
    """Helpers: a fresh conversation over the stubbed backend."""

    def setUp(self) -> None:
        self.backend = Backend(self)
        self.backend.start()
        self.state = ConversationState()

    def say(self, text: str, classifier: str = "semantic_search") -> dict:
        self.backend.calls.clear()
        with patch.object(handler, "predict_intent", return_value=classifier), patch(
            "nlp.turn_parser.date"
        ) as fake_date:
            fake_date.today.return_value = TODAY
            fake_date.side_effect = lambda *a, **k: date(*a, **k)
            outcome = handler.respond(text, self.state, stream_qa=True)
        if outcome.get("kind") == "qa_stream":
            outcome["answer"] = "".join(outcome.pop("stream"))
        return outcome

    def titles(self, outcome: dict) -> list[str]:
        return [s.get("title", "") for s in outcome.get("sources") or []]

    def open_latest(self) -> None:
        self.say("Tin tức mới nhất", classifier="latest_news")


@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class AuditConversations(ConversationTestCase):
    # --- Ordinals and positions --------------------------------------------------

    def test_ordinal_words_after_any_noun(self):  # audit "Ordinal variants"
        self.open_latest()
        for text in ("Đọc toàn văn tin thứ hai", "Đọc toàn văn kết quả thứ hai", "doc toan van bai so 2",
                     "Mở tin thứ 2 giúp mình"):
            with self.subTest(text=text):
                outcome = self.say(text, classifier="question_answering")
                self.assertEqual(outcome["intent"], "article_detail")
                self.assertEqual(self.titles(outcome), [LATEST[1]["title"]])

    def test_next_and_previous_follow_the_open_article(self):  # "References relative to position"
        self.open_latest()
        self.say("Đọc toàn văn bài số 2", classifier="article_detail")
        self.assertEqual(self.titles(self.say("Đọc bài trước đó", "article_detail")), [LATEST[0]["title"]])
        self.assertEqual(self.titles(self.say("Đọc bài sau", "article_detail")), [LATEST[1]["title"]])
        self.assertEqual(self.titles(self.say("Đọc toàn văn bài kế tiếp", "article_detail")), [LATEST[2]["title"]])
        self.assertEqual(self.titles(self.say("Đọc bài cuối cùng", "article_detail")), [LATEST[-1]["title"]])

    def test_correction_uses_the_last_number(self):  # "Correction"
        self.open_latest()
        self.say("Đọc bài số 1", classifier="article_detail")
        outcome = self.say("Không phải bài 1, ý tôi là bài 2. Đọc toàn văn giúp tôi.", "article_detail")
        self.assertEqual(self.titles(outcome), [LATEST[1]["title"]])

    def test_comparison_uses_both_articles(self):
        self.open_latest()
        outcome = self.say("So sánh bài 1 và bài 2. Hai bài khác nhau ở điểm nào?", "question_answering")
        self.assertEqual(self.titles(outcome), [LATEST[0]["title"], LATEST[1]["title"]])
        self.assertIn("2 articles", outcome["answer"])

    def test_recent_selection_beats_older_pin(self):  # "Pinned versus selected"
        self.open_latest()
        self.state.select(LATEST[0], "pin")
        self.say("Đọc toàn văn bài số 2", classifier="article_detail")
        outcome = self.say("Đọc toàn văn bài vừa rồi", classifier="article_detail")
        self.assertEqual(self.titles(outcome), [LATEST[1]["title"]])

    def test_number_in_a_name_or_question_is_not_an_item(self):
        self.open_latest()
        self.backend.retrieve_articles = [LATEST[1]]
        outcome = self.say("Ai giành Quả Bóng Vàng 2026?", classifier="question_answering")
        self.assertEqual(outcome["intent"], "question_answering")
        self.assertTrue(self.backend.called("retrieve"))
        outcome = self.say("Messi giành Quả Bóng Vàng lần thứ 4?", classifier="question_answering")
        self.assertNotEqual(self.titles(outcome), [LATEST[3]["title"]])

    def test_empty_search_does_not_silently_reuse_old_list(self):  # "Empty search context"
        self.open_latest()
        self.backend.search_results = []
        self.say("Tìm các bài báo nhắc đến Qzxvplm98471")
        outcome = self.say("Đọc toàn văn bài số 1", classifier="article_detail")
        self.assertIn("không có kết quả", outcome["answer"])
        self.assertEqual(self.titles(outcome), [LATEST[0]["title"]])

    def test_no_context_messages(self):  # "No context"
        self.assertIn("chưa có bài báo nào", self.say("Tóm tắt bài này", "summarization")["answer"].lower())
        self.assertIn("chưa có danh sách", self.say("Đọc bài thứ hai", "article_detail")["answer"].lower())

    # --- Actions on the open article ------------------------------------------------

    def test_questions_about_the_open_article_are_answered(self):  # "Selected project", "Unsupported specifics"
        self.open_latest()
        self.say("Đọc toàn văn bài số 2", classifier="article_detail")
        for text, classifier in (("Bài này đăng lúc mấy giờ?", "summarization"),
                                 ("Bài này có nêu tổng kinh phí bằng tiền không?", "summarization")):
            with self.subTest(text=text):
                outcome = self.say(text, classifier)
                self.assertEqual(outcome["intent"], "question_answering")
                self.assertEqual(self.titles(outcome), [LATEST[1]["title"]])

    def test_summary_length_is_respected(self):  # "Tightening a summary"
        self.open_latest()
        outcome = self.say("Tóm tắt bài số 2", classifier="summarization")
        self.assertIn("thứ ba.", outcome["answer"])
        outcome = self.say("Tóm tắt bài này bằng đúng một câu.", classifier="summarization")
        self.assertIn("thứ nhất.", outcome["answer"])
        self.assertNotIn("thứ hai.", outcome["answer"])

    def test_shorter_again_resummarizes_the_same_article(self):  # "Explicit source"
        self.backend.search_results = MBAPPE
        self.say("Tìm các bài báo nhắc đến Mbappe trên Tuổi Trẻ")
        self.say("Tóm tắt bài đầu tiên", classifier="summarization")
        outcome = self.say("Nói ngắn hơn, chỉ một câu thôi.", classifier="summarization")
        self.assertEqual(self.titles(outcome), [MBAPPE[0]["title"]])
        self.assertNotIn("thứ hai.", outcome["answer"])

    # --- Filters, counts and dates -----------------------------------------------------

    def test_publisher_refinement_reruns_the_previous_search(self):  # "Publisher and follow-up"
        self.say("Tìm các bài báo nhắc đến Mbappe")
        self.say("Chỉ lấy các bài của Tuổi Trẻ thôi", classifier="topic_search")
        (args, kwargs), = self.backend.called("search")
        self.assertEqual(kwargs.get("source"), "tuoitre")
        self.assertIn("Mbappe", args[0])
        self.say("Còn trên VnExpress thì sao?", classifier="question_answering")
        (args, kwargs), = self.backend.called("search")
        self.assertEqual(kwargs.get("source"), "vnexpress")

    def test_count_and_dates(self):  # "Freshness and count"
        self.say("Cho tôi 3 tin mới nhất", classifier="latest_news")
        (_, kwargs), = self.backend.called("latest")
        self.assertEqual(kwargs.get("limit"), 3)
        outcome = self.say("Có tin nào hôm nay, ngày 22/11/2026 không?", classifier="latest_topic_search")
        self.assertIn("27/10/2026", outcome["answer"])
        self.assertFalse(self.backend.called("latest_topic"))

    def test_category_with_subject_and_publisher(self):  # "Category and narrower topic"
        self.say("Liệt kê bài báo thuộc chuyên mục thể thao về Arsenal", classifier="topic_search")
        (args, kwargs), = self.backend.called("search")
        self.assertEqual(kwargs.get("category"), "sports")
        self.say("Cho tôi tin thể thao mới nhất của Tiền Phong", classifier="latest_topic_search")
        (_, kwargs), = self.backend.called("latest")
        self.assertEqual((kwargs.get("source"), kwargs.get("category")), ("tienphong", "sports"))

    def test_narrowing_the_current_list(self):
        self.say("Liệt kê bài báo thuộc chuyên mục thể thao", classifier="topic_search")
        outcome = self.say("Trong những bài này, chỉ lấy bài về Arsenal", classifier="question_answering")
        self.assertEqual(self.titles(outcome), [LATEST[2]["title"]])

    def test_count_publisher_and_subject_together(self):  # "Headline precision"
        self.say("Chỉ cho tôi 3 bài về Microsoft trên VnExpress")
        (args, kwargs), = self.backend.called("search")
        self.assertEqual(kwargs.get("source"), "vnexpress")
        self.assertIn("Microsoft", args[0])

    # --- Questions and follow-ups ---------------------------------------------------------

    def test_follow_up_question_keeps_the_subject(self):  # "Grounded school project", "Topic recovery"
        self.backend.retrieve_articles = SON_LA
        self.say("Dự án vệ sinh học đường tại Sơn La đã xây mới bao nhiêu công trình?", "question_answering")
        self.backend.retrieve_articles = []
        outcome = self.say("Dự án này ở tỉnh nào?", classifier="question_answering")
        self.assertEqual(outcome["intent"], "question_answering")
        self.assertIn("vệ sinh", outcome["answer"])  # the note says what was understood
        self.assertEqual(self.titles(outcome), [a["title"] for a in SON_LA])
        # "dự án này" points back explicitly: the earlier articles answer it, even though
        # they say "tài trợ" rather than "kinh phí" (a fresh search drifted to Tây Ninh).
        outcome = self.say("Tổng kinh phí của dự án này là bao nhiêu?", classifier="question_answering")
        self.assertEqual(self.titles(outcome), [a["title"] for a in SON_LA])
        self.assertFalse(self.backend.called("retrieve"))

    def test_follow_up_without_a_subject_of_its_own(self):
        self.backend.retrieve_articles = SON_LA
        self.say("Dự án vệ sinh học đường tại Sơn La đã xây mới bao nhiêu công trình?", "question_answering")
        self.backend.retrieve_articles = []
        outcome = self.say("Ở tỉnh nào vậy?", classifier="question_answering")
        self.assertEqual(self.titles(outcome), [a["title"] for a in SON_LA])
        self.assertFalse(self.backend.called("retrieve"))

    def test_new_subject_question_does_not_reuse_old_topic(self):  # "Topic change"
        self.backend.retrieve_articles = SON_LA
        self.say("Dự án vệ sinh học đường tại Sơn La đã xây mới bao nhiêu công trình?", "question_answering")
        self.backend.retrieve_articles = [LATEST[1]]
        outcome = self.say("Vì sao Microsoft đề xuất nhân viên tự nguyện nghỉ việc?", "question_answering")
        self.assertNotIn("Hiểu là câu hỏi tiếp theo", outcome["answer"])

    def test_question_about_unknown_name_is_a_question(self):  # "Miss and near match"
        outcome = self.say("Ai là người phát minh Qzxvplm98471?", classifier="introduction")
        self.assertEqual(outcome["intent"], "question_answering")
        outcome = self.say("Bạn làm được gì?", classifier="introduction")
        self.assertEqual(outcome["intent"], "introduction")


@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class RetestConversations(ConversationTestCase):
    """Conversations of the retest audit (commit 81c5f06) that failed before these fixes."""

    # --- Dates (retest item 3) ----------------------------------------------------------

    def test_two_dates_make_a_range_and_that_period_is_kept(self):
        self.say("Tìm 3 tin thể thao từ ngày 21/10/2026 đến ngày 23/10/2026", classifier="latest_news")
        (_, kwargs), = self.backend.called("latest")
        self.assertEqual((kwargs["published_from"], kwargs["published_to"]), (date(2026, 10, 21), date(2026, 10, 23)))
        self.assertEqual((kwargs.get("category"), kwargs.get("limit")), ("sports", 3))
        # "trong khoảng đó" points back to the period; it does not open an article.
        outcome = self.say("Chỉ lấy bài của Tiền Phong trong khoảng đó", classifier="article_detail")
        self.assertEqual(outcome["intent"], "latest_news")
        (_, kwargs), = self.backend.called("latest")
        self.assertEqual(kwargs.get("source"), "tienphong")
        self.assertEqual((kwargs["published_from"], kwargs["published_to"]), (date(2026, 10, 21), date(2026, 10, 23)))

    def test_category_with_a_date_uses_the_dated_query(self):
        self.say("Liệt kê bài báo thuộc chuyên mục thể thao ngày 26/10/2026", classifier="topic_search")
        self.assertFalse(self.backend.called("category"))
        (_, kwargs), = self.backend.called("latest")
        self.assertEqual((kwargs.get("category"), kwargs["published_from"]), ("sports", date(2026, 10, 26)))

    def test_dated_topic_search_filters_in_the_query(self):
        self.say("Tìm tin về giá vàng ngày 18/09/2026", classifier="semantic_search")
        (args, kwargs), = self.backend.called("latest_topic")
        self.assertEqual((kwargs["published_from"], kwargs["published_to"]), (date(2026, 9, 18), date(2026, 9, 18)))
        self.assertIn("vàng", args[0])

    # --- Small talk and slot-only questions (item 4) ---------------------------------------

    def test_question_with_only_a_date_after_a_greeting_is_a_list(self):
        self.say("Xin chào", classifier="greeting")
        outcome = self.say("Có tin gì mới ngày 27/10/2026?", classifier="question_answering")
        self.assertEqual(outcome["intent"], "latest_news")
        self.assertFalse(self.backend.called("retrieve"))
        self.assertNotIn("Hiểu là câu hỏi tiếp theo", outcome["answer"])
        self.say("Chỉ lấy đúng 2 tin công nghệ", classifier="topic_search")
        (_, kwargs), = self.backend.called("latest")
        self.assertEqual((kwargs.get("category"), kwargs.get("limit")), ("tech", 2))
        self.assertEqual(kwargs["published_from"], date(2026, 10, 27))

    # --- Several articles (items 3, 5, 6) ------------------------------------------------------

    def test_comparison_keeps_the_users_order_and_the_list(self):
        self.open_latest()
        outcome = self.say("So sánh bài số 2 và bài số 5", classifier="article_detail")
        self.assertEqual(outcome["intent"], "question_answering")
        self.assertEqual(self.titles(outcome), [LATEST[1]["title"], LATEST[4]["title"]])
        # Each article answers from its own passages, labelled with its list number.
        (args, kwargs), = self.backend.called("llm")
        evidence = kwargs["evidence"]
        self.assertEqual(evidence.positions, [2, 5])
        self.assertEqual([a["title"] for a in args[1]], [LATEST[1]["title"], LATEST[4]["title"]])
        self.assertIn("Bài 2 trong danh sách", evidence.context())
        self.assertEqual(outcome["list_positions"], [2, 5])
        self.assertEqual(len(self.state.items), len(LATEST))  # still the list of six
        # "ban đầu" describes the list; it does not veto the number.
        outcome = self.say("Bây giờ mở bài số 5 trong danh sách 6 tin ban đầu", classifier="article_detail")
        self.assertEqual(self.titles(outcome), [LATEST[4]["title"]])

    def test_ordinals_joined_by_and_are_both_articles(self):
        self.open_latest()
        outcome = self.say("So sánh bài thứ hai và bài thứ ba trong danh sách", classifier="question_answering")
        self.assertEqual(self.titles(outcome), [LATEST[1]["title"], LATEST[2]["title"]])

    def test_each_requested_article_is_summarized(self):
        self.open_latest()
        outcome = self.say("Tóm tắt cả bài 2 và bài 3, mỗi bài một câu.", classifier="summarization")
        self.assertEqual(outcome["intent"], "summarization")
        self.assertEqual(self.titles(outcome), [LATEST[1]["title"], LATEST[2]["title"]])
        self.assertIn("Bài 2 trong danh sách", outcome["answer"])
        self.assertIn("Bài 3 trong danh sách", outcome["answer"])
        self.assertEqual(outcome["answer"].count("thứ nhất."), 2)
        self.assertNotIn("thứ hai.", outcome["answer"])

    # --- Negation and exclusion (item 8) ---------------------------------------------------------

    def test_negated_article_number_is_skipped(self):
        self.open_latest()
        for text, wanted in (
            ("Tôi muốn đọc bài 3, không phải bài 2.", LATEST[2]),
            ("Đọc bài thứ ba, đừng mở bài thứ hai.", LATEST[2]),
            ("Mở bài số 2 thay cho bài số 1.", LATEST[1]),
        ):
            with self.subTest(text=text):
                outcome = self.say(text, classifier="article_detail")
                self.assertEqual(self.titles(outcome), [wanted["title"]])

    def test_excluded_publisher_is_left_out(self):
        for text, excluded in (
            ("Tìm 3 bài về giá vàng, trừ VnExpress", "vnexpress"),
            ("Tìm các bài về Arsenal nhưng không lấy bài của Tiền Phong", "tienphong"),
        ):
            with self.subTest(text=text):
                self.say(text)
                (_, kwargs), = self.backend.called("search")
                self.assertEqual(kwargs.get("exclude_sources"), [excluded])
                self.assertIsNone(kwargs.get("source"))

    def test_all_publishers_and_the_list_just_shown(self):
        self.backend.search_results = [LATEST[1], MBAPPE[0]]
        self.say("Tìm bài báo về Mbappe trên Tuổi Trẻ")
        self.say("Bỏ lọc theo báo, tìm trên tất cả các báo", classifier="semantic_search")
        (_, kwargs), = self.backend.called("search")
        self.assertIsNone(kwargs.get("source"))
        self.say("Chỉ lấy hai bài của VnExpress", classifier="topic_search")
        (_, kwargs), = self.backend.called("search")
        self.assertEqual(kwargs.get("source"), "vnexpress")
        self.assertEqual(len(self.state.items), 2)
        # "hai bài vừa nêu" is the list just shown (item 10), not a new count.
        outcome = self.say("Chỉ trong hai bài vừa nêu, bài nào nhắc tới Kane?", classifier="question_answering")
        self.assertEqual(self.titles(outcome), [LATEST[1]["title"]])

    # --- More of the same list (item 7) ------------------------------------------------------------

    def test_other_articles_leave_out_the_ones_shown(self):
        self.say("Cho tôi 3 tin mới nhất", classifier="latest_news")
        outcome = self.say("Cho tôi ba bài khác, bỏ qua những bài vừa xem.", classifier="latest_news")
        self.assertEqual(self.titles(outcome), [a["title"] for a in LATEST[3:6]])
        (_, kwargs), = self.backend.called("latest")
        self.assertEqual(kwargs.get("limit"), 6)



@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class PremiseAndRoleConversations(ConversationTestCase):
    """Retest of 9f6a83a: false premises and swapped roles while an article is open."""

    def open_article(self, item: dict) -> None:
        self.backend.latest_articles = [*LATEST, item]
        self.open_latest()
        self.say(f"Đọc bài số {len(LATEST) + 1}", classifier="article_detail")

    def test_false_premise_is_checked_against_the_open_article(self):  # "No false evidence for numbers"
        self.open_article(DIGITAL_WEEK)
        self.backend.retrieve_articles = [HUE_RAIN]
        outcome = self.say("Đúng hay sai: Tuần lễ Số được tổ chức ở Huế cho 700 đại biểu?", "question_answering")
        self.assertEqual(self.titles(outcome), [DIGITAL_WEEK["title"]])
        self.assertIn("không xác nhận", outcome["answer"])
        self.assertIn("không nhắc đến Huế", outcome["answer"])
        self.assertIn("tại Quảng Ninh", outcome["answer"])
        self.assertIn("dự kiến quy tụ 500 đại biểu", outcome["answer"])
        self.assertFalse(self.backend.called("llm") or self.backend.called("retrieve"))
        # The follow-up stays on the article ("con số" frames the question, "tham dự" is a
        # verb), and the article says "dự kiến", not "thực tế": it answers the choice.
        outcome = self.say("Đại biểu là con số dự kiến hay con số thực tế đã tham dự?", "question_answering")
        self.assertEqual(self.titles(outcome), [DIGITAL_WEEK["title"]])
        self.assertIn("đó là con số dự kiến", outcome["answer"])
        self.assertIn("dự kiến quy tụ 500 đại biểu", outcome["answer"])
        self.assertFalse(self.backend.called("retrieve") or self.backend.called("llm"))

    def test_claim_the_article_contains_is_answered_with_its_sentence(self):
        # No bare "Có": the article's own sentence answers a yes-or-no question.
        self.open_article(DIGITAL_WEEK)
        outcome = self.say("Có phải Tuần lễ Số diễn ra tại Quảng Ninh không?", "question_answering")
        self.assertEqual(self.titles(outcome), [DIGITAL_WEEK["title"]])
        self.assertIn("> Tuần lễ Số Quốc tế Việt Nam 2026 (VIDW 2026) diễn ra từ ngày 28 đến 30/10 tại Quảng Ninh.", outcome["answer"])
        self.assertFalse(self.backend.called("llm"))

    def test_claim_about_another_subject_leaves_the_article(self):
        self.open_latest()
        self.say("Đọc toàn văn bài số 2", classifier="article_detail")
        self.backend.retrieve_articles = MBAPPE
        self.say("Messi có đoạt Quả Bóng Vàng không?", "question_answering")
        self.assertTrue(self.backend.called("retrieve"))

    def test_false_premise_without_an_open_article(self):
        self.backend.retrieve_articles = [HUE_RAIN, DIGITAL_WEEK]
        outcome = self.say("Đúng hay sai: Tuần lễ Số được tổ chức ở Huế cho 700 đại biểu?", "question_answering")
        # The weather report mentions Huế and 700 but is not about the event.
        self.assertIn("không tìm thấy bài nào xác nhận", outcome["answer"])
        self.assertEqual(self.titles(outcome), [DIGITAL_WEEK["title"]])
        self.assertFalse(self.backend.called("llm"))

    def test_claim_details_do_not_steer_the_search(self):
        # Only the weather report comes back: the search runs again without "Huế" and
        # "700", and with still no article about the subject the reply says so.
        self.backend.retrieve_articles = [HUE_RAIN]
        outcome = self.say("Đúng hay sai: Tuần lễ Số được tổ chức ở Huế cho 700 đại biểu?", "question_answering")
        first, second = self.backend.called("retrieve")
        self.assertIn("Huế", first[0][0])
        self.assertNotIn("Huế", second[0][0])
        self.assertNotIn("700", second[0][0])
        self.assertIn("không tìm thấy bài báo nào về", outcome["answer"])
        self.assertFalse(self.backend.called("llm"))

    def test_role_questions_stay_on_the_open_article(self):  # "Incoming versus outgoing coach"
        self.open_article(LONG_AN)
        self.backend.retrieve_articles = MBAPPE
        for text in ("CLB Long An chia tay ai và bổ nhiệm ai thay thế?",
                     "Ai rời ghế, ai nhận ghế?",
                     "Sau bốn vòng đấu Long An giành bao nhiêu điểm?"):
            with self.subTest(text=text):
                outcome = self.say(text, "question_answering")
                self.assertEqual(self.titles(outcome), [LONG_AN["title"]])
                self.assertFalse(self.backend.called("retrieve"))



@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class PassageFirstConversations(ConversationTestCase):
    """Answers come from selected passages, and follow-ups keep the subject and its passages."""

    def test_follow_up_keeps_the_article_and_its_passages(self):  # "Percentage in isolation"
        self.backend.retrieve_articles = [MICROSOFT]
        first = self.say("Microsoft đề xuất bao nhiêu nhân viên tự nguyện nghỉ việc?", "question_answering")
        self.assertEqual(self.titles(first), [MICROSOFT["title"]])
        self.assertTrue(first["evidence_offered"])
        self.backend.retrieve_articles = [MBAPPE[0]]  # a new search would find something else
        for text in ("Họ chiếm bao nhiêu phần trăm nhân lực toàn cầu của Microsoft?", "Tỷ lệ là bao nhiêu?"):
            with self.subTest(text=text):
                outcome = self.say(text, "question_answering")
                self.assertEqual(self.titles(outcome), [MICROSOFT["title"]])
                self.assertFalse(self.backend.called("retrieve"))
                (_, kwargs), = self.backend.called("llm")
                self.assertIn("hơn 25% nhân lực toàn cầu", kwargs["evidence"].context())

    def test_explicit_subject_change_starts_a_new_search(self):
        self.backend.retrieve_articles = [MICROSOFT]
        self.say("Microsoft đề xuất bao nhiêu nhân viên tự nguyện nghỉ việc?", "question_answering")
        self.backend.retrieve_articles = [LONG_AN]
        outcome = self.say("Long An giành bao nhiêu điểm?", "question_answering")
        self.assertTrue(self.backend.called("retrieve"))
        self.assertEqual(self.titles(outcome), [LONG_AN["title"]])

    def test_acceptance_count_is_not_documented(self):
        self.backend.retrieve_articles = [MICROSOFT]
        outcome = self.say("Có bao nhiêu nhân viên Microsoft đã chấp nhận nghỉ việc?", "question_answering")
        self.assertIn("chỉ ghi nhận lời đề xuất", outcome["answer"])
        self.assertIn("đề xuất với khoảng 60.000 nhân viên", outcome["answer"])
        self.assertFalse(self.backend.called("llm"))

    def test_quote_from_the_article_just_cited(self):
        self.backend.retrieve_articles = [FUNDING]
        self.say("Ai tài trợ 11 công trình vệ sinh ở Sơn La?", "question_answering")
        self.backend.search_results = MBAPPE  # an unrelated search must not be opened
        outcome = self.say("Hãy trích câu nêu số tiền trong bài vừa dẫn", "summarization")
        self.assertEqual(self.titles(outcome), [FUNDING["title"]])
        self.assertIn("> Sáng 22/10, Quỹ Hy vọng cùng Opella Việt Nam và BW Industrial", outcome["answer"])
        self.assertIn("tổng tài trợ gần 3 tỷ đồng. [Nguồn 1]", outcome["answer"])
        self.assertFalse(self.backend.called("search") or self.backend.called("llm"))

    def test_yes_or_no_is_answered_with_the_sentences(self):
        self.backend.retrieve_articles = [FUNDING]
        outcome = self.say("Có phải BW Industrial tài trợ cả 11 công trình không?", "question_answering")
        self.assertIn("Công trình còn lại do BW Industrial tài trợ.", outcome["answer"])
        self.assertNotIn("Có.", outcome["answer"])
        self.assertFalse(self.backend.called("llm"))

    def test_ambiguous_follow_up_after_a_comparison_is_asked_back(self):
        self.open_latest()
        self.say("So sánh bài số 2 và bài số 5", classifier="question_answering")
        outcome = self.say("Bài đó đăng lúc mấy giờ?", classifier="question_answering")
        self.assertIn("Bạn muốn hỏi về bài nào?", outcome["answer"])
        self.assertIn("bài 2", outcome["answer"])
        self.assertIn("bài 5", outcome["answer"])
        self.assertFalse(self.backend.called("llm"))
        self.assertEqual(len(self.state.items), len(LATEST))  # the list is unchanged



@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class ComparisonAndContinuationScope(ConversationTestCase):
    """Final acceptance cases: explicit comparisons, the pair just compared, thin follow-ups."""

    def compare_pair(self) -> dict:
        self.backend.latest_articles = [LONG_AN, MICROSOFT]
        self.say("Tin tức mới nhất", classifier="latest_news")
        return self.say("So sánh bài 1 và bài 2. Hãy ghi chủ đề riêng của mỗi bài.", classifier="semantic_search")

    def test_explicit_comparison_is_not_a_search(self):
        outcome = self.compare_pair()
        self.assertEqual(self.titles(outcome), [LONG_AN["title"], MICROSOFT["title"]])
        self.assertFalse(self.backend.called("search"))
        (_, kwargs), = self.backend.called("llm")
        self.assertEqual(kwargs["evidence"].positions, [1, 2])

    def test_question_within_the_pair_just_compared(self):
        self.compare_pair()
        outcome = self.say("Trong hai bài vừa so sánh, bài nào nói về 60.000 nhân viên?", "question_answering")
        self.assertIn("bài 2 nhắc tới", outcome["answer"])
        self.assertIn("60.000 nhân viên", outcome["answer"])
        self.assertEqual(self.titles(outcome), [MICROSOFT["title"]])
        self.assertFalse(self.backend.called("search") or self.backend.called("retrieve"))
        self.assertEqual(len(self.state.items), 2)  # the list is unchanged
        outcome = self.say("Trong hai bài vừa so sánh, bài nào nói về Quả Bóng Vàng?", "question_answering")
        self.assertIn("Không bài nào trong các bài vừa so sánh", outcome["answer"])
        self.assertFalse(self.backend.called("search") or self.backend.called("retrieve"))

    def test_the_remaining_article_of_the_pair(self):
        self.compare_pair()
        # Right after the comparison neither article was singled out: ask.
        outcome = self.say("Bài còn lại cho biết HLV nào mới được bổ nhiệm?", "question_answering")
        self.assertIn("Bạn muốn nói bài nào", outcome["answer"])
        self.say("Trong hai bài vừa nêu, bài nào nói về 60.000 nhân viên?", "question_answering")
        self.backend.retrieve_articles = MBAPPE
        outcome = self.say("Bài còn lại cho biết HLV nào mới được bổ nhiệm?", "question_answering")
        self.assertEqual(self.titles(outcome), [LONG_AN["title"]])
        self.assertFalse(self.backend.called("retrieve") or self.backend.called("search"))
        (_, kwargs), = self.backend.called("llm")
        self.assertIn("HLV Huỳnh Ngọc San đã được chọn", kwargs["evidence"].context())

    def open_digital_week(self) -> None:
        self.backend.latest_articles = [*LATEST, DIGITAL_WEEK]
        self.open_latest()
        self.say(f"Đọc bài số {len(LATEST) + 1}", classifier="article_detail")
        self.say("Tuần lễ Số diễn ra ở đâu?", "question_answering")

    def test_thin_continuation_keeps_the_open_article(self):
        self.open_digital_week()
        self.backend.retrieve_articles = [HUE_RAIN]
        outcome = self.say("Có biết chính xác số người đã đến sau khi sự kiện kết thúc không?", "question_answering")
        self.assertEqual(self.titles(outcome), [DIGITAL_WEEK["title"]])
        self.assertIn("chỉ nêu con số dự kiến", outcome["answer"])
        self.assertIn("dự kiến quy tụ 500 đại biểu", outcome["answer"])
        self.assertFalse(self.backend.called("retrieve") or self.backend.called("llm"))

    def test_genuine_topic_change_leaves_the_open_article(self):
        self.open_digital_week()
        self.backend.retrieve_articles = [LONG_AN]
        for text in ("Vì sao giá vàng tăng?", "Long An giành bao nhiêu điểm?"):
            with self.subTest(text=text):
                self.say(text, "question_answering")
                self.assertTrue(self.backend.called("retrieve"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
