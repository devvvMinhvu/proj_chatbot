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
        "body": body or title,
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
SUMMARY = "Đây là câu tóm tắt thứ nhất. Đây là câu tóm tắt thứ hai. Đây là câu tóm tắt thứ ba."
BY_ID = {item["article_id"]: item for item in LATEST + SON_LA + MBAPPE}


class Backend:
    """Stubs for everything below the handler, recording what was asked for."""

    def __init__(self, test: unittest.TestCase) -> None:
        self.test = test
        self.calls: list[tuple[str, tuple, dict]] = []
        self.search_results: list[dict] = MBAPPE
        self.retrieve_articles: list[dict] = []

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
            items = [a for a in LATEST if filters.get("source") in (None, a["source"])
                     and filters.get("category") in (None, a["category"])]
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
                "llm", lambda q, articles: iter([f"<answer from {len(articles)} articles>"]))),
            patch.object(handler, "generate_answer", side_effect=self._record("llm", lambda q, a: "<answer>")),
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


@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class AuditConversations(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
