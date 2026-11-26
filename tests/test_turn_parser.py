"""Tests for the turn parser's slots and the conversation state's list handling.

The parser needs the corpus keyword statistics (``python -m nlp.keyword_stats``);
those tests are skipped when they are not available.
"""

from __future__ import annotations

from datetime import date
import unittest

from nlp.conversation_state import ConversationState, ListRequest
from nlp.keyword_stats import KEYWORD_STATS_PATH, item_nouns
from nlp.turn_parser import parse_turn

TODAY = date(2026, 11, 22)


@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class TurnParserTests(unittest.TestCase):
    def test_item_words_are_learned_from_the_intent_examples(self):
        self.assertTrue({"bài", "tin", "kết quả"} <= item_nouns())
        self.assertNotIn("khoảng", item_nouns())

    def test_number_next_to_a_word_is_not_a_date(self):
        parsed = parse_turn("Bây giờ mở bài số 9 trong danh sách 10 tin ban đầu", TODAY)
        self.assertEqual((parsed.item_refs, parsed.count, parsed.date_from), ([9], 10, None))

    def test_two_dates_are_a_range(self):
        parsed = parse_turn("Tin từ ngày 21/10/2026 đến ngày 23/10/2026", TODAY)
        self.assertEqual((parsed.date_from, parsed.date_to), (date(2026, 10, 21), date(2026, 10, 23)))

    def test_ordinal_after_and_is_an_item_not_a_weekday(self):
        parsed = parse_turn("So sánh bài thứ hai và bài thứ ba", TODAY)
        self.assertEqual((parsed.item_refs, parsed.date_from), ([2, 3], None))

    def test_negation_and_exclusion(self):
        self.assertEqual(parse_turn("Đọc bài 3, không phải bài 2", TODAY).excluded_refs, [2])
        parsed = parse_turn("Tin về giá vàng, trừ VnExpress", TODAY)
        self.assertEqual((parsed.source, parsed.exclude_sources), (None, ["vnexpress"]))
        # A question ending in "không" is not a negation.
        parsed = parse_turn("Bài 2 có nhắc đến VnExpress không?", TODAY)
        self.assertEqual((parsed.item_refs, parsed.source, parsed.exclude_sources), ([2], "vnexpress", []))
        self.assertTrue(parse_turn("Tìm trên tất cả các báo", TODAY).all_sources)
        self.assertTrue(parse_turn("tim tren moi bao", TODAY).all_sources)
        # Unaccented "mới bao nhiêu" is a question, not "mọi báo".
        parsed = parse_turn("Du an tai Son La da xay moi bao nhieu cong trinh?", TODAY)
        self.assertFalse(parsed.all_sources)

    def test_demonstrative_after_other_nouns_points_back(self):
        for text in ("Dự án này ở tỉnh nào?", "Chỉ lấy bài của Tiền Phong trong khoảng đó"):
            with self.subTest(text=text):
                parsed = parse_turn(text, TODAY)
                self.assertTrue(parsed.refers_back)
                self.assertIsNone(parsed.relative)
        self.assertEqual(parse_turn("Đọc tin này", TODAY).relative, "current")
        self.assertIsNone(parse_turn("Tin tuần trước", TODAY).relative)

    def test_recent_verb_and_other(self):
        self.assertEqual(parse_turn("Bài vừa đọc đăng ở báo nào?", TODAY).relative, "current")
        parsed = parse_turn("Trong hai bài vừa nêu, bài nào nhắc tới Kane?", TODAY)
        self.assertTrue(parsed.refers_to_list)
        self.assertIsNone(parsed.count)
        self.assertTrue(parse_turn("Cho tôi ba bài khác", TODAY).wants_other)

    def test_yes_no_questions_and_their_claims(self):
        parsed = parse_turn("Đúng hay sai: Tuần lễ Số được tổ chức ở Huế cho 700 đại biểu?", TODAY)
        self.assertTrue(parsed.yes_no)
        self.assertNotIn("sai", parsed.topic_words)  # "đúng hay sai" asks; it is not a subject
        self.assertEqual(parsed.topic_nouns[:1], ["tuần lễ"])
        self.assertTrue(parse_turn("Bài 2 có nhắc đến VnExpress không?", TODAY).yes_no)
        self.assertTrue(parse_turn("Có phải Kane đoạt Quả Bóng Vàng khi mới 11 tuổi không?", TODAY).yes_no)
        for text in ("Có bao nhiêu đại biểu dự kiến tham dự?", "CLB Long An chia tay ai?", "Đọc bài 3"):
            with self.subTest(text=text):
                self.assertFalse(parse_turn(text, TODAY).yes_no)
        self.assertIn("chia tay", parse_turn("CLB Long An chia tay ai và bổ nhiệm ai thay thế?", TODAY).predicate_words)

    def test_list_words_name_the_articles(self):
        parsed = parse_turn("So sánh bài 1 và bài 2. Hãy ghi chủ đề riêng của mỗi bài.", TODAY)
        self.assertEqual(parsed.item_refs, [1, 2])
        self.assertTrue(parsed.refs_named)
        self.assertNotIn("hãy", parsed.entity_words + parsed.topic_words)
        self.assertFalse(parse_turn("Messi đã giành Quả Bóng Vàng lần thứ 8 chưa?", TODAY).refs_named)


class ConversationStateTests(unittest.TestCase):
    def test_new_list_closes_opened_articles_and_keeps_pins(self):
        state = ConversationState()
        pinned, opened = {"article_id": "p", "title": "P"}, {"article_id": "o", "title": "O"}
        state.select(pinned, "pin")
        state.select(opened, "open")
        state.show_list([{"article_id": "n", "title": "N"}], ListRequest(action="latest"))
        self.assertEqual(state.current_article()["article_id"], "p")

    def test_same_filters_ignores_the_count(self):
        first = ListRequest(action="latest", count=3, source="tuoitre")
        self.assertTrue(first.same_filters(ListRequest(action="latest", count=5, source="tuoitre")))
        self.assertFalse(first.same_filters(ListRequest(action="latest", count=3, source="vnexpress")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
