"""Tests for checking a yes-or-no claim, or an "A hay B" choice, against an article.

The parser needs the corpus keyword statistics (``python -m nlp.keyword_stats``);
the tests are skipped when they are not available.
"""

from __future__ import annotations

from datetime import date
import unittest

from nlp.keyword_stats import KEYWORD_STATS_PATH
from nlp.premise_check import alternative_choice, check_premise, premise_reply
from nlp.turn_parser import parse_turn

TODAY = date(2026, 11, 22)
# Shortened from the stored Digital Week article of the audit.
BODY = (
    "Nhiều chuyên gia sẽ tham dự Tuần lễ Số 2026, trong đó Bộ trưởng Vũ Hải Quân sẽ chủ trì Hội nghị bàn tròn. "
    "Tuần lễ Số Quốc tế Việt Nam 2026 (VIDW 2026) diễn ra từ ngày 28 đến 30/10 tại Quảng Ninh. "
    "Sự kiện do Bộ Khoa học và Công nghệ chủ trì tổ chức, dự kiến quy tụ 500 đại biểu trong và ngoài nước. "
    "Sự kiện do Bộ Khoa học và Công nghệ chủ trì tổ chức, dự kiến quy tụ 500 đại biểu trong và ngoài nước."
)
TEXT = "Những diễn giả nổi bật tại Tuần lễ Số 2026 vnexpress " + BODY


@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class PremiseCheckTests(unittest.TestCase):
    def premise(self, question):
        return check_premise(question, parse_turn(question, TODAY), TEXT, BODY)

    def test_missing_details_and_the_sentences_on_the_same_points(self):
        premise = self.premise("Đúng hay sai: Tuần lễ Số được tổ chức ở Huế cho 700 đại biểu?")
        self.assertEqual((premise.missing_names, premise.missing_numbers), (["Huế"], ["700"]))
        self.assertEqual(len(premise.evidence), 2)  # the repeated sentence is quoted once
        self.assertIn("tại Quảng Ninh", premise.evidence[0])
        self.assertIn("500 đại biểu", premise.evidence[1])
        reply = premise_reply(premise, "Những diễn giả nổi bật tại Tuần lễ Số 2026")
        self.assertIn("không nhắc đến Huế và không có con số 700", reply)

    def test_claims_the_article_contains_and_questions_without_a_claim(self):
        self.assertTrue(self.premise("Có phải Tuần lễ Số diễn ra tại Quảng Ninh không?").holds)
        self.assertTrue(self.premise("Tuần lễ Số có 500 đại biểu, đúng không?").holds)
        self.assertIsNone(self.premise("Tuần lễ Số diễn ra ở đâu?"))  # not a yes-or-no question

    def test_accents_count_in_names(self):
        premise = check_premise("Có phải bà Huệ chủ trì không?", parse_turn("Có phải bà Huệ chủ trì không?", TODAY),
                                "Ông Huế chủ trì.", "Ông Huế chủ trì.")
        self.assertEqual(premise.missing_names, ["Huệ"])


@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class AlternativeChoiceTests(unittest.TestCase):
    def choice(self, question):
        return alternative_choice(question, parse_turn(question, TODAY), BODY)

    def test_the_option_the_article_states(self):
        option, sentence = self.choice("Đại biểu là con số dự kiến hay con số thực tế đã tham dự?")
        self.assertEqual(option, "con số dự kiến")
        self.assertIn("dự kiến quy tụ 500 đại biểu", sentence)
        self.assertEqual(self.choice("Sự kiện diễn ra ở Quảng Ninh hay Hà Nội?")[0], "Quảng Ninh")
        self.assertEqual(self.choice("Tuần lễ Số có 500 hay 700 đại biểu?")[0], "500")

    def test_no_choice(self):
        for question in ("Đúng hay sai: Tuần lễ Số ở Huế?",  # a verdict, not a choice
                         "Bộ trưởng hay chuyên gia tham dự?",  # the article has both
                         "Sự kiện ở Đà Nẵng hay Hà Nội?"):  # the article has neither
            with self.subTest(question=question):
                self.assertIsNone(self.choice(question))


if __name__ == "__main__":
    unittest.main(verbosity=2)
