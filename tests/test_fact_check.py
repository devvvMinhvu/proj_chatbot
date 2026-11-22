"""Tests for checking generated text against its source."""

from __future__ import annotations

import unittest

from processing import fact_check

SOURCE = (
    "Harry Kane đoạt Quả Bóng Vàng 2026 ở tuổi 33. Kane gia nhập Tottenham từ tuổi 11. "
    "Quỹ Hy vọng bàn giao 11 công trình, tổng tài trợ gần 3 tỷ đồng. Dự án 43.000 tỷ đồng."
)


class FactCheckTests(unittest.TestCase):
    def test_supported_sentence(self):
        result = fact_check.check("Harry Kane giành Quả Bóng Vàng ở tuổi 33 [Nguồn 1].", SOURCE)
        self.assertTrue(result.supported)

    def test_wrong_name_and_borrowed_number(self):
        self.assertEqual(fact_check.check("Lionel Messi giành giải.", SOURCE).unsupported_names, ["lionel", "messi"])
        self.assertEqual(fact_check.check("Tổng kinh phí là 3,5 triệu tỷ.", SOURCE).unsupported_numbers, ["3,5"])

    def test_number_spellings_and_question_numbers(self):
        self.assertTrue(fact_check.check("Dự án 43000 tỷ đồng.", SOURCE).supported)
        self.assertTrue(fact_check.check("Năm 2027 chưa có.", SOURCE, allowed="Năm 2027 thì sao?").supported)

    def test_sentence_initial_capital_is_not_a_name(self):
        self.assertEqual(fact_check.names_in("Tổng tài trợ gần 3 tỷ đồng."), set())

    def test_dateline_glued_to_the_first_sentence(self):  # retest item 2
        source = "Sơn LaSáng 22/10, Quỹ Hy vọng đã bàn giao 11 công trình vệ sinh."
        self.assertEqual(fact_check.split_glued_words("Sơn LaSáng"), "Sơn La Sáng")
        self.assertTrue(fact_check.check("Dự án ở Sơn LaĐã bàn giao 11 công trình.", source).supported)
        self.assertTrue(fact_check.check("Dự án ở tỉnh Sơn La.", source).supported)
        self.assertFalse(fact_check.check("Dự án ở Lai Châu.", source).supported)

    def test_unsupported_summary_sentences_are_dropped(self):
        kept, dropped = fact_check.keep_supported_sentences(
            "Kane đoạt giải ở tuổi 33. Messi về nhì ở Chicago. Kane rời Tottenham.", SOURCE
        )
        self.assertEqual(kept, "Kane đoạt giải ở tuổi 33. Kane rời Tottenham.")
        self.assertEqual(dropped, 1)
        self.assertAlmostEqual(fact_check.fact_support("Kane đoạt giải ở tuổi 33. Messi về nhì.", SOURCE), 2 / 3)
        self.assertIsNone(fact_check.fact_support("Một câu không có tên hay số.", SOURCE))


class GroundedAnswerMetadataTests(unittest.TestCase):
    def test_publication_time_from_metadata_is_supported(self):
        from rag.pipeline import GroundedAnswer

        articles = [{
            "title": "Harry Kane - từ người không danh hiệu đến Quả Bóng Vàng",
            "source": "vnexpress",
            "published_at_display": "27/10/2026 11:07",
            "texts": ["Kane đoạt Quả Bóng Vàng."],
        }]
        grounded = GroundedAnswer("Bài này đăng lúc mấy giờ?", articles)
        shown = grounded.feed("Bài báo được đăng lúc 11 giờ 7 phút ngày 27/10/2026. ") + grounded.finish()
        self.assertIn("11 giờ 7 phút", shown)


class PremiseAndRoleTests(unittest.TestCase):
    """The answer check does not let a question's details or swapped roles through."""

    WEEK = [{
        "title": "Những diễn giả nổi bật tại Tuần lễ Số 2026",
        "source": "vnexpress",
        "texts": ["Tuần lễ Số 2026 diễn ra tại Quảng Ninh, dự kiến quy tụ 500 đại biểu."],
    }]

    def shown(self, question, answer, articles):
        from rag.pipeline import GroundedAnswer

        grounded = GroundedAnswer(question, articles)
        return grounded.feed(answer + "\n") + grounded.finish()

    def test_question_details_may_be_denied_not_affirmed(self):
        question = "Đúng hay sai: Tuần lễ Số được tổ chức ở Huế cho 700 đại biểu?"
        shown = self.shown(question, "Đúng. Tuần lễ Số được tổ chức ở Huế cho 700 đại biểu [Nguồn 1].", self.WEEK)
        self.assertNotIn("Đúng", shown)  # the verdict goes with the sentence it stood on
        self.assertNotIn("700 đại biểu", shown)
        # A negative the passage does not state is not shown either: the article is quoted.
        shown = self.shown(question, "Sai. Bài không nhắc đến Huế; sự kiện diễn ra tại Quảng Ninh [Nguồn 1].", self.WEEK)
        self.assertNotIn("Sai", shown)
        self.assertIn("> Tuần lễ Số 2026 diễn ra tại Quảng Ninh, dự kiến quy tụ 500 đại biểu. [Nguồn 1]", shown)

    def test_verdict_helpers(self):
        self.assertTrue(fact_check.is_verdict("Đúng."))
        self.assertTrue(fact_check.is_verdict("Không ạ [Nguồn 1]."))
        self.assertFalse(fact_check.is_verdict("Đúng, sự kiện ở Quảng Ninh."))
        self.assertTrue(fact_check.denies("Bài không nhắc đến Huế."))
        self.assertFalse(fact_check.denies("Tuần lễ Số diễn ra ở Huế."))

    def test_swapped_roles_are_replaced_by_the_article_sentences(self):
        articles = [{
            "title": "CLB hạng nhất đầu tiên thay HLV",
            "source": "tuoitre",
            "texts": [
                "CLB Long An đã thay HLV Trịnh Văn Hậu sau 4 vòng đấu. "
                "HLV Huỳnh Ngọc San đã được chọn. "
                "Đội bóng xứ chè còn có chân sút Victor Mansaray."
            ],
        }]
        shown = self.shown(
            "CLB Long An chia tay ai và bổ nhiệm ai thay thế?",
            "CLB Long An chia tay HLV Huỳnh Ngọc San và bổ nhiệm Victor Mansaray thay thế [Nguồn 1].",
            articles,
        )
        self.assertNotIn("bổ nhiệm Victor Mansaray", shown)
        self.assertIn("không khớp với bài báo", shown)
        self.assertIn("> CLB Long An đã thay HLV Trịnh Văn Hậu sau 4 vòng đấu. [Nguồn 1]", shown)
        correct = "CLB Long An chia tay HLV Trịnh Văn Hậu và chọn HLV Huỳnh Ngọc San [Nguồn 1]."
        self.assertEqual(self.shown("CLB Long An chia tay ai?", correct, articles).strip(), correct)


class CheckedSummaryTests(unittest.TestCase):
    def test_falls_back_to_opening_sentences(self):
        from nlp.intent_handler import _checked_summary

        body = "Harry Kane đoạt Quả Bóng Vàng 2026 ở tuổi 33 tại London. Anh cảm ơn đồng đội và gia đình. Bayern vô địch."
        summary, note = _checked_summary("Lionel Messi đoạt giải ở Chicago.", body, None)
        self.assertTrue(summary.startswith("Harry Kane đoạt Quả Bóng Vàng 2026"))
        self.assertIn("câu mở đầu", note)

    def test_length_limit(self):
        from nlp.intent_handler import _checked_summary

        body = "Kane ghi bàn cho Bayern trong trận đấu. Kane được khen ngợi rất nhiều. Bayern thắng đậm."
        summary, _ = _checked_summary("Kane ghi bàn cho Bayern trong trận đấu. Kane được khen ngợi rất nhiều.", body, 1)
        self.assertEqual(summary, "Kane ghi bàn cho Bayern trong trận đấu.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
