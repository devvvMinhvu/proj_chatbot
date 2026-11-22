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
