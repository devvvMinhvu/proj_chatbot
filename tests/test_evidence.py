"""Tests for passage-first answers: which sentences answer a question, and what they support.

The articles are shortened from stored articles that earlier audits got wrong.
The parser needs the corpus keyword statistics (``python -m nlp.keyword_stats``);
the tests are skipped when they are not available.
"""

from __future__ import annotations

import unittest

from nlp.keyword_stats import KEYWORD_STATS_PATH
from rag.evidence import article_passages, find_support, select_evidence, supported_by_text


def article(key: str, title: str, body: str, source: str = "vnexpress") -> dict:
    return {"article_id": key, "title": title, "source": source, "url": f"https://example.vn/{key}",
            "published_at_display": "27/10/2026 10:00", "body": body}


WEEK = article("week", "Những diễn giả nổi bật tại Tuần lễ Số 2026", (
    "Nhiều chuyên gia, quan chức, lãnh đạo doanh nghiệp hàng đầu về công nghệ số sẽ tham dự Tuần lễ Số 2026.\n"
    "Tuần lễ Số Quốc tế Việt Nam 2026 (VIDW 2026) diễn ra từ ngày 28 đến 30/10 tại Quảng Ninh. "
    "Sự kiện do Bộ Khoa học và Công nghệ chủ trì tổ chức, dự kiến quy tụ 500 đại biểu trong và ngoài nước."
))
MICROSOFT = article("ms", "Microsoft đề xuất 60.000 nhân viên tự nguyện nghỉ việc", (
    "Khoảng 25% nhân lực có thể nghỉ việc kèm trợ cấp của Microsoft.\n"
    "Ngày 26/10, Microsoft cho biết đang đề xuất với khoảng 60.000 nhân viên nghỉ việc tự nguyện kèm trợ cấp. "
    "Con số này tương đương hơn 25% nhân lực toàn cầu của hãng.\n"
    "Microsoft có 223.000 nhân viên toàn thời gian trên toàn cầu, tính đến cuối quý II."
))
SON_LA = article("sonla", "Khánh thành cụm 11 công trình vệ sinh đạt chuẩn ở Sơn La", (
    "Sơn LaSáng 22/10, Quỹ Hy vọng cùng Opella Việt Nam và BW Industrial bàn giao cụm 11 công trình vệ sinh "
    "học đường tại 6 xã, tổng tài trợ gần 3 tỷ đồng.\n"
    "Trong đó, 10 công trình do Opella Việt Nam cùng nhãn hàng Enterogermina đồng hành, thuộc 5 xã. "
    "Công trình còn lại do BW Industrial tài trợ tại Trường Phổ thông Dân tộc Bán trú THCS Nậm Ty, xã Nậm Ty."
))
LONG_AN = article("longan", "CLB hạng nhất đầu tiên thay HLV", (
    "CLB Long An đã thay HLV Trịnh Văn Hậu sau thành tích không tốt của đội qua 4 vòng đấu đầu tiên.\n"
    "HLV Huỳnh Ngọc San ở buổi ra mắt CLB Long An chiều 26-10 - Ảnh: ANH TUẤN TRẦN\n"
    "Sau 4 trận, CLB Long An chỉ giành có 1 điểm trong trận hòa Thép Cần Thơ 0-0 trên sân nhà ở vòng 2.\n"
    "Cụ thể, HLV Huỳnh Ngọc San đã được chọn và đã chính thức bắt tay vào công việc vào chiều 26-10.\n"
    "Đội bóng xứ chè còn có chân sút Victor Mansaray đang dẫn đầu danh sách Vua phá lưới với 8 bàn thắng. "
    "HLV Huỳnh Ngọc San là gương mặt quen thuộc với người hâm mộ Long An."
), source="tuoitre")
KANE = article("kane", "Harry Kane - từ người không danh hiệu đến Quả Bóng Vàng", (
    "Từng bị Arsenal loại khỏi học viện, trải qua bốn lần cho mượn và hơn một thập niên không danh hiệu, "
    "Harry Kane cuối cùng chạm đến đỉnh cao bóng đá thế giới khi đoạt Quả Bóng Vàng 2026 ở tuổi 33.\n"
    "Mãi đến mùa 2014-2015, Kane mới thực sự bùng nổ trong màu áo Tottenham. "
    "Anh ghi 21 bàn ở Ngoại hạng Anh mùa đó.\n"
    "Trong gần 10 năm tiếp theo, Kane ba lần giành Vua phá lưới Ngoại hạng Anh.\n"
    "Anh cảm ơn CLB đã giúp mình thực hiện giấc mơ, từ khi gia nhập ở tuổi 11."
))
TRANG_MOON = article("moon", "Trang Moon gây cười với vai cô bán bánh rán", (
    "Diễn viên Trang Moon hóa thân Tươi - cô gái bán bánh rán với chất giọng địa phương - trong phim "
    "''Căn nhà không mái che''.\n"
    "Nhân vật Tươi xuất hiện từ tập 3 của series, gây ấn tượng với màn chặn đầu xe tải của Toản (Duy Hưng)."
))


def supports(question: str, item: dict, answer: str) -> bool:
    return bool(find_support(answer, select_evidence(question, [item])))


@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class PassageTests(unittest.TestCase):
    def test_passages_keep_their_place_in_the_article(self):
        passages = article_passages(SON_LA)
        self.assertEqual([p.index for p in passages[:3]], [-1, 0, 1])
        self.assertTrue(passages[2].text.startswith("Sơn La Sáng 22/10"))  # the glued dateline is split
        self.assertEqual([p.pid for p in article_passages(SON_LA)], [p.pid for p in passages])

    def test_identity_does_not_depend_on_what_was_read(self):
        # Two retrieved chunks of one article: positions restart, identities do not collide,
        # and a sentence keeps its identity in the chunk and in the whole article.
        first = article_passages({"article_id": "example", "title": "T", "texts": ["Câu một. Câu hai."]})
        second = article_passages({"article_id": "example", "title": "T", "texts": ["Câu ba. Câu hai."]})
        whole = article_passages({"article_id": "example", "title": "T", "body": "Câu một. Câu hai. Câu ba."})
        body = lambda ps: {p.text: p.pid for p in ps if p.index >= 1}
        self.assertEqual([p.index for p in first if p.index >= 1], [1, 2])
        self.assertEqual([p.index for p in second if p.index >= 1], [1, 2])
        self.assertNotEqual(body(first)["Câu một."], body(second)["Câu ba."])
        self.assertEqual(body(first)["Câu hai."], body(second)["Câu hai."])
        self.assertEqual(body(whole), {**body(first), **body(second)})
        self.assertTrue(body(whole)["Câu một."].startswith("example#"))

    def test_a_breakdown_comes_with_its_total(self):
        evidence = select_evidence("Ai tài trợ 11 công trình vệ sinh?", [SON_LA])
        texts = [p.text for p in evidence.passages[0]]
        self.assertTrue(any(t.startswith("Trong đó, 10 công trình") for t in texts))
        self.assertTrue(any(t.startswith("Công trình còn lại") for t in texts))

    def test_only_articles_with_a_matching_passage_are_kept(self):
        evidence = select_evidence("Long An giành bao nhiêu điểm?", [WEEK, LONG_AN])
        self.assertEqual([a["article_id"] for a in evidence.articles], ["longan"])


@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class WhatThePassagesEstablish(unittest.TestCase):
    def test_expected_is_not_actual(self):  # Digital Week
        question = "Có bao nhiêu đại biểu thực tế tham dự?"
        reply = select_evidence(question, [WEEK]).partial_reply()
        self.assertIn("chỉ nêu con số dự kiến", reply)
        self.assertIn("dự kiến quy tụ 500 đại biểu", reply)
        self.assertFalse(supports(question, WEEK, "Có 500 đại biểu tham dự [Nguồn 1]."))
        self.assertTrue(supports("Có bao nhiêu đại biểu dự kiến tham dự?", WEEK,
                                 "Sự kiện dự kiến quy tụ 500 đại biểu trong và ngoài nước [Nguồn 1]."))

    def test_what_happened_in_other_wordings(self):
        # General rules, not the audit's wording: "số" + a noun asks for a count, and
        # "đã …" or "chính xác" in a number question asks what actually happened.
        for question in ("Số khách đã có mặt là bao nhiêu?",
                         "Có bao nhiêu đại biểu đã có mặt sau khi hội nghị khép lại?",
                         "Số đại biểu tham dự cụ thể là bao nhiêu?",
                         "Có biết chính xác số người đã đến sau khi sự kiện kết thúc không?"):
            with self.subTest(question=question):
                reply = select_evidence(question, [WEEK]).partial_reply()
                self.assertIn("chỉ nêu con số dự kiến", reply)
                self.assertIn("dự kiến quy tụ 500 đại biểu", reply)
        reply = select_evidence("Bao nhiêu nhân viên Microsoft đã nghỉ việc?", [MICROSOFT]).partial_reply()
        self.assertIn("chỉ ghi nhận lời đề xuất", reply)
        self.assertIn("đề xuất với khoảng 60.000 nhân viên", reply)

    def test_general_rules_do_not_fire_on_ordinary_questions(self):
        # A stated fact with "đã" is still an answer.
        question = "Long An đã giành bao nhiêu điểm?"
        self.assertIsNone(select_evidence(question, [LONG_AN]).partial_reply())
        self.assertTrue(supports(question, LONG_AN, "CLB Long An chỉ giành 1 điểm sau 4 trận."))
        # "số điện thoại" is not a count: nothing in the article counts telephones.
        evidence = select_evidence("Số điện thoại của ban tổ chức là gì?", [WEEK])
        self.assertTrue(evidence.need.unit_from_noun)
        self.assertIsNone(evidence.partial_reply())
        # "bài số 3" picks an article; it asks for no count.
        self.assertFalse(select_evidence("Đọc bài số 3", [WEEK]).need.wants_number)

    def test_an_offer_is_not_an_acceptance(self):  # Microsoft
        for question in ("Bao nhiêu nhân viên Microsoft đã chấp nhận nghỉ việc?",
                         "Có bao nhiêu người đã chấp nhận đề xuất của Microsoft?"):
            with self.subTest(question=question):
                reply = select_evidence(question, [MICROSOFT]).partial_reply()
                self.assertIn("chỉ ghi nhận lời đề xuất", reply)
                self.assertIn("không nêu bao nhiêu người", reply)
        self.assertFalse(supports("Microsoft đề xuất bao nhiêu nhân viên nghỉ việc?", MICROSOFT,
                                  "60.000 nhân viên nghỉ việc [Nguồn 1]."))

    def test_a_share_keeps_its_noun_and_its_approximation(self):  # Microsoft workforce share
        question = "Họ chiếm bao nhiêu phần trăm nhân lực toàn cầu của Microsoft?"
        self.assertTrue(supports(question, MICROSOFT, "Con số này tương đương hơn 25% nhân lực toàn cầu của hãng."))
        self.assertFalse(supports(question, MICROSOFT, "Họ chiếm 25% nhân lực toàn cầu."))
        self.assertFalse(supports(question, MICROSOFT, "Họ chiếm khoảng 25% nhân lực toàn cầu của hãng."))
        self.assertFalse(supports(question, MICROSOFT, "Microsoft chiếm 14,74% thị phần toàn cầu."))

    def test_totals_and_shares(self):  # Sơn La
        question = "Số tiền tài trợ cho 11 công trình vệ sinh ở Sơn La là bao nhiêu?"
        self.assertIsNone(select_evidence(question, [SON_LA]).partial_reply())
        self.assertTrue(supports(question, SON_LA, "Tổng tài trợ gần 3 tỷ đồng [Nguồn 1]."))
        self.assertFalse(supports(question, SON_LA, "Tổng tài trợ là 3 tỷ đồng."))
        self.assertFalse(supports(question, SON_LA, "Số tiền tài trợ là 1,7 tỷ đồng."))
        self.assertFalse(supports(question, SON_LA, "Bài viết không nêu số tiền tài trợ cho 11 công trình."))
        compound = "BW Industrial tài trợ bao nhiêu công trình, còn Opella đồng hành bao nhiêu?"
        self.assertFalse(supports(compound, SON_LA, "BW Industrial tài trợ 11 công trình."))
        self.assertTrue(supports(compound, SON_LA, "Opella Việt Nam đồng hành 10 công trình."))
        reply = select_evidence(compound, [SON_LA]).extractive_reply()
        self.assertIn("Công trình còn lại do BW Industrial tài trợ", reply)
        self.assertIn("10 công trình do Opella Việt Nam", reply)
        self.assertFalse(supports("Có phải BW Industrial tài trợ cả 11 công trình không?", SON_LA,
                                  "Có, BW Industrial tài trợ cả 11 công trình."))

    def test_roles_and_teams(self):  # Long An coach and the opponent's striker
        question = "CLB Long An chia tay ai và bổ nhiệm ai thay thế?"
        self.assertTrue(supports(question, LONG_AN,
                                 "CLB Long An chia tay HLV Trịnh Văn Hậu và bổ nhiệm HLV Huỳnh Ngọc San thay thế."))
        self.assertFalse(supports(question, LONG_AN,
                                  "CLB Long An chia tay HLV Huỳnh Ngọc San và bổ nhiệm Victor Mansaray thay thế."))
        self.assertFalse(supports("Victor Mansaray là cầu thủ của đội nào?", LONG_AN, "Victor Mansaray thuộc CLB Long An."))
        points = "Sau bốn vòng đấu Long An giành bao nhiêu điểm?"
        self.assertTrue(supports(points, LONG_AN, "CLB Long An chỉ giành 1 điểm sau 4 trận."))
        self.assertFalse(supports(points, LONG_AN, "Long An giành 7 điểm sau 4 vòng đấu."))

    def test_a_count_of_something_else(self):  # Kane's three top-scoring titles
        question = "Kane giành Quả Bóng Vàng mấy lần?"
        self.assertFalse(supports(question, KANE, "Kane đã giành Quả Bóng Vàng 3 lần."))
        reply = select_evidence(question, [KANE]).partial_reply()
        self.assertIn("không có con số trả lời trực tiếp", reply)
        self.assertIn("Kane ba lần giành Vua phá lưới Ngoại hạng Anh", reply)
        self.assertTrue(supports("Kane ghi 21 bàn ở giải nào, mùa nào?", KANE,
                                 "Kane ghi 21 bàn ở Ngoại hạng Anh mùa 2014-2015."))
        self.assertFalse(supports("Kane bao nhiêu tuổi khi đoạt Quả Bóng Vàng?", KANE,
                                  "Kane 11 tuổi khi đoạt Quả Bóng Vàng."))

    def test_the_answer_comes_from_the_clause_with_the_questions_number(self):  # Kane's three titles
        profile = dict(KANE, body=KANE["body"] + (
            "\nTrong màu áo tuyển Anh, tiền đạo này giành Vua phá lưới World Cup 2018, "
            "nhưng đội nhà dừng bước ở bán kết."
        ))
        question = "Trong bài, ba danh hiệu Vua phá lưới thuộc giải đấu nào?"
        self.assertFalse(supports(question, profile, "Vua phá lưới World Cup 2018: Tuyển Anh [Nguồn 1]"))
        self.assertTrue(supports(question, profile, "Ngoại hạng Anh [Nguồn 1]"))
        reply = select_evidence(question, [profile]).extractive_reply()
        self.assertTrue(reply.startswith("Theo bài viết:\n\n> Trong gần 10 năm tiếp theo, Kane ba lần giành Vua phá lưới"))

    def test_cast_and_character(self):  # Trang Moon plays Tươi; Duy Hưng plays Toản
        question = "Trang Moon là ai trong phim?"
        self.assertFalse(supports(question, TRANG_MOON, "Trang Moon là tên nhân vật."))
        self.assertTrue(supports("Ai đóng vai Tươi?", TRANG_MOON,
                                 "Trang Moon hóa thân Tươi trong phim Căn nhà không mái che."))
        # A bare name stands for the question's claim, and needs to share a clause with Tươi.
        self.assertFalse(supports("Trong bài này, ai đóng vai Tươi?", TRANG_MOON, "Duy Hưng [Nguồn 1]"))
        self.assertTrue(supports("Trong bài này, ai đóng vai Tươi?", TRANG_MOON, "Trang Moon [Nguồn 1]"))

    def test_no_plan_the_article_does_not_state(self):
        self.assertFalse(supports("Tỷ lệ là bao nhiêu?", MICROSOFT,
                                  "Dự kiến hơn 25% nhân lực toàn cầu của Microsoft sẽ nghỉ việc."))

    def test_a_question_that_gives_the_number_is_not_asking_for_it(self):
        question = "Số tiền gần 3 tỷ đồng là cho cả dự án hay cho riêng BW Industrial?"
        self.assertIsNone(select_evidence(question, [SON_LA]).partial_reply())

    def test_a_quoted_sentence_brings_what_it_refers_to(self):
        reply = select_evidence("Họ chiếm bao nhiêu phần trăm nhân lực toàn cầu của Microsoft?", [MICROSOFT]).extractive_reply()
        self.assertIn("Con số này tương đương hơn 25% nhân lực toàn cầu của hãng.", reply)
        self.assertIn("đang đề xuất với khoảng 60.000 nhân viên", reply)


@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class GroundedAnswerTests(unittest.TestCase):
    def shown(self, question: str, answer: str, articles: list[dict]) -> str:
        from rag.pipeline import GroundedAnswer

        grounded = GroundedAnswer(question, articles)
        return grounded.feed(answer + "\n") + grounded.finish()

    def test_a_bare_verdict_is_never_the_answer(self):
        shown = self.shown("Có phải BW Industrial tài trợ cả 11 công trình không?", "Có. [Nguồn 1]", [SON_LA])
        self.assertNotIn("Có.", shown)
        self.assertIn("> ", shown)  # the article's sentences instead

    def test_citation_comes_from_the_supporting_article(self):
        shown = self.shown("Long An giành bao nhiêu điểm sau 4 trận?",
                           "CLB Long An chỉ giành 1 điểm sau 4 trận [Nguồn 2].", [WEEK, LONG_AN])
        self.assertIn("[Nguồn 1]", shown)  # the week article has no matching passage and is left out
        self.assertNotIn("[Nguồn 2]", shown)

    def test_a_short_answer_comes_with_its_sentence(self):
        shown = self.shown("Tuần lễ Số diễn ra ở đâu?", "Quảng Ninh [Nguồn 1]", [WEEK])
        self.assertIn("Quảng Ninh [Nguồn 1]", shown)
        self.assertIn("> Tuần lễ Số Quốc tế Việt Nam 2026 (VIDW 2026) diễn ra từ ngày 28 đến 30/10 tại Quảng Ninh.", shown)

    def test_compound_question_gets_both_parts(self):
        shown = self.shown("BW Industrial tài trợ bao nhiêu công trình, còn Opella đồng hành bao nhiêu?",
                           "Opella Việt Nam đồng hành 10 công trình [Nguồn 1].", [SON_LA])
        self.assertIn("Opella Việt Nam đồng hành 10 công trình [Nguồn 1].", shown)
        self.assertIn("Công trình còn lại do BW Industrial tài trợ", shown)


@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class PredicateTests(unittest.TestCase):
    """Matching names and numbers, or none at all, is not enough: the statement must be the same."""

    def test_unrelated_statement(self):
        self.assertFalse(supported_by_text("Công ty đã phá sản.", "Doanh thu tăng trong quý này."))

    def test_names_keep_their_order_around_the_relation(self):
        source = "Trang Moon đóng vai Tươi trong phim."
        self.assertFalse(supported_by_text("Tươi đóng vai Trang Moon.", source))
        self.assertTrue(supported_by_text("Trang Moon đóng vai Tươi.", source))
        # Names joined by "và" have no order.
        self.assertTrue(supported_by_text("Opella và BW Industrial bàn giao công trình.",
                                          "BW Industrial và Opella bàn giao công trình."))

    def test_verb_and_status(self):
        source = "Đơn vị đang xem xét bàn giao."
        self.assertFalse(supported_by_text("Đơn vị đã hoàn tất bàn giao.", source))
        self.assertFalse(supported_by_text("Đơn vị đã bàn giao.", source))
        self.assertTrue(supported_by_text("Đơn vị đang xem xét bàn giao.", source))

    def test_invented_acceptance_with_the_offer_paragraph(self):
        question = "Microsoft đề xuất bao nhiêu nhân viên tự nguyện nghỉ việc?"
        for answer in ("Khoảng 60.000 nhân viên đã chấp nhận nghỉ việc tự nguyện.",
                       "Microsoft đề xuất với khoảng 60.000 nhân viên và họ đã đồng ý nghỉ việc."):
            with self.subTest(answer=answer):
                self.assertFalse(supports(question, MICROSOFT, answer))


@unittest.skipUnless(KEYWORD_STATS_PATH.exists(), "run `python -m nlp.keyword_stats` first")
class SummaryCheckTests(unittest.TestCase):
    def test_summary_clauses_against_the_article(self):
        self.assertFalse(supported_by_text(
            "Tiền đạo 11 tuổi Kane ghi 21 bàn thắng tại Ngoại hạng Anh, rồi lần thứ ba liên tiếp giành danh hiệu.",
            KANE["body"],
        ))
        self.assertTrue(supported_by_text(
            "Harry Kane, người từng bị Arsenal loại khỏi học viện, đoạt Quả Bóng Vàng 2026 ở tuổi 33.", KANE["body"],
        ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
