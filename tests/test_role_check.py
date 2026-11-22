"""Tests for the check that an answer does not swap who left and who arrived."""

from __future__ import annotations

import unittest

from processing.role_check import JOIN, LEAVE, role_claims, role_conflicts, role_evidence

# Shortened from the stored Long An article of the audit ("Incoming versus outgoing coach").
LONG_AN = (
    "CLB Long An đã thay HLV Trịnh Văn Hậu sau thành tích không tốt của đội qua 4 vòng đấu đầu tiên. "
    "HLV Huỳnh Ngọc San ở buổi ra mắt CLB Long An chiều 26-10. "
    "Cụ thể, HLV Huỳnh Ngọc San đã được chọn và đã chính thức bắt tay vào công việc vào chiều 26-10. "
    "Tuy nhiên, tân HLV của CLB Long An chỉ trực tiếp cầm quân kể từ vòng 6. "
    "Trợ lý Phan Văn Giàu sẽ chỉ đạo CLB Long An ở trận tiếp Thái Nguyên trên sân Tân An. "
    "Đội bóng xứ chè còn có chân sút Victor Mansaray đang dẫn đầu danh sách Vua phá lưới với 8 bàn thắng."
)


class RoleClaimTests(unittest.TestCase):
    def test_roles_stated_by_the_article(self):
        roles = {(" ".join(claim.name), claim.role) for claim in role_claims(LONG_AN)}
        self.assertEqual(roles, {("trinh van hau", LEAVE), ("huynh ngoc san", JOIN)})

    def test_object_subject_passive_and_instead(self):
        def roles(text):
            return [(" ".join(c.name), c.role) for c in role_claims(text)]

        self.assertEqual(roles("Ông Đinh Chí Hiếu được bổ nhiệm thay thế ông Jun Sungbae."),
                         [("dinh chi hieu", JOIN), ("jun sungbae", LEAVE)])
        self.assertEqual(roles("Long An thay HLV Trịnh Văn Hậu bằng HLV Huỳnh Ngọc San."),
                         [("trinh van hau", LEAVE), ("huynh ngoc san", JOIN)])
        self.assertEqual(roles("Kane rời Tottenham năm 2023."), [("kane", LEAVE)])
        # A capitalized word inside a sentence is part of a name, not a verb ("sân Tân An").
        self.assertEqual(roles("Trận đấu trên sân Tân An."), [])


class RoleConflictTests(unittest.TestCase):
    def test_swapped_roles_are_caught(self):  # the audit answer
        answer = "CLB Long An chia tay HLV Huỳnh Ngọc San và bổ nhiệm Victor Mansaray thay thế [Nguồn 5]."
        self.assertEqual(set(role_conflicts(answer, LONG_AN)), {"huynh ngoc san", "victor mansaray"})
        swapped = "HLV Huỳnh Ngọc San rời ghế, còn Trịnh Văn Hậu được bổ nhiệm."
        self.assertEqual(set(role_conflicts(swapped, LONG_AN)), {"huynh ngoc san", "trinh van hau"})

    def test_correct_wordings_pass(self):
        for answer in (
            "CLB Long An chia tay HLV Trịnh Văn Hậu và bổ nhiệm HLV Huỳnh Ngọc San thay thế [Nguồn 1].",
            "HLV Trịnh Văn Hậu rời ghế, HLV San nhận ghế.",
            "Victor Mansaray dẫn đầu danh sách Vua phá lưới với 8 bàn.",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(role_conflicts(answer, LONG_AN), [])

    def test_evidence_sentences(self):
        evidence = role_evidence(LONG_AN)
        self.assertTrue(evidence[0].startswith("CLB Long An đã thay HLV Trịnh Văn Hậu"))
        self.assertTrue(any("Huỳnh Ngọc San đã được chọn" in sentence for sentence in evidence))


if __name__ == "__main__":
    unittest.main(verbosity=2)
