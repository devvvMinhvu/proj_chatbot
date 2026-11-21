"""Unit tests for hybrid retrieval, phrase priority, source filtering, and follow-up context."""

from __future__ import annotations

from datetime import datetime
import unittest
from unittest.mock import patch

from sqlalchemy import Column, DateTime, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from database import repository as repo
from nlp import intent_handler as handler
from rag import pipeline as rag


class TestHybridSearchAndEdgeCases(unittest.TestCase):
    def test_single_entity_keyword_extraction_and_title_overlap(self):
        self.assertEqual(rag.get_keyword_list("Tìm một bài nhắc đến Mbappe"), ["mbappe"])
        self.assertEqual(rag.get_keyword_list("Find one mentioning Mbappe"), ["mbappe"])

        item = {
            "title": "Tổng biên tập France Football: 'Mbappe đã có lúc dẫn đầu cuộc đua'",
            "sapo": "",
            "text": "Mbappé từng dẫn đầu cuộc đua Quả Bóng Vàng.",
        }
        self.assertTrue(rag.has_title_overlap("Find one mentioning Mbappe", item))
        self.assertTrue(rag.has_lexical_overlap("Tìm bài nhắc đến Mbappé", item))

    def test_out_of_dataset_entity_returns_empty_without_faiss_hallucination(self):
        fake_metadata = [
            {
                "article_id": "art-1",
                "chunk_id": 0,
                "title": "Hệ thống thanh toán quốc tế Swift cập nhật mới",
                "sapo": "Ngân hàng áp dụng chuẩn mới",
                "source": "vnexpress",
                "category": "economy",
                "url": "https://vnexpress.net/swift-1.html",
                "text": "Giao dịch qua Swift đạt kỷ lục trong quý này.",
            }
        ]
        with patch.object(rag, "search_fts_articles", return_value=[]), patch.object(
            rag, "search_vectors", return_value=([0.75], [0], fake_metadata)
        ):
            results = rag.retrieve("Tìm bài nhắc đến Taylor Swift", candidate_k=10, top_k=5)

        self.assertEqual(results, [])

    def test_hybrid_retrieve_recovers_single_entity_missed_by_faiss(self):
        fts_hits = [
            {
                "article_id": "vne_sports_0001",
                "title": "Tổng biên tập France Football: 'Mbappe đã có lúc dẫn đầu cuộc đua'",
                "sapo": "Phát biểu về cuộc đua Quả Bóng Vàng.",
                "body": "Đoạn 1 chung chung.\n\nMbappe có cơ hội lớn ở mùa giải này.",
                "source": "vnexpress",
                "category": "sports",
                "url": "https://vnexpress.net/mbappe.html",
                "published_at": "2026-10-27 11:26:00",
                "published_at_display": "27/10/2026 11:26",
                "similarity": 0.91,
            }
        ]
        with patch.object(rag, "search_fts_articles", return_value=fts_hits), patch.object(
            rag, "search_vectors", return_value=([], [], [])
        ):
            results = rag.retrieve("Find one mentioning Mbappe", candidate_k=10, top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["article_id"], "vne_sports_0001")
        self.assertIn("Mbappe", results[0]["text"])

    def test_contiguous_phrase_beats_scattered_syllables_in_latest_topic_search(self):
        Base = declarative_base()

        class TestArticle(Base):
            __tablename__ = "test_phrase_articles"
            id = Column(String, primary_key=True)
            title = Column(Text)
            sapo = Column(Text)
            body = Column(Text)
            source = Column(String)
            url = Column(Text)
            category = Column(String)
            published_at = Column(DateTime)

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        TestSession = sessionmaker(bind=engine)
        db = TestSession()
        db.add_all(
            [
                TestArticle(
                    id="mbappe_false_hit",
                    title="Đánh giá Quả Bóng Vàng của Mbappe",
                    sapo="",
                    body="Chuyên gia đánh giá cơ hội giành Quả Bóng Vàng.",
                    source="vnexpress",
                    url="https://example.com/mbappe",
                    category="sports",
                    published_at=datetime(2026, 11, 5, 11, 26),
                ),
                TestArticle(
                    id="gold_true_hit",
                    title="Giá vàng hôm nay tăng mạnh",
                    sapo="Giá vàng SJC tăng 2 triệu đồng.",
                    body="Thị trường vàng trong nước biến động.",
                    source="tienphong",
                    url="https://example.com/gold",
                    category="economy",
                    published_at=datetime(2026, 11, 1, 9, 0),
                ),
            ]
        )
        db.commit()
        db.close()

        articles, total_count, dated_count = repo.search_latest_topic_articles(
            ["giá", "vàng"],
            limit=10,
            session_factory=TestSession,
            article_model=TestArticle,
        )
        engine.dispose()

        self.assertEqual(total_count, 1)
        self.assertEqual(dated_count, 1)
        self.assertEqual([item["article_id"] for item in articles], ["gold_true_hit"])

    def test_source_extraction_and_mention_routing(self):
        self.assertEqual(
            handler._extract_source("Tìm bài trên báo VnExpress về giá vàng"),
            "vnexpress",
        )
        self.assertEqual(
            handler._extract_source("Tin thể thao của Tuổi Trẻ"),
            "tuoitre",
        )

        mock_sources = [
            {
                "article_id": "vne_sports_0001",
                "title": "Mbappe dẫn đầu cuộc đua",
                "source": "vnexpress",
                "url": "https://vnexpress.net/mbappe.html",
                "similarity": 0.95,
            }
        ]
        with patch.object(handler, "predict_intent", return_value="question_answering"), patch.object(
            handler, "_search_sources", return_value=mock_sources
        ) as mock_search, patch.object(handler, "ask") as mock_ask:
            answer, sources = handler.handle_intent("Find one mentioning Mbappe")

        mock_search.assert_called_once_with("Find one mentioning Mbappe")
        mock_ask.assert_not_called()
        self.assertIn("Mbappe dẫn đầu cuộc đua", answer)
        self.assertEqual(sources, mock_sources)

    def test_followup_ordinal_summarization_uses_previous_sources(self):
        context_sources = [
            {
                "article_id": "art-1",
                "title": "Bài thứ nhất",
                "source": "vnexpress",
                "url": "https://vnexpress.net/1.html",
            },
            {
                "article_id": "art-2",
                "title": "Bài thứ hai",
                "source": "tuoitre",
                "url": "https://tuoitre.vn/2.html",
            },
        ]
        fake_content = {
            "article_id": "art-2",
            "title": "Bài thứ hai",
            "sapo": "Tóm tắt sapo",
            "body": "Nội dung đầy đủ của bài báo thứ hai.",
            "source": "tuoitre",
            "category": "economy",
            "published_at_display": "27/10/2026 09:00",
        }
        with patch.object(handler, "get_article_content", return_value=fake_content) as mock_get, patch.object(
            handler, "_load_summarizer", return_value=lambda body: "Bản tóm tắt ngắn gọn."
        ), patch.object(handler, "_search_sources") as mock_search:
            answer, sources = handler.handle_intent(
                "Tóm tắt bài số 2",
                context_sources=context_sources,
            )

        mock_get.assert_called_once_with(article_id="art-2", url="https://tuoitre.vn/2.html")
        mock_search.assert_not_called()
        self.assertIn("Bản tóm tắt ngắn gọn.", answer)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["article_id"], "art-2")

    def test_bare_followup_summarization_uses_active_or_first_source(self):
        context_sources = [
            {
                "article_id": "art-1",
                "title": "Bài thứ nhất",
                "source": "vnexpress",
                "url": "https://vnexpress.net/1.html",
            }
        ]
        active_article = {
            "article_id": "art-active",
            "title": "Bài đang xem",
            "source": "tuoitre",
            "url": "https://tuoitre.vn/active.html",
        }
        fake_content = {
            "article_id": "art-active",
            "title": "Bài đang xem",
            "sapo": "Sapo bài đang xem",
            "body": "Nội dung đầy đủ bài đang xem.",
            "source": "tuoitre",
        }
        with patch.object(handler, "predict_intent", return_value="summarization"), patch.object(
            handler, "get_article_content", return_value=fake_content
        ) as mock_get, patch.object(
            handler, "_load_summarizer", return_value=lambda body, article_id=None: "Tóm tắt bài đang xem."
        ), patch.object(handler, "_search_sources") as mock_search:
            answer, sources = handler.handle_intent(
                "Tóm tắt giúp tôi",
                context_sources=context_sources,
                active_article=active_article,
            )

        mock_get.assert_called_once_with(article_id="art-active", url="https://tuoitre.vn/active.html")
        mock_search.assert_not_called()
        self.assertIn("Tóm tắt bài đang xem.", answer)
        self.assertEqual(sources[0]["article_id"], "art-active")

    def test_out_of_range_ordinal_returns_bounds_message(self):
        context_sources = [
            {"article_id": "art-1", "title": "Bài 1", "url": "https://example.com/1"},
            {"article_id": "art-2", "title": "Bài 2", "url": "https://example.com/2"},
        ]
        with patch.object(handler, "predict_intent", return_value="summarization"), patch.object(
            handler, "_search_sources"
        ) as mock_search:
            answer, sources = handler.handle_intent(
                "Tóm tắt bài số 9",
                context_sources=context_sources,
            )

        mock_search.assert_not_called()
        self.assertEqual(sources, [])
        self.assertIn("chỉ có 2 bài báo", answer)

    def test_vietnamese_word_ordinal_and_consecutive_drills(self):
        context_sources = [
            {"article_id": "art-1", "title": "Bài 1", "url": "https://example.com/1"},
            {"article_id": "art-2", "title": "Bài 2", "url": "https://example.com/2"},
        ]
        fake_2 = {
            "article_id": "art-2",
            "title": "Bài 2",
            "body": "Toàn văn bài hai.",
            "source": "vnexpress",
        }
        with patch.object(handler, "predict_intent", return_value="summarization"), patch.object(
            handler, "get_article_content", return_value=fake_2
        ) as mock_get, patch.object(
            handler, "_load_summarizer", return_value=lambda body, article_id=None: "Tóm tắt bài hai."
        ):
            answer, sources = handler.handle_intent(
                "Tóm tắt bài thứ hai",
                context_sources=context_sources,
            )

        mock_get.assert_called_once_with(article_id="art-2", url="https://example.com/2")
        self.assertIn("Tóm tắt bài hai.", answer)
        self.assertEqual(sources[0]["article_id"], "art-2")

    def test_implicit_followup_qa_on_active_article(self):
        active_article = {
            "article_id": "art-active",
            "title": "Giá vàng lập đỉnh mới",
            "sapo": "Thị trường vàng biến động mạnh.",
            "source": "vnexpress",
            "url": "https://vnexpress.net/gold.html",
        }
        fake_content = {
            "article_id": "art-active",
            "title": "Giá vàng lập đỉnh mới",
            "sapo": "Thị trường vàng biến động mạnh.",
            "body": "Nguyên nhân tăng giá là do nhu cầu trú ẩn tăng cao.",
            "source": "vnexpress",
        }
        with patch.object(handler, "predict_intent", return_value="question_answering"), patch.object(
            handler, "get_article_content", return_value=fake_content
        ), patch.object(
            handler, "generate_answer", return_value="Do nhu cầu trú ẩn tăng cao [Nguồn 1]."
        ) as mock_gen, patch.object(handler, "ask") as mock_ask:
            answer, sources = handler.handle_intent(
                "Nguyên nhân là gì?",
                active_article=active_article,
            )

        mock_ask.assert_not_called()
        mock_gen.assert_called_once()
        self.assertIn("nhu cầu trú ẩn", answer)
        self.assertEqual(sources[0]["article_id"], "art-active")

    def test_concise_reading_text_and_fast_first_tts_chunk(self):
        from voice.reading_text import build_reading_text
        from voice.tts_adapter import _split_long_text

        long_list = "\n".join(
            [f"## 10 bài báo mới nhất theo ngày đăng đã lưu"]
            + [
                f"{i}. **Tiêu đề bài báo số {i}** — vnexpress\n   Ngày đăng theo dữ liệu lưu: 27/10/2026 0{i % 10}:00\n   https://example.com/{i}"
                for i in range(1, 11)
            ]
        )
        spoken = build_reading_text(long_list, max_list_items=3)
        self.assertIn("Tiêu đề bài báo số 1", spoken)
        self.assertIn("Tiêu đề bài báo số 3", spoken)
        self.assertNotIn("Tiêu đề bài báo số 4", spoken)
        self.assertIn("hiển thị trên màn hình", spoken)

        sample = (
            "Đây là câu đầu tiên ngắn gọn để bắt đầu phát âm thanh nhanh nhất có thể. "
            "Câu thứ hai tiếp tục cung cấp chi tiết đầy đủ hơn về sự kiện chính hôm nay. "
            "Câu thứ ba bổ sung thêm số liệu và bối cảnh liên quan từ bài báo gốc."
        )
        chunks = _split_long_text(sample, max_chars=380, first_chunk_max_chars=160)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertLessEqual(len(chunks[0]), 160)
        self.assertEqual(
            chunks[0],
            "Đây là câu đầu tiên ngắn gọn để bắt đầu phát âm thanh nhanh nhất có thể.",
        )

    def test_summarizer_caches_by_article_id_and_caps_chunks(self):
        from processing import summarizer as summ

        summ._SUMMARY_CACHE.clear()
        long_body = " ".join(["Từ khóa nội dung báo chí."] * 350)
        with patch.object(
            summ,
            "summarize_chunks_batch",
            return_value=["Tóm tắt đoạn 1.", "Tóm tắt đoạn 2."],
        ) as mock_batch:
            first = summ.summarize(long_body, article_id="art-cache-1")
            second = summ.summarize(long_body, article_id="art-cache-1")

        self.assertEqual(first, "Tóm tắt đoạn 1. Tóm tắt đoạn 2.")
        self.assertEqual(second, first)
        mock_batch.assert_called_once()
        passed_chunks = mock_batch.call_args[0][0]
        self.assertLessEqual(len(passed_chunks), summ.MAX_SUMMARY_CHUNKS)

    def test_stream_answer_falls_back_gracefully_when_ollama_fails(self):
        fake_articles = [
            {
                "article_id": "vne-1",
                "title": "Thị trường chứng khoán hồi phục",
                "source": "vnexpress",
                "sapo": "VN-Index tăng 15 điểm trong phiên sáng.",
                "texts": ["Thanh khoản cải thiện rõ rệt."],
            }
        ]
        with patch.object(rag.ollama, "chat", side_effect=ConnectionError("Ollama offline")):
            output = "".join(rag.stream_answer("VN-Index tăng bao nhiêu?", fake_articles))

        self.assertIn("Thị trường chứng khoán hồi phục", output)
        self.assertIn("VN-Index tăng 15 điểm", output)


class TestDenseFilterAndFinanceEvidence(unittest.TestCase):
    SWIFT_PAYMENTS = {
        "article_id": "art-1",
        "chunk_id": 0,
        "title": "Hệ thống thanh toán quốc tế Swift cập nhật mới",
        "sapo": "Ngân hàng áp dụng chuẩn mới",
        "source": "vnexpress",
        "category": "economy",
        "url": "https://vnexpress.net/swift-1.html",
        "text": "Giao dịch qua Swift đạt kỷ lục trong quý này.",
    }

    def test_entity_terms_detects_names_in_any_case(self):
        self.assertEqual(rag.entity_terms("Tìm bài nhắc đến Mbappé"), {"mbappe"})
        self.assertEqual(rag.entity_terms("tìm bài về taylor swift"), {"taylor", "swift"})
        self.assertEqual(rag.entity_terms("Mưa lớn tại Nghệ An"), {"nghe", "an"})
        self.assertEqual(rag.entity_terms("giá xăng tăng mạnh"), set())
        self.assertEqual(rag.entity_terms("Tin kinh tế trên VnExpress"), set())

    def test_lowercase_unknown_entity_still_returns_nothing(self):
        with patch.object(rag, "search_fts_articles", return_value=[]), patch.object(
            rag, "search_vectors", return_value=([0.9], [0], [self.SWIFT_PAYMENTS])
        ):
            results = rag.retrieve("tìm bài về taylor swift", candidate_k=10, top_k=5)

        self.assertEqual(results, [])

    def test_paraphrased_dense_hit_is_kept_but_ranked_below_lexical_match(self):
        paraphrase = {
            "article_id": "para-1",
            "chunk_id": 0,
            "title": "Hộ kinh doanh chật vật khi nhiên liệu đắt đỏ",
            "sapo": "",
            "source": "tuoitre",
            "category": "economy",
            "url": "https://tuoitre.vn/para-1.html",
            "text": "Nhiều hộ kinh doanh phải cắt giảm chi tiêu vì nhiên liệu đắt đỏ.",
        }
        lexical = {
            "article_id": "lex-1",
            "chunk_id": 0,
            "title": "Giá xăng tăng lần thứ ba liên tiếp",
            "sapo": "",
            "source": "vnexpress",
            "category": "economy",
            "url": "https://vnexpress.net/lex-1.html",
            "text": "Người dân lo lắng khi giá xăng tăng.",
        }
        with patch.object(rag, "search_fts_articles", return_value=[]), patch.object(
            rag, "search_vectors", return_value=([0.79, 0.78], [0, 1], [paraphrase, lexical])
        ):
            results = rag.retrieve("người dân lo lắng vì giá xăng tăng", candidate_k=10, top_k=5)

        self.assertEqual([r["article_id"] for r in results], ["lex-1", "para-1"])

    def _expressway_article(self, sentence: str) -> list[dict]:
        return [
            {
                "title": "Cao tốc Bắc Nam được bổ sung ngân sách",
                "texts": [f"Bộ Giao thông vận tải vừa có tờ trình. {sentence}"],
            }
        ]

    def test_financial_evidence_accepts_any_topic(self):
        question = "Ngân sách xây cao tốc Bắc Nam là bao nhiêu?"
        self.assertTrue(rag.asks_for_financial_amount(question))
        articles = self._expressway_article("Ngân sách cho cao tốc Bắc Nam là 120.000 tỷ đồng.")
        self.assertTrue(rag.has_financial_evidence(question, articles))

    def test_financial_evidence_rejects_unrelated_or_non_finance_sentences(self):
        question = "Ngân sách xây cao tốc Bắc Nam là bao nhiêu?"
        no_finance_word = self._expressway_article("Tuyến dài 1.000 km, đã thu 50 tỷ đồng phí.")
        other_topic = self._expressway_article("Ngân sách cho bệnh viện là 20 tỷ đồng.")
        self.assertFalse(rag.has_financial_evidence(question, no_finance_word))
        self.assertFalse(rag.has_financial_evidence(question, other_topic))


class TestQuestionRoutingAndAccentMatching(unittest.TestCase):
    def _routed_intent(self, question: str, classifier_intent: str) -> str:
        seen: list[str] = []
        with patch.object(handler, "predict_intent", return_value=classifier_intent), patch.object(
            handler, "retrieve", return_value=[]
        ), patch("builtins.print", side_effect=lambda *a, **k: seen.append(" ".join(map(str, a)))):
            handler._execute_turn_plan(question)
        line = next(item for item in seen if item.startswith("[INTENT]"))
        return line.split()[1]

    def test_question_words_route_to_question_answering(self):
        cases = {
            "Người dân lo lắng thế nào khi giá xăng tăng?": "latest_topic_search",
            "Vì sao giá vàng tăng mạnh": "semantic_search",
            "Ai là người giành Quả Bóng Vàng 2026?": "semantic_search",
        }
        for question, classifier_intent in cases.items():
            with self.subTest(question=question):
                self.assertEqual(self._routed_intent(question, classifier_intent), "question_answering")

    def test_search_and_browse_requests_are_not_turned_into_questions(self):
        cases = {
            "Tìm bài nhắc đến Mbappe?": ("semantic_search", "semantic_search"),
            "Có nội dung nào về thị trường lao động?": ("semantic_search", "semantic_search"),
            "Chuyên mục thể thao có gì?": ("topic_search", "topic_search"),
            "Bạn là ai?": ("introduction", "introduction"),
        }
        for question, (classifier_intent, expected) in cases.items():
            with self.subTest(question=question):
                self.assertEqual(self._routed_intent(question, classifier_intent), expected)

    def test_new_question_is_not_answered_from_unrelated_active_article(self):
        fuel_article = {
            "article_id": "fuel-1",
            "title": "Châu Âu chạy đua ứng phó khủng hoảng giá xăng dầu",
            "sapo": "Giá xăng dầu tăng vọt, châu Âu mạnh tay giảm thuế.",
            "source": "vnexpress",
            "url": "https://vnexpress.net/fuel-1.html",
        }
        fuel_content = {**fuel_article, "body": "Giá nhiên liệu tăng mạnh khiến người dân lo lắng."}
        with patch.object(handler, "predict_intent", return_value="semantic_search"), patch.object(
            handler, "get_article_content", return_value=fuel_content
        ), patch.object(handler, "_handle_target_article") as mock_target, patch.object(
            handler, "retrieve", return_value=[]
        ):
            handler._execute_turn_plan("Vì sao giá vàng tăng mạnh?", active_article=fuel_article)
            mock_target.assert_not_called()

            handler._execute_turn_plan("Giá xăng dầu tăng bao nhiêu?", active_article=fuel_article)
            mock_target.assert_called_once()

    def test_accented_keywords_need_matching_accents(self):
        self.assertEqual(rag.matched_keywords(["giá", "xăng"], "Món quà cho mẹ ở Gia Lai"), set())
        self.assertEqual(rag.matched_keywords(["tăng"], "Quà tặng cho phụ nữ"), set())
        self.assertEqual(rag.matched_keywords(["giá"], "Giá xăng hôm nay"), {"giá"})
        # Unaccented input and foreign names still match loosely.
        self.assertEqual(rag.matched_keywords(["gia", "xang"], "Giá xăng tăng"), {"gia", "xang"})
        self.assertEqual(rag.matched_keywords(["mbappé"], "Mbappe ghi bàn"), {"mbappé"})

    def test_title_bonus_ignores_accent_collisions(self):
        question = "Người dân lo lắng thế nào khi giá xăng tăng?"
        gia_lai = {"title": "Món quà tháng 10 cho những người mẹ khó khăn miền núi Gia Lai"}
        fuel = {"title": "Xăng tăng gần 5.000 đồng/lít, người dân lo giá hàng hóa"}
        self.assertFalse(rag.has_title_overlap(question, gia_lai))
        self.assertTrue(rag.has_title_overlap(question, fuel))

    def test_fts_hit_matching_only_without_accents_is_demoted(self):
        loose_hit = {
            "article_id": "gift-1",
            "title": "Món quà tháng 10 cho những người mẹ ở Gia Lai",
            "sapo": "",
            "body": "Người dân lo lắng. Giá xăng cao. Quà tặng cho mẹ.",
            "source": "tuoitre",
            "category": "news",
            "url": "https://tuoitre.vn/gift-1.html",
            "similarity": 0.83,
            "fts_tier": 1,
        }
        with patch.object(rag, "search_fts_articles", return_value=[loose_hit]), patch.object(
            rag, "search_vectors", return_value=([], [], [])
        ):
            results = rag.retrieve("người dân lo lắng vì giá xăng tăng", candidate_k=10, top_k=5)

        self.assertEqual(len(results), 1)
        self.assertLessEqual(results[0]["similarity"], rag.FTS_LOOSE_MATCH_SCORE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
