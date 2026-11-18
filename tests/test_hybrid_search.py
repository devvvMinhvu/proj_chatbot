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


if __name__ == "__main__":
    unittest.main(verbosity=2)
