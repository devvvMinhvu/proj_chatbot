"""Unit tests for the topic_search intent.

Run from the project root:
    python -m unittest discover -s tests -p "test_topic_search.py" -v
"""

import sys
import types
import unittest
from datetime import datetime
from unittest.mock import patch

# Mock costly project dependencies before importing the handler.
database_pkg = types.ModuleType("database")
database_pkg.__path__ = []
database_module = types.ModuleType("database.database_15k")
database_module.SessionLocal = lambda: None
database_module.Article = type("Article", (), {})
sys.modules["database"] = database_pkg
sys.modules["database.database_15k"] = database_module

rag_pkg = types.ModuleType("rag")
rag_pkg.__path__ = []
rag_module = types.ModuleType("rag.pipeline_15k")
rag_module.ask = lambda question: {"answer": "mock answer", "sources": []}
rag_module.retrieve = lambda question, **kwargs: []
rag_module.group_chunks_by_article = lambda chunks: []
sys.modules["rag"] = rag_pkg
sys.modules["rag.pipeline_15k"] = rag_module

from nlp import intent_handler as handler  # noqa: E402


class TestTopicSearch(unittest.TestCase):
    def setUp(self):
        self.articles = [
            {
                "article_id": "article-1",
                "title": "Thị trường kinh tế có nhiều biến động",
                "source": "VnExpress",
                "url": "https://example.com/article-1",
                "category": "economy",
                "published_at": "03/11/2026 08:00",
                "similarity": None,
            }
        ]

    def test_extract_topic_with_vietnamese_accents(self):
        self.assertEqual(
            handler._extract_topic("Liệt kê bài báo thuộc chuyên mục Kinh tế"),
            "Kinh tế",
        )

    def test_extract_topic_with_english_category(self):
        self.assertEqual(
            handler._extract_topic("Show me articles in the tech category"),
            "Công nghệ",
        )

    def test_ambiguous_multiple_topics_are_not_guessed(self):
        self.assertIsNone(
            handler._extract_topic(
                "Liệt kê bài báo thuộc chuyên mục kinh tế và công nghệ"
            )
        )

    def test_format_topic_results(self):
        answer = handler._format_topic_search(
            "Kinh tế", self.articles, 1, True, ["economy", "tech"]
        )
        self.assertIn("Bài báo thuộc chuyên mục: Kinh tế", answer)
        self.assertIn("Thị trường kinh tế có nhiều biến động", answer)
        self.assertIn("https://example.com/article-1", answer)

    def test_unknown_category_reports_available_values(self):
        answer = handler._format_topic_search(
            "Kinh tế", [], 0, False, ["news", "tech"]
        )
        self.assertIn("chưa tìm thấy chuyên mục", answer.lower())
        self.assertIn("news", answer)
        self.assertIn("tech", answer)

    def test_topic_search_uses_database_path_without_rag(self):
        with patch.object(
            handler, "predict_intent", return_value="topic_search"
        ), patch.object(
            handler,
            "_search_topic_articles",
            return_value=(self.articles, 1, True, ["economy"]),
        ) as mock_search, patch.object(handler, "ask") as mock_ask:
            answer, sources = handler.handle_intent(
                "Liệt kê bài báo thuộc chuyên mục kinh tế"
            )

        mock_search.assert_called_once_with("Kinh tế", limit=10)
        mock_ask.assert_not_called()
        self.assertIn("Kinh tế", answer)
        self.assertEqual(sources, self.articles)

    def test_topic_search_without_clear_topic_does_not_guess(self):
        with patch.object(
            handler, "predict_intent", return_value="topic_search"
        ), patch.object(handler, "_search_topic_articles") as mock_search, patch.object(
            handler, "ask"
        ) as mock_ask:
            answer, sources = handler.handle_intent(
                "Liệt kê bài báo thuộc nhiều chuyên mục"
            )

        self.assertIn("nêu rõ một chuyên mục", answer)
        self.assertEqual(sources, [])
        mock_search.assert_not_called()
        mock_ask.assert_not_called()


    def test_explicit_category_cue_overrides_ambiguous_model_prediction(self):
        with patch.object(
            handler, "predict_intent", return_value="latest_news"
        ), patch.object(
            handler,
            "_search_topic_articles",
            return_value=(self.articles, 1, True, ["economy"]),
        ) as mock_search, patch.object(handler, "ask") as mock_ask:
            answer, sources = handler.handle_intent(
                "Liệt kê bài báo thuộc chuyên mục kinh tế"
            )

        mock_search.assert_called_once_with("Kinh tế", limit=10)
        mock_ask.assert_not_called()
        self.assertIn("Bài báo thuộc chuyên mục", answer)
        self.assertEqual(sources, self.articles)


class TestLatestTopicSearch(unittest.TestCase):
    def test_extract_terms_for_gold_price_query(self):
        self.assertEqual(
            handler._extract_latest_topic_terms(
                "Tìm bài báo về giá vàng mới nhất"
            ),
            ["giá", "vàng"],
        )

    def test_extract_terms_for_national_assembly_query(self):
        self.assertEqual(
            handler._extract_latest_topic_terms(
                "Tìm bài báo về Quốc hội mới nhất"
            ),
            ["quốc", "hội"],
        )

    def test_latest_cue_is_detected(self):
        self.assertTrue(handler._has_latest_cue("Tin mới nhất về giá vàng"))
        self.assertTrue(handler._has_latest_cue("Bài báo gần đây về Quốc hội"))
        self.assertFalse(handler._has_latest_cue("Tìm bài báo về giá vàng"))

    def test_format_latest_topic_results_and_dates(self):
        articles = [
            {
                "title": "Giá vàng hôm nay tăng",
                "source": "VnExpress",
                "url": "https://example.com/gold",
                "published_at_display": "03/11/2026 09:00",
            }
        ]
        answer = handler._format_latest_topic_search(
            ["giá", "vàng"], articles, 4, 3
        )
        self.assertIn("Bài báo mới nhất về: giá vàng", answer)
        self.assertIn("03/11/2026 09:00", answer)
        self.assertIn("1 bài khớp từ khóa chưa có ngày đăng", answer)
        self.assertIn("https://example.com/gold", answer)

    def test_latest_topic_search_overrides_classifier_and_skips_rag(self):
        articles = [
            {
                "article_id": "gold-1",
                "title": "Giá vàng hôm nay tăng",
                "source": "VnExpress",
                "url": "https://example.com/gold",
                "published_at_display": "03/11/2026 09:00",
                "similarity": None,
            }
        ]
        with patch.object(
            handler, "predict_intent", return_value="semantic_search"
        ), patch.object(
            handler,
            "_search_latest_topic_articles",
            return_value=(articles, 2, 2),
        ) as mock_search, patch.object(handler, "ask") as mock_ask:
            answer, sources = handler.handle_intent(
                "Tìm bài báo về giá vàng mới nhất"
            )

        mock_search.assert_called_once_with(["giá", "vàng"], limit=10)
        mock_ask.assert_not_called()
        self.assertIn("giá vàng", answer)
        self.assertEqual(sources, articles)


    def test_national_assembly_latest_query_is_routed_correctly(self):
        with patch.object(
            handler, "predict_intent", return_value="latest_news"
        ), patch.object(
            handler,
            "_search_latest_topic_articles",
            return_value=([], 0, 0),
        ) as mock_search:
            answer, sources = handler.handle_intent(
                "Tìm bài báo về Quốc hội mới nhất"
            )

        mock_search.assert_called_once_with(["quốc", "hội"], limit=10)
        self.assertIn("quốc hội", answer.lower())
        self.assertEqual(sources, [])

    def test_search_without_recency_cue_remains_semantic_search(self):
        with patch.object(
            handler, "predict_intent", return_value="latest_topic_search"
        ), patch.object(
            handler, "_search_sources", return_value=[]
        ) as mock_search, patch.object(handler, "_search_latest_topic_articles") as mock_latest:
            answer, sources = handler.handle_intent("Tìm bài báo về giá vàng")

        mock_search.assert_called_once_with("Tìm bài báo về giá vàng")
        mock_latest.assert_not_called()
        self.assertIn("chưa tìm thấy", answer.lower())
        self.assertEqual(sources, [])

    def test_database_search_matches_all_terms_and_sorts_by_date(self):
        from sqlalchemy import Column, DateTime, String, Text, create_engine
        from sqlalchemy.orm import declarative_base, sessionmaker

        Base = declarative_base()

        class TestArticle(Base):
            __tablename__ = "test_latest_topic_articles"
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
        db.add_all([
            TestArticle(
                id="old", title="Giá vàng tuần trước", sapo="",
                body="Giá vàng biến động", source="Báo A",
                url="https://example.com/old", category="economy",
                published_at=datetime(2026, 10, 1, 8, 0),
            ),
            TestArticle(
                id="new", title="Giá vàng hôm nay", sapo="",
                body="Thị trường vàng tăng", source="Báo B",
                url="https://example.com/new", category="economy",
                published_at=datetime(2026, 11, 3, 9, 0),
            ),
            TestArticle(
                id="unrelated", title="Giá xăng hôm nay", sapo="",
                body="Giá xăng tăng", source="Báo C",
                url="https://example.com/fuel", category="economy",
                published_at=datetime(2026, 11, 3, 10, 0),
            ),
        ])
        db.commit()
        db.close()

        with patch.object(handler, "Article", TestArticle), patch.object(
            handler, "SessionLocal", TestSession
        ):
            articles, total_count, dated_count = handler._search_latest_topic_articles(
                ["giá", "vàng"], limit=10
            )

        self.assertEqual(total_count, 2)
        self.assertEqual(dated_count, 2)
        self.assertEqual(
            [article["article_id"] for article in articles],
            ["new", "old"],
        )
        engine.dispose()


    def test_latest_news_without_topic_remains_general_latest_news(self):
        with patch.object(
            handler, "predict_intent", return_value="latest_topic_search"
        ), patch.object(
            handler,
            "_get_latest_news",
            return_value=([], 0, 0),
        ) as mock_latest_news:
            answer, sources = handler.handle_intent("Tin mới nhất")

        mock_latest_news.assert_called_once_with(limit=10)
        self.assertIn("chưa thể tạo danh sách tin mới nhất", answer.lower())
        self.assertEqual(sources, [])



if __name__ == "__main__":
    unittest.main(verbosity=2)
