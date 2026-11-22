"""Tests for topic classification input and model selection (no model loading)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nlp import text_classifier as tc


class PhoBERTInputTests(unittest.TestCase):
    def test_repeated_sapo_paragraph_is_removed_and_words_segmented(self):
        sapo = "Giá vàng trong nước tăng mạnh trong phiên sáng nay."
        body = f"{sapo}\n\nNgân hàng Nhà nước tiếp tục theo dõi thị trường vàng."
        text = tc.phobert_input_text("Giá vàng tăng", sapo, body)
        self.assertEqual(text.count("phiên sáng"), 1)
        self.assertIn("Ngân_hàng", text)
        self.assertTrue(text.startswith("Giá vàng tăng"))

    def test_empty_parts_are_skipped(self):
        self.assertEqual(tc.phobert_input_text("Tiêu đề", "", ""), "Tiêu_đề")


class _FakePipeline:
    named_steps = {"classifier": type("C", (), {"classes_": ["economy", "sports"]})()}

    def predict(self, texts):
        return ["sports"]

    def predict_proba(self, texts):
        return [[0.2, 0.8]]


class BackendSelectionTests(unittest.TestCase):
    def test_auto_falls_back_to_tfidf_without_phobert_model(self):
        with tempfile.TemporaryDirectory() as empty, patch.object(
            tc, "PHOBERT_TOPIC_MODEL_DIR", Path(empty)
        ), patch.object(tc, "load_model", return_value=_FakePipeline()):
            result = tc.predict_category("Arsenal thắng", "", "")
        self.assertEqual((result["category"], result["model"]), ("sports", "tfidf"))

    def test_auto_uses_phobert_when_its_model_exists(self):
        with tempfile.TemporaryDirectory() as model_dir:
            (Path(model_dir) / "config.json").write_text("{}", encoding="utf-8")
            with patch.object(tc, "PHOBERT_TOPIC_MODEL_DIR", Path(model_dir)), patch.object(
                tc, "_predict_phobert", return_value={"category": "economy", "model": "phobert"}
            ) as fake:
                result = tc.predict_category("Giá vàng tăng", "", "")
        fake.assert_called_once()
        self.assertEqual(result["model"], "phobert")

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            tc.predict_category("Tiêu đề", backend="bert")


if __name__ == "__main__":
    unittest.main(verbosity=2)
