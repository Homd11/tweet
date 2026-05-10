import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import compute_all_metrics, generate_classification_report, generate_confusion_matrix, ErrorAnalyzer
from sklearn.preprocessing import LabelEncoder


class TestComputeMetrics:
    def test_basic_metrics(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 0])
        metrics = compute_all_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics

    def test_perfect_accuracy(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_all_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0


class TestClassificationReport:
    def test_report_generation(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        report = generate_classification_report(y_true, y_pred, target_names=["neg", "neu", "pos"])
        assert isinstance(report, str)


class TestConfusionMatrix:
    def test_confusion_matrix(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 1])
        cm = generate_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)
        assert cm.sum() == 6


class TestErrorAnalyzer:
    @pytest.fixture
    def error_data(self):
        texts = [
            "I love this product",
            "This is terrible",
            "Just okay, nothing special",
            "Amazing experience!",
            "Worst thing ever",
            "Not bad, not great",
            "Absolutely fantastic!",
            "I hate this",
            "Normal regular day",
            "Great quality but slow delivery",
        ]
        y_true = np.array([2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([2, 0, 1, 2, 1, 0, 2, 0, 1, 0])
        return texts, y_true, y_pred

    def test_error_analyzer_init(self, error_data):
        texts, y_true, y_pred = error_data
        analyzer = ErrorAnalyzer(texts, y_true, y_pred)
        assert len(analyzer.misclassified_indices) > 0

    def test_get_misclassified_df(self, error_data):
        texts, y_true, y_pred = error_data
        analyzer = ErrorAnalyzer(texts, y_true, y_pred)
        df = analyzer.get_misclassified_df()
        assert isinstance(df, pd.DataFrame)
        assert "text" in df.columns
        assert "true_label" in df.columns
        assert "predicted_label" in df.columns

    def test_categorize_errors(self, error_data):
        texts, y_true, y_pred = error_data
        analyzer = ErrorAnalyzer(texts, y_true, y_pred)
        categories = analyzer.categorize_errors()
        assert isinstance(categories, dict)
        assert "sarcasm" in categories
        assert "ambiguity" in categories

    def test_error_summary(self, error_data):
        texts, y_true, y_pred = error_data
        analyzer = ErrorAnalyzer(texts, y_true, y_pred)
        summary = analyzer.get_error_summary()
        assert "total_samples" in summary
        assert "total_errors" in summary
        assert "error_rate" in summary

    def test_error_report(self, error_data):
        texts, y_true, y_pred = error_data
        analyzer = ErrorAnalyzer(texts, y_true, y_pred)
        report = analyzer.generate_error_report()
        assert isinstance(report, str)
        assert "ERROR ANALYSIS" in report