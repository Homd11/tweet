import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.trend_analysis import TrendAnalyzer


class TestTrendAnalyzer:
    def setup_method(self):
        self.analyzer = TrendAnalyzer()

    def test_compute_sentiment_trends(self, temporal_sentiment_data):
        trends = self.analyzer.compute_sentiment_trends(temporal_sentiment_data)
        assert isinstance(trends, pd.DataFrame)
        assert "mean_sentiment" in trends.columns
        assert "positive_count" in trends.columns
        assert "negative_count" in trends.columns

    def test_compute_rolling_average(self, temporal_sentiment_data):
        trends = self.analyzer.compute_sentiment_trends(temporal_sentiment_data)
        rolling = self.analyzer.compute_rolling_average(trends, window=7)
        assert "mean_sentiment_rolling" in rolling.columns

    def test_detect_sentiment_shifts(self, temporal_sentiment_data):
        trends = self.analyzer.compute_sentiment_trends(temporal_sentiment_data)
        anomalies = self.analyzer.detect_sentiment_shifts(trends)
        assert isinstance(anomalies, pd.DataFrame)
        if len(anomalies) > 0:
            assert "shift_direction" in anomalies.columns

    def test_detect_emerging_topics(self, temporal_sentiment_data):
        results = self.analyzer.detect_emerging_topics(temporal_sentiment_data)
        assert isinstance(results, dict)

    def test_generate_trend_report(self, temporal_sentiment_data):
        trends = self.analyzer.compute_sentiment_trends(temporal_sentiment_data)
        report = self.analyzer.generate_trend_report(trends)
        assert isinstance(report, str)
        assert "SENTIMENT TREND REPORT" in report

    def test_empty_data_handling(self):
        df = pd.DataFrame({"date": [], "sentiment": [], "text": []})
        trends = self.analyzer.compute_sentiment_trends(df)
        assert isinstance(trends, pd.DataFrame)

    def test_correlate_with_events(self, temporal_sentiment_data):
        trends = self.analyzer.compute_sentiment_trends(temporal_sentiment_data)
        events = [
            {"name": "Product Launch", "date": "2024-01-10"},
            {"name": "Bug Report", "date": "2024-01-20"},
        ]
        correlations = self.analyzer.correlate_with_events(trends, events)
        assert isinstance(correlations, pd.DataFrame)
        assert "event" in correlations.columns