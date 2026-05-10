import sys
import os
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import TweetDataLoader


class TestDataLoader:
    def setup_method(self):
        self.loader = TweetDataLoader()

    def test_load_sample_data(self):
        df = self.loader.load_sample_data()
        assert isinstance(df, pd.DataFrame)
        assert "text" in df.columns
        assert "sentiment" in df.columns
        assert len(df) > 0

    def test_sample_data_sentiment_values(self):
        df = self.loader.load_sample_data()
        valid_sentiments = {"positive", "negative", "neutral"}
        for sentiment in df["sentiment"].unique():
            assert sentiment in valid_sentiments

    def test_sample_data_has_dates(self):
        df = self.loader.load_sample_data()
        assert "date" in df.columns

    def test_validate_dataframe_missing_text(self):
        df = pd.DataFrame({"sentiment": ["positive"]})
        with pytest.raises(Exception):
            self.loader._validate_dataframe(df)

    def test_validate_dataframe_removes_duplicates(self):
        df = pd.DataFrame({
            "text": ["hello", "hello", "world"],
            "sentiment": ["positive", "positive", "negative"],
        })
        result = self.loader._validate_dataframe(df)
        assert len(result) == 2

    def test_save_and_load_processed(self, tmp_path):
        df = pd.DataFrame({
            "text": ["hello", "world"],
            "sentiment": ["positive", "negative"],
        })
        import tempfile
        from src.config.settings import PROCESSED_DATA_DIR

        self.loader.save_processed(df, "test_processed.csv")
        loaded = self.loader.load_processed("test_processed.csv")
        assert len(loaded) == 2

        import os
        os.remove(PROCESSED_DATA_DIR / "test_processed.csv")