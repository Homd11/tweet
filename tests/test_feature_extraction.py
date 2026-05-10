import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import FeatureExtractor, extract_all_features
from sklearn.preprocessing import LabelEncoder


class TestFeatureExtractor:
    def setup_method(self):
        self.texts = [
            "love great amazing excellent wonderful fantastic",
            "hate terrible awful horrible worst bad",
            "okay fine normal average standard typical",
        ]
        self.extractor = FeatureExtractor(config={"max_features": 100, "ngram_range": (1, 2), "min_df": 1, "max_df": 0.95})

    def test_fit_transform_tfidf(self):
        X = self.extractor.fit_transform_tfidf(self.texts)
        assert X.shape[0] == len(self.texts)
        assert X.shape[1] > 0

    def test_transform_new_data(self):
        self.extractor.fit_transform_tfidf(self.texts)
        new_texts = ["love amazing great"]
        X_new = self.extractor.transform_tfidf(new_texts)
        assert X_new.shape[0] == 1

    def test_transform_without_fit_raises_error(self):
        extractor = FeatureExtractor()
        with pytest.raises(Exception):
            extractor.transform_tfidf(["test"])

    def test_deterministic_output(self):
        extractor1 = FeatureExtractor(config={"max_features": 50, "min_df": 1})
        extractor2 = FeatureExtractor(config={"max_features": 50, "min_df": 1})
        result1 = extractor1.fit_transform_tfidf(self.texts)
        result2 = extractor2.fit_transform_tfidf(self.texts)
        np.testing.assert_array_equal(result1.toarray(), result2.toarray())

    def test_sparse_output(self):
        from scipy.sparse import issparse
        X = self.extractor.fit_transform_tfidf(self.texts)
        assert issparse(X)

    def test_correct_n_samples(self):
        X = self.extractor.fit_transform_tfidf(self.texts)
        assert X.shape[0] == len(self.texts)

    def test_max_features_limit(self):
        texts = ["word" + str(i) for i in range(100)]
        extractor = FeatureExtractor(config={"max_features": 10, "min_df": 1})
        X = extractor.fit_transform_tfidf(texts)
        assert X.shape[1] <= 10

    def test_word_frequency(self):
        texts = ["love love hate hate great", "bad bad terrible wonderful"]
        extractor = FeatureExtractor()
        df = extractor.get_word_frequency(texts, top_n=5)
        assert isinstance(df, pd.DataFrame)
        assert "word" in df.columns
        assert "frequency" in df.columns

    def test_ngrams(self):
        texts = ["love product great", "hate product bad"]
        extractor = FeatureExtractor()
        df = extractor.get_ngrams(texts, n=2, top_n=10)
        assert isinstance(df, pd.DataFrame)
        assert "ngram" in df.columns

    def test_top_features_per_class(self):
        texts = [
            "love great amazing", "love great amazing",
            "hate terrible awful", "hate terrible awful",
            "okay fine normal", "okay fine normal",
        ]
        labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]

        extractor = FeatureExtractor(config={"min_df": 1, "max_features": 100})
        X = extractor.fit_transform_tfidf(texts)
        result = extractor.get_top_features_per_class(X, np.array(labels), n=5)

        assert isinstance(result, dict)
        assert "positive" in result
        assert "negative" in result
        assert "neutral" in result

    def test_save_load_vectorizer(self, tmp_path):
        self.extractor.fit_transform_tfidf(self.texts)
        path = str(tmp_path / "vectorizer.joblib")
        self.extractor.save_vectorizer(path)

        new_extractor = FeatureExtractor()
        new_extractor.load_vectorizer(path)

        X_new = new_extractor.transform_tfidf(["love amazing"])
        assert X_new.shape[0] == 1

    def test_extract_all_features(self):
        X, extractor = extract_all_features(self.texts, config={"max_features": 50, "min_df": 1})
        assert X.shape[0] == len(self.texts)
        assert extractor.is_fitted