import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix
from src.utils.logger import app_logger
from src.utils.exceptions import FeatureExtractionError
from src.config.settings import MODEL_CONFIG
import joblib
from pathlib import Path


class FeatureExtractor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or MODEL_CONFIG["tfidf"].copy()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.count_vectorizer: Optional[CountVectorizer] = None
        self.is_fitted = False

    def fit_transform_tfidf(
        self, texts: List[str], y: Optional[np.ndarray] = None
    ) -> csr_matrix:
        try:
            self.vectorizer = TfidfVectorizer(**self.config)
            X = self.vectorizer.fit_transform(texts)
            self.is_fitted = True
            app_logger.info(f"TF-IDF matrix shape: {X.shape}")
            return X
        except Exception as e:
            raise FeatureExtractionError(f"TF-IDF fitting failed: {e}")

    def transform_tfidf(self, texts: List[str]) -> csr_matrix:
        if self.vectorizer is None:
            raise FeatureExtractionError("Vectorizer not fitted. Call fit_transform_tfidf first.")
        return self.vectorizer.transform(texts)

    def get_top_features_per_class(
        self, X: csr_matrix, y: np.ndarray, n: int = 20
    ) -> Dict[str, List[Tuple[str, float]]]:
        if self.vectorizer is None:
            raise FeatureExtractionError("Vectorizer not fitted")

        feature_names = self.vectorizer.get_feature_names_out()
        classes = np.unique(y)
        result = {}

        for cls in classes:
            mask = y == cls
            class_vectors = X[mask]
            mean_scores = np.array(class_vectors.mean(axis=0)).flatten()
            top_indices = mean_scores.argsort()[::-1][:n]
            result[str(cls)] = [(feature_names[i], mean_scores[i]) for i in top_indices]

        return result

    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        if self.vectorizer is None:
            raise FeatureExtractionError("Vectorizer not fitted")

        feature_names = self.vectorizer.get_feature_names_out()
        weights = self.vectorizer.idf_
        top_indices = np.argsort(weights)[::-1][:n]
        return [(feature_names[i], weights[i]) for i in top_indices]

    def get_word_frequency(
        self, texts: List[str], top_n: int = 100
    ) -> pd.DataFrame:
        self.count_vectorizer = CountVectorizer(max_features=top_n)
        freq_matrix = self.count_vectorizer.fit_transform(texts)

        word_freq = np.array(freq_matrix.sum(axis=0)).flatten()
        feature_names = self.count_vectorizer.get_feature_names_out()

        return pd.DataFrame(
            {"word": feature_names, "frequency": word_freq}
        ).sort_values("frequency", ascending=False)

    def get_ngrams(
        self, texts: List[str], n: int = 2, top_n: int = 20
    ) -> pd.DataFrame:
        ngram_vectorizer = CountVectorizer(ngram_range=(n, n), max_features=top_n)
        ngram_matrix = ngram_vectorizer.fit_transform(texts)

        ngram_freq = np.array(ngram_matrix.sum(axis=0)).flatten()
        ngram_names = ngram_vectorizer.get_feature_names_out()

        return pd.DataFrame(
            {"ngram": ngram_names, "frequency": ngram_freq}
        ).sort_values("frequency", ascending=False)

    def save_vectorizer(self, path: str) -> None:
        if self.vectorizer is None:
            raise FeatureExtractionError("No vectorizer to save")
        joblib.dump(self.vectorizer, path)
        app_logger.info(f"Vectorizer saved to {path}")

    def load_vectorizer(self, path: str) -> None:
        self.vectorizer = joblib.load(path)
        self.is_fitted = True
        app_logger.info(f"Vectorizer loaded from {path}")


def extract_all_features(
    texts: List[str], config: Optional[Dict] = None
) -> Tuple[csr_matrix, FeatureExtractor]:
    extractor = FeatureExtractor(config)
    X = extractor.fit_transform_tfidf(texts)
    return X, extractor