import sys
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import TweetDataLoader
from src.preprocessing import TweetPreprocessor
from src.feature_extraction import FeatureExtractor
from src.models.baseline import SentimentClassifier, train_all_baselines, compare_models
from src.evaluation import compute_all_metrics, generate_classification_report, ErrorAnalyzer
from src.opinion_mining import OpinionMiner
from src.trend_analysis import TrendAnalyzer


@pytest.fixture
def full_pipeline_data():
    loader = TweetDataLoader()
    df = loader.load_sample_data()

    preprocessor = TweetPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df)

    df_processed = df_processed[df_processed["cleaned_text"].str.len() > 0]

    feature_ext = FeatureExtractor(config={"max_features": 1000, "min_df": 1})
    X = feature_ext.fit_transform_tfidf(df_processed["cleaned_text"].tolist())

    le = LabelEncoder()
    y = le.fit_transform(df_processed["sentiment"].tolist())

    return {
        "df": df,
        "df_processed": df_processed,
        "X": X,
        "y": y,
        "le": le,
        "feature_ext": feature_ext,
        "preprocessor": preprocessor,
    }


class TestEndToEndPipeline:
    def test_data_loading_to_cleaning(self, full_pipeline_data):
        df = full_pipeline_data["df"]
        df_processed = full_pipeline_data["df_processed"]

        assert len(df) > 0
        assert "cleaned_text" in df_processed.columns
        assert "original_text" in df_processed.columns

    def test_feature_extraction_to_model(self, full_pipeline_data):
        X = full_pipeline_data["X"]
        y = full_pipeline_data["y"]

        assert X.shape[0] == len(y)
        assert X.shape[1] > 0

    def test_model_training_and_evaluation(self, full_pipeline_data):
        X = full_pipeline_data["X"]
        y = full_pipeline_data["y"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = SentimentClassifier("logistic_regression")
        clf.fit(X_train, y_train, cv_folds=3, use_calibration=False)

        y_pred = clf.predict(X_test)
        assert len(y_pred) == len(y_test)

        metrics = clf.evaluate(X_test, y_test)
        assert metrics["accuracy"] > 0
        assert metrics["f1_macro"] > 0

    def test_full_classification_report(self, full_pipeline_data):
        X = full_pipeline_data["X"]
        y = full_pipeline_data["y"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = SentimentClassifier("logistic_regression")
        clf.fit(X_train, y_train, cv_folds=3, use_calibration=False)

        report = clf.get_classification_report(X_test, y_test)
        assert isinstance(report, str)

    def test_error_analysis_pipeline(self, full_pipeline_data):
        X = full_pipeline_data["X"]
        y = full_pipeline_data["y"]
        texts = full_pipeline_data["df_processed"]["original_text"].tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = SentimentClassifier("logistic_regression")
        clf.fit(X_train, y_train, cv_folds=3, use_calibration=False)

        y_pred = clf.predict(X_test)
        _, X_test_aligned, _, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        texts_test = list(np.array(texts)[np.isin(np.arange(len(texts)), np.arange(len(y)))])

        analyzer = ErrorAnalyzer(texts[:len(y_pred)], y[:len(y_pred)], y_pred)
        misclassified = analyzer.get_misclassified_df()
        assert isinstance(misclassified, pd.DataFrame)

    def test_opinion_mining_integration(self, full_pipeline_data):
        texts = full_pipeline_data["df"]["text"].tolist()
        miner = OpinionMiner()
        opinions = miner.extract_opinions_batch(texts)
        assert isinstance(opinions, pd.DataFrame)

    def test_trend_analysis_integration(self, full_pipeline_data):
        df = full_pipeline_data["df"]
        if "date" in df.columns:
            analyzer = TrendAnalyzer()
            trends = analyzer.compute_sentiment_trends(df)
            assert isinstance(trends, pd.DataFrame)

    def test_all_baseline_models(self, full_pipeline_data):
        X = full_pipeline_data["X"]
        y = full_pipeline_data["y"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        models_results = train_all_baselines(
            X_train, y_train, X_test, y_test,
            model_types=["logistic_regression", "multinomial_nb", "complement_nb"],
        )

        models, results = models_results
        assert len(models) >= 2
        assert len(results) >= 2

        comparison = compare_models(results)
        assert isinstance(comparison, pd.DataFrame)
        assert "model" in comparison.columns
        assert "f1_macro" in comparison.columns