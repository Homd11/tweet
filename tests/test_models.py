import sys
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baseline import SentimentClassifier, train_all_baselines, compare_models
from src.feature_extraction import FeatureExtractor


@pytest.fixture
def trained_model(balanced_dataset):
    texts = balanced_dataset["text"].tolist()
    labels = balanced_dataset["sentiment"].tolist()

    extractor = FeatureExtractor(config={"max_features": 1000, "min_df": 1})
    X = extractor.fit_transform_tfidf(texts)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = SentimentClassifier("logistic_regression")
    clf.fit(X, y, cv_folds=3, use_calibration=False)

    return {"model": clf, "extractor": extractor, "label_encoder": le, "X": X, "y": y}


class TestSentimentClassifier:
    def test_init_logistic_regression(self):
        clf = SentimentClassifier("logistic_regression")
        assert clf.model_type == "logistic_regression"

    def test_init_multinomial_nb(self):
        clf = SentimentClassifier("multinomial_nb")
        assert clf.model_type == "multinomial_nb"

    def test_init_complement_nb(self):
        clf = SentimentClassifier("complement_nb")
        assert clf.model_type == "complement_nb"

    def test_init_linear_svm(self):
        clf = SentimentClassifier("linear_svm")
        assert clf.model_type == "linear_svm"

    def test_init_random_forest(self):
        clf = SentimentClassifier("random_forest")
        assert clf.model_type == "random_forest"

    def test_invalid_model_raises_error(self):
        with pytest.raises(ValueError):
            SentimentClassifier("invalid_model")


class TestModelTraining:
    def test_fit_logistic_regression(self, balanced_dataset):
        texts = balanced_dataset["text"].tolist()
        labels = balanced_dataset["sentiment"].tolist()

        extractor = FeatureExtractor(config={"max_features": 500, "min_df": 1})
        X = extractor.fit_transform_tfidf(texts)

        le = LabelEncoder()
        y = le.fit_transform(labels)

        clf = SentimentClassifier("logistic_regression")
        metrics = clf.fit(X, y, cv_folds=3, use_calibration=False)

        assert "cv_f1_mean" in metrics
        assert "model_type" in metrics
        assert clf.is_fitted

    def test_fit_all_models(self, balanced_dataset):
        texts = balanced_dataset["text"].tolist()
        labels = balanced_dataset["sentiment"].tolist()

        extractor = FeatureExtractor(config={"max_features": 500, "min_df": 1})
        X = extractor.fit_transform_tfidf(texts)

        le = LabelEncoder()
        y = le.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        for model_type in ["logistic_regression", "multinomial_nb", "complement_nb"]:
            clf = SentimentClassifier(model_type)
            clf.fit(X_train, y_train, cv_folds=3, use_calibration=False)
            assert clf.is_fitted


class TestModelPrediction:
    def test_predict(self, trained_model):
        model = trained_model["model"]
        X = trained_model["X"]
        predictions = model.predict(X)
        assert len(predictions) == X.shape[0]

    def test_predict_proba(self, trained_model):
        model = trained_model["model"]
        X = trained_model["X"]
        probabilities = model.predict_proba(X)
        assert probabilities.shape[0] == X.shape[0]
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_predict_before_fit_raises_error(self):
        clf = SentimentClassifier("logistic_regression")
        with pytest.raises(Exception):
            clf.predict(np.array([[]]))


class TestModelEvaluation:
    def test_evaluate(self, trained_model):
        model = trained_model["model"]
        X = trained_model["X"]
        y = trained_model["y"]

        metrics = model.evaluate(X, y)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics

    def test_classification_report(self, trained_model):
        model = trained_model["model"]
        X = trained_model["X"]
        y = trained_model["y"]

        report = model.get_classification_report(X, y)
        assert isinstance(report, str)

    def test_confusion_matrix(self, trained_model):
        model = trained_model["model"]
        X = trained_model["X"]
        y = trained_model["y"]

        cm = model.get_confusion_matrix(X, y)
        assert cm.shape == (3, 3)

    def test_misclassified(self, trained_model):
        model = trained_model["model"]
        X = trained_model["X"]
        y = trained_model["y"]

        misclassified = model.get_misclassified(X, y)
        assert isinstance(misclassified, pd.DataFrame)
        assert "true_label" in misclassified.columns
        assert "predicted_label" in misclassified.columns


class TestModelSaveLoad:
    def test_save_and_load(self, trained_model, tmp_path):
        model = trained_model["model"]
        path = str(tmp_path / "test_model.joblib")

        model.save(path)
        assert os.path.exists(path)

        new_model = SentimentClassifier("logistic_regression")
        new_model.load(path)
        assert new_model.is_fitted


class TestCompareModels:
    def test_compare_models(self, balanced_dataset):
        from src.data_loader import TweetDataLoader
        loader = TweetDataLoader()
        df = loader.load_sample_data()
        texts = df["text"].tolist() + balanced_dataset["text"].tolist()
        labels = df["sentiment"].tolist() + balanced_dataset["sentiment"].tolist()

        extractor = FeatureExtractor(config={"max_features": 500, "min_df": 1})
        X = extractor.fit_transform_tfidf(texts)

        le = LabelEncoder()
        y = le.fit_transform(labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        classifier = SentimentClassifier("logistic_regression")
        classifier.fit(X_train, y_train, cv_folds=3, use_calibration=False)
        metrics = classifier.evaluate(X_test, y_test)

        results = {"logistic_regression": metrics}
        comparison = compare_models(results)
        assert isinstance(comparison, pd.DataFrame)
        assert "model" in comparison.columns
        assert "model" in comparison.columns