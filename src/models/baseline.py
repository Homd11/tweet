import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from scipy.sparse import csr_matrix
import joblib
from pathlib import Path
from src.utils.logger import app_logger
from src.utils.exceptions import ModelTrainingError


class SentimentClassifier:
    SUPPORTED_MODELS = {
        "logistic_regression": LogisticRegression,
        "multinomial_nb": MultinomialNB,
        "complement_nb": ComplementNB,
        "linear_svm": LinearSVC,
        "random_forest": RandomForestClassifier,
    }

    def __init__(self, model_type: str = "logistic_regression", **kwargs):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_type}. Choose from {list(self.SUPPORTED_MODELS.keys())}")

        self.model_type = model_type
        self.config = kwargs
        self.model = self._create_model()
        self.is_fitted = False
        self.label_mapping = None
        self.classes_ = None

    def _create_model(self) -> Any:
        model_class = self.SUPPORTED_MODELS[self.model_type]

        defaults = {
            "logistic_regression": {"max_iter": 1000, "class_weight": "balanced", "random_state": 42},
            "multinomial_nb": {"alpha": 1.0},
            "complement_nb": {"alpha": 1.0},
            "linear_svm": {"max_iter": 2000, "class_weight": "balanced", "random_state": 42},
            "random_forest": {"n_estimators": 100, "class_weight": "balanced", "n_jobs": -1, "random_state": 42},
        }

        final_config = {**defaults.get(self.model_type, {}), **self.config}

        if self.model_type == "linear_svm":
            return model_class(**final_config)

        return model_class(**final_config)

    def fit(
        self,
        X: csr_matrix,
        y: np.ndarray,
        cv_folds: int = 5,
        use_calibration: bool = True,
    ) -> Dict[str, Any]:
        try:
            app_logger.info(f"Training {self.model_type}...")

            if use_calibration and self.model_type == "linear_svm":
                self.model = CalibratedClassifierCV(self.model, cv=3)

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring="f1_macro")

            self.model.fit(X, y)
            self.is_fitted = True
            self.classes_ = self.model.classes_

            metrics = {
                "cv_f1_mean": float(cv_scores.mean()),
                "cv_f1_std": float(cv_scores.std()),
                "model_type": self.model_type,
            }

            app_logger.info(f"Training complete. CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            return metrics

        except Exception as e:
            raise ModelTrainingError(f"Training failed: {e}")

    def predict(self, X: csr_matrix) -> np.ndarray:
        if not self.is_fitted:
            raise ModelTrainingError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: csr_matrix) -> np.ndarray:
        if not self.is_fitted:
            raise ModelTrainingError("Model not fitted. Call fit() first.")

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(X)
            if len(decision.shape) == 1:
                decision = np.column_stack([-decision, decision])
            exp_decision = np.exp(decision - decision.max(axis=1, keepdims=True))
            return exp_decision / exp_decision.sum(axis=1, keepdims=True)
        else:
            raise ModelTrainingError("Model does not support probability prediction")

    def evaluate(self, X: csr_matrix, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        }

        try:
            y_proba = self.predict_proba(X)
            classes = np.unique(y)
            if len(classes) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
                )
        except Exception:
            metrics["roc_auc"] = None

        return metrics

    def get_classification_report(self, X: csr_matrix, y: np.ndarray) -> str:
        y_pred = self.predict(X)
        return classification_report(y, y_pred, zero_division=0)

    def get_confusion_matrix(self, X: csr_matrix, y: np.ndarray) -> np.ndarray:
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)

    def get_misclassified(self, X: csr_matrix, y: np.ndarray, texts: Optional[list] = None) -> pd.DataFrame:
        y_pred = self.predict(X)
        misclassified_mask = y_pred != y

        result = pd.DataFrame({
            "true_label": y[misclassified_mask],
            "predicted_label": y_pred[misclassified_mask],
        })

        if texts is not None:
            result["text"] = np.array(texts)[misclassified_mask]

        return result.reset_index(drop=True)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        app_logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
        self.is_fitted = True
        self.classes_ = self.model.classes_ if hasattr(self.model, "classes_") else None
        app_logger.info(f"Model loaded from {path}")


def train_all_baselines(
    X_train: csr_matrix,
    y_train: np.ndarray,
    X_val: csr_matrix,
    y_val: np.ndarray,
    model_types: Optional[list] = None,
) -> Tuple[Dict[str, SentimentClassifier], Dict[str, Dict]]:
    models = {}
    results = {}

    if model_types is None:
        model_types = [
            "logistic_regression",
            "multinomial_nb",
            "complement_nb",
            "linear_svm",
            "random_forest",
        ]

    for model_type in model_types:
        try:
            app_logger.info(f"Training {model_type}...")
            clf = SentimentClassifier(model_type)
            train_metrics = clf.fit(X_train, y_train)
            val_metrics = clf.evaluate(X_val, y_val)

            models[model_type] = clf
            results[model_type] = {**train_metrics, **val_metrics}

            app_logger.info(f"{model_type}: Val F1={val_metrics['f1_macro']:.4f}")

        except Exception as e:
            app_logger.error(f"Failed to train {model_type}: {e}")

    return models, results


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    comparison_data = []

    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update({k: v for k, v in metrics.items() if v is not None})
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    if "f1_macro" in df.columns:
        df = df.sort_values("f1_macro", ascending=False)
    return df