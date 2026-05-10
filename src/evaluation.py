import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from src.utils.logger import app_logger


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    if y_proba is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: Optional[List] = None) -> str:
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List] = None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


class ErrorAnalyzer:
    def __init__(self, texts: List[str], y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None):
        self.texts = texts
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_proba = y_proba

        misclassified = y_pred != y_true
        self.misclassified_indices = np.where(misclassified)[0]
        self.misclassified_texts = [texts[i] for i in self.misclassified_indices if i < len(texts)]
        self.misclassified_true = y_true[misclassified]
        self.misclassified_pred = y_pred[misclassified]

    def get_misclassified_df(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "text": self.misclassified_texts,
            "true_label": self.misclassified_true,
            "predicted_label": self.misclassified_pred,
        })

        if self.y_proba is not None:
            misclassified_proba = self.y_proba[self.misclassified_indices]
            df["max_confidence"] = misclassified_proba.max(axis=1)
            df["true_confidence"] = [
                misclassified_proba[i, self.y_proba.shape[1] - 1] if self.y_proba.shape[1] > 2
                else misclassified_proba[i, 1]
                for i in range(len(misclassified_proba))
            ]

        return df

    def categorize_errors(self) -> Dict[str, List[Dict]]:
        categories = {
            "sarcasm": [],
            "ambiguity": [],
            "domain_specific": [],
            "short_text": [],
            "negation": [],
            "low_confidence": [],
            "other": [],
        }

        sarcasm_indicators = [
            "oh great", "just what i needed", "wonderful", "fantastic",
            "yeah right", "sure", "love how",
        ]
        negation_words = ["not", "no", "never", "neither", "nobody", "nothing", "nowhere", "nor"]

        for i, text in enumerate(self.misclassified_texts):
            text_lower = text.lower()
            error_entry = {
                "text": text,
                "true_label": str(self.misclassified_true[i]),
                "predicted_label": str(self.misclassified_pred[i]),
            }

            if any(indicator in text_lower for indicator in sarcasm_indicators):
                categories["sarcasm"].append(error_entry)
            elif any(neg in text_lower.split() for neg in negation_words):
                categories["negation"].append(error_entry)
            elif len(text.split()) <= 5:
                categories["short_text"].append(error_entry)
            elif self.y_proba is not None:
                confidence = self.y_proba[self.misclassified_indices[i]].max()
                if confidence < 0.6:
                    categories["low_confidence"].append(error_entry)
                else:
                    categories["ambiguity"].append(error_entry)
            else:
                categories["other"].append(error_entry)

        return categories

    def get_error_summary(self) -> Dict:
        total = len(self.y_true)
        errors = len(self.misclassified_indices)
        error_rate = errors / total if total > 0 else 0

        confusion_pairs = Counter(
            zip(self.misclassified_true, self.misclassified_pred)
        )
        top_confusions = confusion_pairs.most_common(10)

        categories = self.categorize_errors()
        category_counts = {k: len(v) for k, v in categories.items()}

        return {
            "total_samples": total,
            "total_errors": errors,
            "error_rate": error_rate,
            "top_confusion_pairs": [(f"{true}->{pred}", count) for (true, pred), count in top_confusions],
            "error_categories": category_counts,
        }

    def generate_error_report(self, n_examples: int = 30) -> str:
        categories = self.categorize_errors()
        summary = self.get_error_summary()

        report_lines = [
            "=" * 60,
            "ERROR ANALYSIS REPORT",
            "=" * 60,
            f"Total Samples: {summary['total_samples']}",
            f"Misclassified: {summary['total_errors']} ({summary['error_rate']:.2%})",
            "",
            "TOP CONFUSION PAIRS:",
        ]

        for pair, count in summary["top_confusion_pairs"]:
            report_lines.append(f"  {pair}: {count} times")

        report_lines.append("")
        report_lines.append("ERROR CATEGORIES:")

        for category, count in summary["error_categories"].items():
            if count > 0:
                report_lines.append(f"  {category}: {count}")
                for example in categories[category][:n_examples]:
                    report_lines.append(f"    - True: {example['true_label']}, Pred: {example['predicted_label']}")
                    report_lines.append(f"      \"{example['text'][:80]}...\"")

        return "\n".join(report_lines)