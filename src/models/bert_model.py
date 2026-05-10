import numpy as np
import re
from typing import List, Optional, Dict, Any
from src.utils.logger import app_logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from src.utils.exceptions import ModelTrainingError
from src.utils.helpers import is_arabic_text


def get_device():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()


class BERTClassifier:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        num_labels: int = 3,
        is_arabic: bool = False,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required for BERT. Install with: pip install transformers torch")
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        self.is_arabic = is_arabic
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.label_mapping = {0: "negative", 1: "neutral", 2: "positive"}

    def load_pretrained(self) -> None:
        try:
            app_logger.info(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if DEVICE == "cuda" else -1,
                truncation=True,
                max_length=self.max_length,
            )
            app_logger.info("BERT model loaded successfully")
        except Exception as e:
            raise ModelTrainingError(f"Failed to load BERT model: {e}")

    def load_fine_tuned(self, model_path: str) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if DEVICE == "cuda" else -1,
            )
            app_logger.info(f"Fine-tuned BERT model loaded from {model_path}")
        except Exception as e:
            raise ModelTrainingError(f"Failed to load fine-tuned model: {e}")

    def preprocess_for_arabic(self, texts: List[str]) -> List[str]:
        processed = []
        for text in texts:
            text = re.sub(r"[إأآا]", "ا", text)
            text = re.sub(r"ى", "ي", text)
            text = re.sub(r"ة", "ه", text)
            text = re.sub(r"گ", "ك", text)
            processed.append(text)
        return processed

    LABEL_MAP = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
        "pos": "positive",
        "neg": "negative",
    }

    def _map_label(self, label: str) -> str:
        return self.LABEL_MAP.get(label.lower(), label.lower())

    def predict_single(self, text: str) -> Dict[str, Any]:
        if self.pipeline is None:
            raise ModelTrainingError("Model not loaded. Call load_pretrained() first.")
        if self.is_arabic:
            text = self.preprocess_for_arabic([text])[0]
        result = self.pipeline(text)[0]
        mapped = self._map_label(result["label"])
        return {
            "sentiment": mapped,
            "confidence": float(result["score"]),
            "label": mapped,
            "score": float(result["score"]),
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        if self.pipeline is None:
            raise ModelTrainingError("Model not loaded")
        if self.is_arabic:
            texts = self.preprocess_for_arabic(texts)
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = self.pipeline(batch)
            for result in batch_results:
                mapped = self._map_label(result["label"])
                results.append({
                    "sentiment": mapped,
                    "confidence": float(result["score"]),
                    "label": mapped,
                    "score": float(result["score"]),
                })
        return results

    def predict_with_aspects(self, text: str) -> Dict[str, Any]:
        result = self.predict_single(text)
        words = text.split()
        aspects = []
        sentiment_words = {
            "positive": ["love", "great", "amazing", "excellent", "fantastic", "wonderful", "best", "good", "happy", "awesome"],
            "negative": ["hate", "terrible", "awful", "worst", "bad", "horrible", "disappointing", "angry", "poor", "frustrated"],
            "neutral": ["okay", "fine", "normal", "average", "standard", "regular", "usual", "moderate", "mediocre", "decent"],
        }
        sentiment = result.get("sentiment", "neutral")
        for word in words:
            word_lower = word.lower().strip(".,!?;:")
            for sent, keywords in sentiment_words.items():
                if word_lower in keywords:
                    aspects.append({"word": word, "sentiment": sent})
        result["aspects"] = aspects
        return result

    def get_model_info(self) -> Dict[str, Any]:
        info = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "num_labels": self.num_labels,
            "is_arabic": self.is_arabic,
            "device": DEVICE,
            "pipeline_loaded": self.pipeline is not None,
        }
        if self.model is not None and TORCH_AVAILABLE:
            info["total_params"] = sum(p.numel() for p in self.model.parameters())
            info["trainable_params"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return info


def get_sentiment_pipeline(model_name: str = "bert-base-uncased", is_arabic: bool = False):
    if is_arabic:
        model_name = "aubmindlab/bert-base-arabertv02"
    classifier = BERTClassifier(model_name=model_name, is_arabic=is_arabic)
    classifier.load_pretrained()
    return classifier