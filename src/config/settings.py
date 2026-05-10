import os
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample_data"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODELS_DIR = MODELS_DIR / "baseline"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR,
                 MODELS_DIR, BASELINE_MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

API_KEYS: Dict[str, Any] = {
    "twitter_api_key": os.getenv("TWITTER_API_KEY"),
    "twitter_api_secret": os.getenv("TWITTER_API_SECRET"),
    "huggingface_token": os.getenv("HF_TOKEN"),
}

MODEL_CONFIG = {
    "tfidf": {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
    },
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
    },
    "multinomial_nb": {
        "alpha": 1.0,
    },
    "complement_nb": {
        "alpha": 1.0,
    },
    "linear_svm": {
        "C": 1.0,
        "max_iter": 1000,
        "class_weight": "balanced",
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "bert": {
        "model_name": "bert-base-uncased",
        "max_length": 128,
        "batch_size": 32,
    },
    "arabert": {
        "model_name": "aubmindlab/bert-base-arabertv02",
        "max_length": 128,
        "batch_size": 32,
    },
}

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}

LOG_CONFIG = {
    "rotation": "500 MB",
    "retention": "30 days",
    "level": "INFO",
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
}

DASHBOARD_CONFIG = {
    "title": "Tweet Sentiment Analyzer",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}