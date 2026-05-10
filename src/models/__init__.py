from src.models.baseline import SentimentClassifier, train_all_baselines, compare_models

def get_bert_classifier():
    from src.models.bert_model import BERTClassifier
    return BERTClassifier

def get_sentiment_pipeline():
    from src.models.bert_model import get_sentiment_pipeline as _get_pipeline
    return _get_pipeline