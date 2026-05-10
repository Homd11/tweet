import sys
import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_tweets():
    return [
        "I absolutely love this product! It's amazing and works perfectly. 😊",
        "This is the worst experience ever. Totally disappointed and angry! 😡",
        "Just had a great meeting with the team. Looking forward to the project!",
        "Terrible customer service, will never buy again.",
        "Happy to announce our new initiative launching next week! #excited",
        "Frustrated with the bugs. Every time I try to use it, something breaks.",
        "Neutral about this. Not good, not bad, just okay.",
        "Can't believe how awesome the support team was! They solved my issue instantly!",
        "Why is this still broken after 3 updates? Very frustrated 😤",
        "Enjoying my vacation in Hawaii! Beautiful beaches and great weather 🌴",
    ]


@pytest.fixture
def sample_tweets_with_sentiments():
    return pd.DataFrame({
        "text": [
            "I love this! So happy and excited! 🎉",
            "This is terrible. I hate it.",
            "Just a normal day, nothing special.",
            "Amazing product, highly recommend!",
            "Worst purchase of my life.",
            "The weather is okay today.",
            "Fantastic service, very helpful!",
            "Disappointing. Not what I expected.",
            "Received my order, looks fine.",
            "Love the new features! Great improvement!",
        ],
        "sentiment": [
            "positive", "negative", "neutral", "positive",
            "negative", "neutral", "positive", "negative",
            "neutral", "positive",
        ],
    })


@pytest.fixture
def edge_case_tweets():
    return [
        "",
        "   ",
        "http://example.com",
        "@username",
        "#hashtag",
        "🎉🔥💯✨",
        "abc123",
        "A" * 1000,
        "RT @user: this is a retweet",
        "Check this out: https://t.co/abc123 #topic @user",
    ]


@pytest.fixture
def balanced_dataset():
    return pd.DataFrame({
        "text": [
            "I love this product! It's amazing!",
            "This is so good, highly recommend!",
            "Great experience, will buy again!",
            "Fantastic! Exactly what I needed!",
            "Wonderful service, very helpful team!",
            "This is terrible. Worst purchase ever.",
            "I hate this, completely disappointed.",
            "Awful experience, never again!",
            "Horrible quality, total waste.",
            "Terrible service, very rude staff.",
            "The weather is okay today.",
            "Nothing special, just normal.",
            "It's fine, nothing to complain about.",
            "Average experience, as expected.",
            "Standard product, does the job.",
        ],
        "sentiment": [
            "positive", "positive", "positive", "positive", "positive",
            "negative", "negative", "negative", "negative", "negative",
            "neutral", "neutral", "neutral", "neutral", "neutral",
        ],
    })


@pytest.fixture
def temporal_sentiment_data():
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
    scores = 0.5 + 0.3 * np.sin(np.linspace(0, 3 * np.pi, 31)) + np.random.normal(0, 0.1, 31)
    return pd.DataFrame({
        "date": dates,
        "sentiment": ["positive" if s > 0.6 else "negative" if s < 0.4 else "neutral" for s in scores],
        "text": [f"Tweet about day {i}" for i in range(31)],
        "sentiment_score": scores,
    })


@pytest.fixture
def test_config():
    return {
        "max_features": 1000,
        "ngram_range": (1, 2),
        "min_df": 1,
        "max_df": 0.95,
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 3,
    }


@pytest.fixture
def sentiment_labels():
    return ["negative", "neutral", "positive"]


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "unit: marks unit tests")