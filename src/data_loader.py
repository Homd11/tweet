import pandas as pd
from pathlib import Path
from typing import Optional
from src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR
from src.utils.logger import app_logger
from src.utils.exceptions import DataLoadingError


class TweetDataLoader:
    SUPPORTED_FORMATS = [".csv", ".tsv", ".json", ".parquet"]

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, file_path: Path, encoding: str = "utf-8", **kwargs) -> pd.DataFrame:
        try:
            app_logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            app_logger.info(f"Loaded {len(df)} records")
            return self._validate_dataframe(df)
        except Exception as e:
            raise DataLoadingError(f"Failed to load CSV: {e}")

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if "text" not in df.columns:
            if "tweet" in df.columns:
                df = df.rename(columns={"tweet": "text"})
            elif "content" in df.columns:
                df = df.rename(columns={"content": "text"})
            else:
                raise DataLoadingError('Missing required column: "text"')

        if "sentiment" not in df.columns:
            if "label" in df.columns:
                df = df.rename(columns={"label": "sentiment"})
            elif "polarity" in df.columns:
                df = df.rename(columns={"polarity": "sentiment"})

        initial_len = len(df)
        df = df.drop_duplicates(subset=["text"])
        if len(df) < initial_len:
            app_logger.warning(f"Removed {initial_len - len(df)} duplicate records")

        df = df.dropna(subset=["text"])
        if "sentiment" in df.columns:
            df = df.dropna(subset=["sentiment"])

        return df

    def load_sentiment140(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        columns = ["sentiment", "id", "date", "query", "user", "text"]

        if file_path is None:
            file_path = RAW_DATA_DIR / "sentiment140.csv"

        if not file_path.exists():
            raise DataLoadingError(
                f"Sentiment140 file not found at {file_path}. "
                "Download from https://www.kaggle.com/datasets/kazanova/sentiment140"
            )

        df = pd.read_csv(file_path, encoding="latin-1", header=None, names=columns)
        df["sentiment"] = df["sentiment"].map({0: "negative", 2: "neutral", 4: "positive"})
        df = df.dropna(subset=["sentiment"])
        return self._validate_dataframe(df[["text", "sentiment", "date", "user"]])

    def load_sample_data(self) -> pd.DataFrame:
        sample_data = {
            "text": [
                "I absolutely love this movie! The plot was amazing!",
                "Terrible product, broke after one day. Very disappointed.",
                "Just had lunch, the weather is nice today.",
                "Can't believe how bad the service was at this restaurant.",
                "This is the best phone I've ever used!",
                "Not sure how I feel about the new update...",
                "The movie was okay, nothing special really.",
                "Outstanding customer service! They resolved my issue immediately.",
                "Worst purchase ever. Complete waste of money.",
                "Average experience, would neither recommend nor avoid.",
                "Incredible quality and fast shipping! Highly recommend!",
                "I'm so frustrated with this app. It keeps crashing!",
                "The concert last night was absolutely phenomenal!",
                "Meh, the food was edible but nothing to write home about.",
                "This laptop exceeds all my expectations. Great value!",
                "Disappointed with the durability. Broke in two weeks.",
                "Regular day at work, nothing exciting happened.",
                "Love the new design! Clean and modern interface.",
                "The wait times are ridiculous. Been on hold for an hour.",
                "Pretty decent product overall. Does what it says.",
                "BRILLIANT documentary! A must-watch for everyone!",
                "The software update bricked my device. Furious!",
                "Had a standard coffee this morning. It was coffee.",
                "What an amazing weekend getaway! Best trip ever!",
                "Customer support was useless. Didn't help at all.",
                "The book was a decent read. Some parts were slow.",
                "ABSOLUTELY TERRIBLE! Stay away from this company!",
                "Satisfactory quality for the price. No complaints.",
                "This game is so fun! Can't stop playing!",
                "Delivery was late and the package was damaged.",
                "Meeting went as expected. Productive but uneventful.",
                "The restaurant ambiance was lovely and food delicious!",
                "Another overpriced product that doesn't deliver.",
                "It's fine. Just fine. Nothing more, nothing less.",
                "Phenomenal performance by the entire cast tonight!",
                "The laptop screen flickers constantly. Very annoying.",
                "Normal commute today. Traffic was the usual.",
                "Best birthday ever! So grateful for my friends!",
                "This product failed right after the warranty expired.",
                "The weather is typical for this time of year.",
            ],
            "sentiment": [
                "positive", "negative", "neutral", "negative", "positive",
                "neutral", "neutral", "positive", "negative", "neutral",
                "positive", "negative", "positive", "neutral", "positive",
                "negative", "neutral", "positive", "negative", "neutral",
                "positive", "negative", "neutral", "positive", "negative",
                "neutral", "negative", "positive", "positive", "negative",
                "neutral", "positive", "negative", "neutral", "positive",
                "negative", "neutral", "positive", "negative", "neutral",
            ],
            "date": pd.date_range("2024-01-01", periods=40, freq="6H"),
        }
        return pd.DataFrame(sample_data)

    def load_twitter_samples(self) -> pd.DataFrame:
        try:
            import nltk
            from nltk.corpus import twitter_samples

            nltk.data.find("corpora/twitter_samples")
        except (LookupError, ImportError):
            app_logger.warning("NLTK twitter_samples not available. Using sample data.")
            return self.load_sample_data()

        positive_tweets = twitter_samples.strings("positive_tweets.json")
        negative_tweets = twitter_samples.strings("negative_tweets.json")

        df = pd.DataFrame(
            {
                "text": positive_tweets + negative_tweets,
                "sentiment": ["positive"] * len(positive_tweets)
                + ["negative"] * len(negative_tweets),
            }
        )

        return self._validate_dataframe(df)

    def save_processed(self, df: pd.DataFrame, filename: str = "processed_data.csv") -> Path:
        output_path = PROCESSED_DATA_DIR / filename
        df.to_csv(output_path, index=False)
        app_logger.info(f"Processed data saved to {output_path}")
        return output_path

    def load_processed(self, filename: str = "processed_data.csv") -> pd.DataFrame:
        file_path = PROCESSED_DATA_DIR / filename
        if not file_path.exists():
            raise DataLoadingError(f"Processed file not found: {file_path}")
        return pd.read_csv(file_path)