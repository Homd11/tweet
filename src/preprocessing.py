import re
import nltk
import pandas as pd
from typing import List, Optional, Dict, Tuple
from src.utils.logger import app_logger
from src.utils.exceptions import PreprocessingError
from src.config.settings import SENTIMENT_LABELS

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TweetPreprocessor:
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        remove_emojis: bool = True,
        remove_special_chars: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        stem: bool = False,
        lemmatize: bool = True,
        min_word_length: int = 2,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_emojis = remove_emojis
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length

        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_url(self, text: str) -> str:
        return re.sub(r"http\S+|www\.\S+|https?\S+", "", text)

    def clean_mention(self, text: str) -> str:
        return re.sub(r"@\w+", "", text)

    def clean_hashtag(self, text: str) -> str:
        return re.sub(r"#(\w+)", r"\1", text)

    def clean_emoji(self, text: str) -> str:
        try:
            import emoji as emoji_lib
            text = emoji_lib.demojize(text, delimiters=(" ", " "))
            text = re.sub(r":\w+:", "", text)
        except ImportError:
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"
                "\U0001F300-\U0001F5FF"
                "\U0001F680-\U0001F6FF"
                "\U0001F1E0-\U0001F1FF"
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+",
                flags=re.UNICODE,
            )
            text = emoji_pattern.sub("", text)
        return text

    def clean_special_chars(self, text: str) -> str:
        text = re.sub(r"RT\s+", "", text)
        text = re.sub(r"&\w+;", "", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        return text

    def clean_numbers(self, text: str) -> str:
        return re.sub(r"\d+", "", text)

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        return [w for w in tokens if w.lower() not in self.stop_words]

    def apply_stemming(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(w) for w in tokens]

    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(w) for w in tokens]

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def preprocess_single(self, text: str) -> str:
        try:
            if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
                return ""

            if self.remove_urls:
                text = self.clean_url(text)
            if self.remove_mentions:
                text = self.clean_mention(text)
            if not self.remove_hashtags:
                text = self.clean_hashtag(text)
            elif self.remove_hashtags:
                text = re.sub(r"#\w+", "", text)
            if self.remove_emojis:
                text = self.clean_emoji(text)
            if self.remove_special_chars:
                text = self.clean_special_chars(text)
            if self.remove_numbers:
                text = self.clean_numbers(text)

            if self.lowercase:
                text = text.lower()

            text = re.sub(r"\s+", " ", text).strip()

            tokens = self.tokenize(text)

            filters = [lambda w: len(w) >= self.min_word_length]
            if self.remove_stopwords:
                filters.append(lambda w: w.lower() not in self.stop_words)

            for f in filters:
                tokens = [w for w in tokens if f(w)]

            if self.stem and not self.lemmatize:
                tokens = self.apply_stemming(tokens)
            elif self.lemmatize and not self.stem:
                tokens = self.apply_lemmatization(tokens)

            return " ".join(tokens)

        except Exception as e:
            app_logger.error(f"Preprocessing failed: {e}")
            return ""

    def preprocess_batch(
        self, texts: List[str], show_progress: bool = True
    ) -> List[str]:
        results = []
        iterator = enumerate(texts)

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(enumerate(texts), total=len(texts), desc="Preprocessing")
            except ImportError:
                pass

        for idx, text in iterator:
            results.append(self.preprocess_single(text))

        return results

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        df = df.copy()
        df["original_text"] = df[text_column]
        df["cleaned_text"] = self.preprocess_batch(df[text_column].tolist())
        df["text_length_before"] = df["original_text"].str.len()
        df["text_length_after"] = df["cleaned_text"].str.len()
        df["word_count_before"] = df["original_text"].str.split().str.len()
        df["word_count_after"] = df["cleaned_text"].str.split().str.len()
        return df

    def get_preprocessing_stats(
        self, original_texts: List[str], processed_texts: List[str]
    ) -> Dict:
        original_lens = [len(t.split()) for t in original_texts if isinstance(t, str)]
        processed_lens = [len(t.split()) for t in processed_texts if isinstance(t, str)]

        return {
            "total_documents": len(original_texts),
            "avg_original_length": sum(original_lens) / len(original_lens) if original_lens else 0,
            "avg_processed_length": sum(processed_lens) / len(processed_lens) if processed_lens else 0,
            "total_words_removed": sum(original_lens) - sum(processed_lens),
            "empty_documents": sum(1 for t in processed_texts if not t),
            "reduction_ratio": 1 - (sum(processed_lens) / sum(original_lens)) if sum(original_lens) > 0 else 0,
        }


def create_preprocessor(config: Optional[Dict] = None) -> TweetPreprocessor:
    if config is None:
        config = {}
    return TweetPreprocessor(**config)