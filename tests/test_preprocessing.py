import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import TweetPreprocessor, create_preprocessor


class TestURLRemoval:
    def setup_method(self):
        self.preprocessor = TweetPreprocessor()

    def test_removes_single_url(self):
        assert "http" not in self.preprocessor.clean_url("Check this out https://example.com").lower()

    def test_removes_multiple_urls(self):
        result = self.preprocessor.clean_url("Visit https://a.com and http://b.com")
        assert "http" not in result.lower()

    def test_removes_tco_urls(self):
        result = self.preprocessor.clean_url("Link https://t.co/abc123")
        assert "t.co" not in result.lower()

    def test_text_without_url_unchanged(self):
        text = "This is a normal tweet"
        assert self.preprocessor.clean_url(text) == text

    def test_empty_string(self):
        assert self.preprocessor.clean_url("") == ""


class TestMentionRemoval:
    def setup_method(self):
        self.preprocessor = TweetPreprocessor()

    def test_removes_single_mention(self):
        result = self.preprocessor.clean_mention("Thanks @username for help")
        assert "@username" not in result

    def test_removes_multiple_mentions(self):
        result = self.preprocessor.clean_mention("Hey @user1 and @user2")
        assert "@" not in result

    def test_text_without_mention_unchanged(self):
        text = "No mentions here"
        assert self.preprocessor.clean_mention(text) == text


class TestHashtagHandling:
    def setup_method(self):
        self.preprocessor = TweetPreprocessor(remove_hashtags=False)

    def test_keeps_hashtag_text(self):
        result = self.preprocessor.clean_hashtag("Tweet with #python")
        assert "python" in result
        assert "#" not in result


class TestEmojiRemoval:
    def setup_method(self):
        self.preprocessor = TweetPreprocessor(remove_emojis=True)

    def test_removes_emojis(self):
        result = self.preprocessor.clean_emoji("Happy 😊 and sad 😢")
        assert "😊" not in result
        assert "😢" not in result

    def test_text_without_emoji_unchanged(self):
        text = "No emojis here"
        result = self.preprocessor.clean_emoji(text)
        assert text in result or result.strip() == text.strip()


class TestSpecialCharRemoval:
    def setup_method(self):
        self.preprocessor = TweetPreprocessor()

    def test_removes_special_chars(self):
        result = self.preprocessor.clean_special_chars("Hello!!! What???")
        assert "!" not in result
        assert "?" not in result

    def test_preserves_alphanumeric(self):
        result = self.preprocessor.clean_special_chars("abc123")
        assert "abc123" in result


class TestLowercase:
    def setup_method(self):
        self.preprocessor = TweetPreprocessor(lowercase=True)

    def test_converts_uppercase(self):
        result = self.preprocessor.preprocess_single("HELLO WORLD")
        assert result == result.lower()

    def test_preserves_already_lowercase(self):
        result = self.preprocessor.preprocess_single("hello world")
        assert result == "hello world"


class TestStopWordRemoval:
    def setup_method(self):
        self.preprocessor = TweetPreprocessor(remove_stopwords=True)

    def test_removes_stopwords(self):
        result = self.preprocessor.preprocess_single("the quick brown fox jumps over the lazy dog")
        words = result.split()
        assert "the" not in words
        assert "over" not in words


class TestFullPipeline:
    def setup_method(self):
        self.preprocessor = TweetPreprocessor()

    def test_full_pipeline_positive(self, sample_tweets):
        result = self.preprocessor.preprocess_single(sample_tweets[0])
        assert result == result.lower()
        assert "@" not in result

    def test_full_pipeline_empty(self):
        result = self.preprocessor.preprocess_single("")
        assert result == ""

    def test_preserves_sentiment_words(self):
        result = self.preprocessor.preprocess_single("I love this amazing wonderful product")
        assert any(word in result for word in ["love", "amazing", "wonderful"])

    def test_handles_retweet(self):
        result = self.preprocessor.preprocess_single("RT @user: This is a retweet")
        assert "@user" not in result


class TestPreprocessingEdgeCases:
    def setup_method(self):
        self.preprocessor = TweetPreprocessor()

    def test_only_urls(self):
        result = self.preprocessor.preprocess_single("https://t.co/abc123 https://example.com")
        assert result.strip() == ""

    def test_only_mentions(self):
        result = self.preprocessor.preprocess_single("@user1 @user2 @user3")
        assert result.strip() == ""

    def test_very_long_text(self):
        text = "word " * 1000
        result = self.preprocessor.preprocess_single(text)
        assert isinstance(result, str)

    def test_batch_preprocessing(self, sample_tweets):
        results = self.preprocessor.preprocess_batch(sample_tweets, show_progress=False)
        assert len(results) == len(sample_tweets)
        assert all(isinstance(r, str) for r in results)

    def test_preprocessing_stats(self):
        preprocessor = TweetPreprocessor()
        original = ["This is a great product!", "Terrible service"]
        processed = preprocessor.preprocess_batch(original, show_progress=False)
        stats = preprocessor.get_preprocessing_stats(original, processed)
        assert "total_documents" in stats
        assert stats["total_documents"] == 2

    def test_create_preprocessor_factory(self):
        preprocessor = create_preprocessor({"lowercase": False, "remove_urls": False})
        assert preprocessor.lowercase is False
        assert preprocessor.remove_urls is False