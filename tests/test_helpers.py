from src.utils.helpers import sanitize_text, is_arabic_text, extract_hashtags, extract_mentions


class TestSanitizeText:
    def test_removes_mentions(self):
        result = sanitize_text("Hello @user check this out")
        assert "@user" not in result
        assert "[USER]" in result

    def test_removes_urls(self):
        result = sanitize_text("Visit https://example.com for more info")
        assert "https://example.com" not in result
        assert "[URL]" in result

    def test_preserves_regular_text(self):
        text = "Hello world this is normal text"
        assert sanitize_text(text) == text


class TestIsArabicText:
    def test_detects_arabic(self):
        assert is_arabic_text("مرحبا بالعالم")

    def test_english_not_arabic(self):
        assert not is_arabic_text("Hello world")

    def test_mixed_text_with_arabic(self):
        assert is_arabic_text("Hello مرحبا")


class TestExtractHashtags:
    def test_extracts_single(self):
        assert "python" in extract_hashtags("Check #python out")

    def test_extracts_multiple(self):
        result = extract_hashtags("#python #testing #pytest")
        assert len(result) == 3

    def test_no_hashtags(self):
        assert extract_hashtags("No hashtags here") == []


class TestExtractMentions:
    def test_extracts_single(self):
        assert "user" in extract_mentions("Hello @user")

    def test_extracts_multiple(self):
        result = extract_mentions("@user1 @user2")
        assert len(result) == 2

    def test_no_mentions(self):
        assert extract_mentions("No mentions here") == []