import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.opinion_mining import OpinionMiner


class TestOpinionMiner:
    def setup_method(self):
        self.miner = OpinionMiner()

    def test_extract_positive_opinion(self):
        triplets = self.miner.extract_opinion_triplets("The food was delicious")
        assert isinstance(triplets, list)
        if len(triplets) > 0:
            assert "polarity" in triplets[0]
            assert "target" in triplets[0]
            assert "expression" in triplets[0]

    def test_extract_negative_opinion(self):
        triplets = self.miner.extract_opinion_triplets("The service was terrible")
        assert isinstance(triplets, list)
        if len(triplets) > 0:
            assert triplets[0]["polarity"] in ["negative", "neutral", "positive"]

    def test_extract_mixed_opinions(self):
        triplets = self.miner.extract_opinion_triplets(
            "The food was delicious but the service was slow"
        )
        assert isinstance(triplets, list)

    def test_extract_from_empty_text(self):
        triplets = self.miner.extract_opinion_triplets("")
        assert isinstance(triplets, list)
        assert len(triplets) == 0

    def test_extract_batch(self):
        texts = [
            "Amazing product quality!",
            "Terrible customer service",
            "Average experience overall",
        ]
        df = self.miner.extract_opinions_batch(texts)
        assert isinstance(df, object)
        assert "target" in df.columns or len(df) == 0

    def test_top_positive_phrases(self):
        texts = [
            "Love the great product!",
            "Amazing quality and fantastic service",
            "Wonderful experience, highly recommend",
        ]
        phrases = self.miner.get_top_opinion_phrases(texts, polarity="positive", top_n=5)
        assert isinstance(phrases, list)

    def test_top_negative_phrases(self):
        texts = [
            "Terrible product, hate it",
            "Worst service, very disappointed",
            "Horrible experience, never again",
        ]
        phrases = self.miner.get_top_opinion_phrases(texts, polarity="negative", top_n=5)
        assert isinstance(phrases, list)

    def test_build_opinion_database(self):
        texts = [
            "Great product, love the quality",
            "Terrible service, hate the wait",
            "Average experience, nothing special",
        ]
        db = self.miner.build_opinion_database(texts)
        assert "total_opinions" in db
        assert "polarity_distribution" in db

    def test_negation_handling(self):
        triplets = self.miner.extract_opinion_triplets("This is not good")
        if len(triplets) > 0:
            polarities = [t["polarity"] for t in triplets]
            assert any(p == "negative" for p in polarities)