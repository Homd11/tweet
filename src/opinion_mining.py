import re
from typing import List, Dict, Tuple, Optional
import pandas as pd
from collections import Counter
from src.utils.logger import app_logger


try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    app_logger.warning("spaCy not available. Opinion mining will use rule-based fallback.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


class OpinionMiner:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = None
        self.use_spacy = SPACY_AVAILABLE

        if self.use_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
                app_logger.info(f"spaCy model '{spacy_model}' loaded successfully")
            except OSError:
                app_logger.warning(f"spaCy model '{spacy_model}' not found. Attempting download...")
                try:
                    spacy.cli.download(spacy_model)
                    self.nlp = spacy.load(spacy_model)
                    app_logger.info(f"spaCy model '{spacy_model}' downloaded and loaded")
                except Exception as e:
                    app_logger.warning(f"Failed to download spaCy model: {e}. Using rule-based fallback.")
                    self.use_spacy = False
                    self.nlp = None

    def extract_opinion_triplets(self, text: str) -> List[Dict[str, str]]:
        triplets = []

        if self.use_spacy and self.nlp is not None:
            doc = self.nlp(text)
            for sent in doc.sents:
                sent_triplets = self._extract_spacy_triplets(sent)
                triplets.extend(sent_triplets)
        else:
            triplets = self._extract_rule_based_triplets(text)

        return triplets

    def _extract_spacy_triplets(self, sent) -> List[Dict[str, str]]:
        triplets = []

        for token in sent:
            if token.dep_ in ("amod", "acomp") and token.head.pos_ in ("NOUN", "PROPN"):
                target = token.head.text
                expression = token.text
                polarity = self._get_polarity(expression)
                triplets.append({
                    "target": target,
                    "expression": expression,
                    "polarity": polarity,
                })

            elif token.dep_ == "nsubj" and token.head.pos_ == "ADJ":
                target = token.text
                expression = token.head.text
                polarity = self._get_polarity(expression)
                triplets.append({
                    "target": target,
                    "expression": expression,
                    "polarity": polarity,
                })

            elif token.dep_ == "dobj" and token.head.pos_ == "VERB":
                subject_tokens = [t for t in token.head.lefts if t.dep_ in ("nsubj", "nsubjpass")]
                if subject_tokens:
                    target = token.text
                    expression = token.head.text
                    polarity = self._get_polarity(expression)
                    triplets.append({
                        "target": target,
                        "expression": expression,
                        "polarity": polarity,
                    })

            elif token.dep_ == "neg":
                governor = token.head
                if governor.pos_ in ("ADJ", "VERB", "ADV"):
                    polarity = self._get_polarity(governor.text)
                    if polarity == "positive":
                        polarity = "negative"
                    elif polarity == "negative":
                        polarity = "positive"

                    target_tokens = [t for t in governor.head.lefts if t.dep_ in ("nsubj", "nsubjpass")]
                    if not target_tokens:
                        target_tokens = [t for t in governor.lefts if t.dep_ == "nsubj"]

                    target = target_tokens[0].text if target_tokens else "unknown"
                    expression = f"not {governor.text}"
                    triplets.append({
                        "target": target,
                        "expression": expression,
                        "polarity": polarity,
                    })

        return triplets

    def _extract_rule_based_triplets(self, text: str) -> List[Dict[str, str]]:
        triplets = []

        positive_words = {
            "love", "great", "amazing", "excellent", "fantastic", "wonderful",
            "best", "good", "happy", "awesome", "beautiful", "perfect", "brilliant",
            "superb", "outstanding", "incredible", "delightful", "enjoy", "like",
        }
        negative_words = {
            "hate", "terrible", "awful", "worst", "bad", "horrible", "disappointing",
            "angry", "poor", "frustrated", "ugly", "broken", "useless", "annoying",
            "pathetic", "disgusting", "mediocre", "unacceptable", "fail", "suck",
        }

        negation_words = {"not", "no", "never", "neither", "nobody", "nothing", "nowhere", "nor", "don't", "doesn't", "didn't"}

        words = text.lower().split()
        words = [w.strip(".,!?;:") for w in words]

        for i, word in enumerate(words):
            has_negation = any(words[j] in negation_words for j in range(max(0, i - 2), i))

            if word in positive_words:
                polarity = "negative" if has_negation else "positive"
                nearby_nouns = self._find_nearby_noun(words, i)
                if nearby_nouns:
                    for noun in nearby_nouns:
                        triplets.append({"target": noun, "expression": word, "polarity": polarity})
                else:
                    triplets.append({"target": "general", "expression": word, "polarity": polarity})

            elif word in negative_words:
                polarity = "positive" if has_negation else "negative"
                nearby_nouns = self._find_nearby_noun(words, i)
                if nearby_nouns:
                    for noun in nearby_nouns:
                        triplets.append({"target": noun, "expression": word, "polarity": polarity})
                else:
                    triplets.append({"target": "general", "expression": word, "polarity": polarity})

        return triplets

    def _find_nearby_noun(self, words: List[str], index: int, window: int = 3) -> List[str]:
        common_nouns = {
            "product", "service", "quality", "price", "experience", "food", "movie",
            "book", "app", "phone", "laptop", "camera", "battery", "screen", "design",
            "support", "delivery", "staff", "room", "hotel", "car", "game", "music",
        }
        nouns = []
        start = max(0, index - window)
        end = min(len(words), index + window + 1)
        for j in range(start, end):
            if j != index and words[j] in common_nouns:
                nouns.append(words[j])
        return nouns

    def _get_polarity(self, word: str) -> str:
        positive_words = {
            "love", "great", "amazing", "excellent", "fantastic", "wonderful",
            "best", "good", "happy", "awesome", "beautiful", "perfect", "brilliant",
            "superb", "outstanding", "incredible", "delightful", "enjoy",
        }
        negative_words = {
            "hate", "terrible", "awful", "worst", "bad", "horrible", "disappointing",
            "angry", "poor", "frustrated", "ugly", "broken", "useless", "annoying",
        }

        if word.lower() in positive_words:
            return "positive"
        elif word.lower() in negative_words:
            return "negative"
        else:
            if TEXTBLOB_AVAILABLE:
                polarity = TextBlob(word).sentiment.polarity
                if polarity > 0.1:
                    return "positive"
                elif polarity < -0.1:
                    return "negative"
            return "neutral"

    def extract_opinions_batch(self, texts: List[str]) -> pd.DataFrame:
        all_triplets = []

        for text in texts:
            triplets = self.extract_opinion_triplets(text)
            for triplet in triplets:
                triplet["text"] = text
                all_triplets.append(triplet)

        if not all_triplets:
            return pd.DataFrame(columns=["text", "target", "expression", "polarity"])

        return pd.DataFrame(all_triplets)

    def build_opinion_database(self, texts: List[str], sentiments: Optional[List[str]] = None) -> Dict:
        df = self.extract_opinions_batch(texts)

        opinion_db = {
            "total_opinions": len(df),
            "unique_targets": df["target"].nunique() if len(df) > 0 else 0,
            "polarity_distribution": df["polarity"].value_counts().to_dict() if len(df) > 0 else {},
            "top_targets": df["target"].value_counts().head(20).to_dict() if len(df) > 0 else {},
            "opinions_by_target": {},
        }

        if len(df) > 0:
            for target in df["target"].unique():
                target_opinions = df[df["target"] == target]
                opinion_db["opinions_by_target"][target] = {
                    "expressions": target_opinions["expression"].tolist(),
                    "polarities": target_opinions["polarity"].tolist(),
                    "positive_count": (target_opinions["polarity"] == "positive").sum(),
                    "negative_count": (target_opinions["polarity"] == "negative").sum(),
                    "neutral_count": (target_opinions["polarity"] == "neutral").sum(),
                }

        if sentiments is not None:
            df["overall_sentiment"] = sentiments[: len(df)] if len(df) <= len(sentiments) else sentiments
            opinion_db["sentiment_polarity_correlation"] = (
                df.groupby(["overall_sentiment", "polarity"]).size().unstack(fill_value=0).to_dict()
                if len(df) > 0
                else {}
            )

        return opinion_db

    def get_top_opinion_phrases(
        self, texts: List[str], polarity: str = "positive", top_n: int = 20
    ) -> List[Tuple[str, int]]:
        df = self.extract_opinions_batch(texts)
        if len(df) == 0:
            return []

        filtered = df[df["polarity"] == polarity]
        phrase_counts = Counter(filtered["expression"])
        return phrase_counts.most_common(top_n)