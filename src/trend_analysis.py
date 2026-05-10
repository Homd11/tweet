import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime, timedelta
from src.utils.logger import app_logger


class TrendAnalyzer:
    def __init__(self):
        self.trends_data = None

    def compute_sentiment_trends(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        sentiment_column: str = "sentiment",
        freq: str = "D",
    ) -> pd.DataFrame:
        try:
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column)

            sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
            if df[sentiment_column].dtype == object:
                df["sentiment_score"] = df[sentiment_column].map(sentiment_map).astype(float)
            else:
                df["sentiment_score"] = df[sentiment_column].astype(float)

            df = df.dropna(subset=["sentiment_score"])

            grouped = df.set_index(date_column).groupby(pd.Grouper(freq=freq))

            results = []
            for name, group in grouped:
                if len(group) == 0:
                    continue
                scores = group["sentiment_score"]
                results.append({
                    "date": name,
                    "mean_sentiment": float(scores.mean()),
                    "positive_count": int((scores > 0).sum()),
                    "negative_count": int((scores < 0).sum()),
                    "neutral_count": int((scores == 0).sum()),
                    "total_count": len(scores),
                    "positive_ratio": float((scores > 0).sum() / len(scores)),
                    "negative_ratio": float((scores < 0).sum() / len(scores)),
                })

            if not results:
                return pd.DataFrame()

            trends = pd.DataFrame(results).set_index("date")

            trends = trends.dropna(subset=["mean_sentiment"])
            self.trends_data = trends
            return trends

        except Exception as e:
            app_logger.error(f"Trend computation failed: {e}")
            return pd.DataFrame()

    def compute_rolling_average(
        self, trends: pd.DataFrame, window: int = 7, column: str = "mean_sentiment"
    ) -> pd.DataFrame:
        result = trends.copy()
        result[f"{column}_rolling"] = result[column].rolling(window=window, min_periods=1).mean()
        result[f"{column}_rolling_std"] = result[column].rolling(window=window, min_periods=1).std()
        return result

    def detect_sentiment_shifts(
        self,
        trends: pd.DataFrame,
        column: str = "mean_sentiment",
        threshold: float = 2.0,
    ) -> pd.DataFrame:
        if len(trends) < 3:
            return pd.DataFrame()

        shifts = trends.copy()
        mean_val = shifts[column].mean()
        std_val = shifts[column].std()

        shifts["z_score"] = (shifts[column] - mean_val) / std_val if std_val > 0 else 0
        shifts["is_anomaly"] = shifts["z_score"].abs() > threshold
        shifts["shift_direction"] = shifts["z_score"].apply(
            lambda x: "positive_spike" if x > threshold else ("negative_spike" if x < -threshold else "normal")
        )

        shifts["diff"] = shifts[column].diff()
        shifts["pct_change"] = shifts[column].pct_change()
        shifts["significant_shift"] = shifts["diff"].abs() > (std_val * 0.5)

        anomalies = shifts[shifts["is_anomaly"]]
        return anomalies

    def detect_emerging_topics(
        self,
        df: pd.DataFrame,
        date_column: str = "date",
        text_column: str = "text",
        sentiment_column: str = "sentiment",
        freq: str = "W",
        top_n: int = 10,
    ) -> Dict[str, pd.DataFrame]:
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        results = {}

        negative_df = df[df[sentiment_column] == "negative"]
        if len(negative_df) > 0:
            negative_df = negative_df.copy()
            negative_df[date_column] = pd.to_datetime(negative_df[date_column])
            negative_grouped = negative_df.set_index(date_column).groupby(pd.Grouper(freq=freq))
            topic_trends = []

            for period, group in negative_grouped:
                if len(group) > 0:
                    words = []
                    for text in group[text_column]:
                        if isinstance(text, str):
                            words.extend(text.lower().split())

                    stop_words = {
                        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                        "have", "has", "had", "do", "does", "did", "will", "would", "could",
                        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
                        "on", "with", "at", "by", "from", "as", "into", "about", "but",
                        "or", "and", "not", "no", "this", "that", "it", "i", "me", "my",
                        "we", "our", "you", "your", "he", "she", "they", "them", "so",
                        "just", "very", "really", "too", "also", "then", "than", "more",
                    }
                    words = [w for w in words if w not in stop_words and len(w) > 2]
                    word_counts = Counter(words).most_common(top_n)
                    topic_trends.append(
                        {"period": period, "top_words": dict(word_counts), "count": len(group)}
                    )

            results["negative_topics"] = pd.DataFrame(topic_trends)

        positive_df = df[df[sentiment_column] == "positive"]
        if len(positive_df) > 0:
            positive_df = positive_df.copy()
            positive_df[date_column] = pd.to_datetime(positive_df[date_column])
            positive_grouped = positive_df.set_index(date_column).groupby(pd.Grouper(freq=freq))
            pos_topics = []

            for period, group in positive_grouped:
                if len(group) > 0:
                    words = []
                    for text in group[text_column]:
                        if isinstance(text, str):
                            words.extend(text.lower().split())
                    words = [w for w in words if w not in stop_words and len(w) > 2]
                    word_counts = Counter(words).most_common(top_n)
                    pos_topics.append(
                        {"period": period, "top_words": dict(word_counts), "count": len(group)}
                    )

            results["positive_topics"] = pd.DataFrame(pos_topics)

        return results

    def correlate_with_events(
        self,
        trends: pd.DataFrame,
        events: List[Dict[str, str]],
        column: str = "mean_sentiment",
        window_days: int = 3,
    ) -> pd.DataFrame:
        correlations = []

        for event in events:
            event_date = pd.to_datetime(event["date"])
            event_name = event["name"]

            before_start = event_date - timedelta(days=window_days)
            after_end = event_date + timedelta(days=window_days)

            before_mask = (trends.index >= before_start) & (trends.index < event_date)
            after_mask = (trends.index >= event_date) & (trends.index <= after_end)

            before_sentiment = trends.loc[before_mask, column].mean() if before_mask.any() else None
            after_sentiment = trends.loc[after_mask, column].mean() if after_mask.any() else None
            change = (after_sentiment - before_sentiment) if (before_sentiment is not None and after_sentiment is not None) else None

            correlations.append({
                "event": event_name,
                "date": event_date,
                "before_sentiment": before_sentiment,
                "after_sentiment": after_sentiment,
                "sentiment_change": change,
            })

        return pd.DataFrame(correlations)

    def generate_trend_report(
        self,
        trends: pd.DataFrame,
        anomalies: Optional[pd.DataFrame] = None,
    ) -> str:
        report_lines = [
            "=" * 60,
            "SENTIMENT TREND REPORT",
            "=" * 60,
            f"Period: {trends.index.min()} to {trends.index.max()}",
            f"Total Data Points: {len(trends)}",
            f"Average Sentiment: {trends['mean_sentiment'].mean():.4f}",
            f"Sentiment Std Dev: {trends['mean_sentiment'].std():.4f}",
            f"Total Positive: {trends['positive_count'].sum():.0f}",
            f"Total Negative: {trends['negative_count'].sum():.0f}",
            f"Total Neutral: {trends['neutral_count'].sum():.0f}",
            "",
            "TREND DIRECTION:",
        ]

        if len(trends) > 1:
            first_half = trends["mean_sentiment"].iloc[: len(trends) // 2].mean()
            second_half = trends["mean_sentiment"].iloc[len(trends) // 2 :].mean()
            if second_half > first_half + 0.05:
                report_lines.append("  Overall: IMPROVING (sentiment trending positive)")
            elif second_half < first_half - 0.05:
                report_lines.append("  Overall: DECLINING (sentiment trending negative)")
            else:
                report_lines.append("  Overall: STABLE (sentiment relatively unchanged)")

        if anomalies is not None and len(anomalies) > 0:
            report_lines.append("")
            report_lines.append(f"ANOMALIES DETECTED: {len(anomalies)}")
            for _, row in anomalies.head(10).iterrows():
                report_lines.append(f"  {row.name}: {row['shift_direction']} (score: {row['mean_sentiment']:.4f})")

        return "\n".join(report_lines)