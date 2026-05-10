"""
Quick-start script to run the full pipeline:
1. Load sample data
2. Preprocess
3. Extract features
4. Train baseline models
5. Generate reports
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import TweetDataLoader
from src.preprocessing import TweetPreprocessor
from src.feature_extraction import FeatureExtractor
from src.models.baseline import SentimentClassifier, train_all_baselines, compare_models
from src.evaluation import compute_all_metrics, generate_classification_report, ErrorAnalyzer
from src.opinion_mining import OpinionMiner
from src.trend_analysis import TrendAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def main():
    print("=" * 60)
    print("TWEET SENTIMENT ANALYSIS - FULL PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading data...")
    loader = TweetDataLoader()
    df = loader.load_sample_data()
    print(f"  Loaded {len(df)} tweets")
    print(f"  Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")

    # 2. Preprocess
    print("\n[2/6] Preprocessing text...")
    preprocessor = TweetPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df)
    df_processed = df_processed[df_processed["cleaned_text"].str.len() > 0]
    print(f"  {len(df_processed)} tweets after preprocessing")

    stats = preprocessor.get_preprocessing_stats(
        df_processed["original_text"].tolist(),
        df_processed["cleaned_text"].tolist(),
    )
    print(f"  Avg length reduction: {stats['reduction_ratio']:.1%}")

    # 3. Feature extraction
    print("\n[3/6] Extracting features...")
    feature_ext = FeatureExtractor()
    X = feature_ext.fit_transform_tfidf(df_processed["cleaned_text"].tolist())
    print(f"  Feature matrix shape: {X.shape}")

    # 4. Train models
    print("\n[4/6] Training baseline models...")
    le = LabelEncoder()
    y = le.fit_transform(df_processed["sentiment"].tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models, results = train_all_baselines(X_train, y_train, X_test, y_test)

    # 5. Evaluate
    print("\n[5/6] Evaluating models...")
    comparison = compare_models(results)
    print("\nModel Comparison:")
    print(comparation.to_string(index=False))

    best_model_name = comparison.iloc[0]["model"]
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name}")

    report = best_model.get_classification_report(X_test, y_test)
    print(f"\nClassification Report ({best_model_name}):")
    print(report)

    # 6. Error analysis
    print("\n[6/6] Error analysis...")
    y_pred = best_model.predict(X_test)
    texts_test = [df_processed["original_text"].iloc[i] for i in range(len(df_processed))]

    analyzer = ErrorAnalyzer(
        texts_test[:len(y_test)],
        y_test,
        y_pred[:len(y_test)],
    )
    error_report = analyzer.generate_error_report()
    print(error_report)

    # Opinion mining
    print("\n" + "=" * 60)
    print("OPINION MINING")
    print("=" * 60)
    miner = OpinionMiner()
    opinions = miner.extract_opinions_batch(df["text"].tolist())
    print(f"  Extracted {len(opinions)} opinion triplets")
    if len(opinions) > 0:
        print(f"  Polarity distribution: {opinions['polarity'].value_counts().to_dict()}")

    # Trend analysis
    print("\n" + "=" * 60)
    print("TREND ANALYSIS")
    print("=" * 60)
    trend_analyzer = TrendAnalyzer()
    trends = trend_analyzer.compute_sentiment_trends(df)
    if len(trends) > 0:
        print(f"  Average sentiment: {trends['mean_sentiment'].mean():.4f}")

    # Save processed data
    loader.save_processed(df_processed)
    print("\nPipeline complete! Processed data saved.")
    print("Run 'python run_dashboard.py' to launch the dashboard.")


if __name__ == "__main__":
    main()