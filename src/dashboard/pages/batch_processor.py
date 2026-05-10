import streamlit as st
import pandas as pd
import io
from src.data_loader import TweetDataLoader
from src.preprocessing import TweetPreprocessor
from src.feature_extraction import FeatureExtractor
from src.opinion_mining import OpinionMiner


def _load_models():
    if "baseline_model" not in st.session_state:
        try:
            from src.data_loader import TweetDataLoader
            from src.preprocessing import TweetPreprocessor
            from src.feature_extraction import FeatureExtractor
            from src.models.baseline import SentimentClassifier
            from sklearn.preprocessing import LabelEncoder

            loader = TweetDataLoader()
            df = loader.load_sample_data()

            preprocessor = TweetPreprocessor()
            df = preprocessor.preprocess_dataframe(df)

            feature_ext = FeatureExtractor()
            X = feature_ext.fit_transform_tfidf(df["cleaned_text"].tolist())

            le = LabelEncoder()
            y = le.fit_transform(df["sentiment"].tolist())

            model = SentimentClassifier("logistic_regression")
            model.fit(X, y, use_calibration=True)

            st.session_state["baseline_model"] = model
            st.session_state["feature_extractor"] = feature_ext
            st.session_state["label_encoder"] = le
            st.session_state["preprocessor"] = preprocessor

        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    return True


def render():
    st.title("📁 Batch Processor")
    st.markdown("Upload a CSV file of reviews and download results with predicted labels and ABSA output.")

    models_loaded = _load_models()
    if not models_loaded:
        st.warning("Models are loading... Please wait.")
        return

    st.markdown("---")
    st.markdown("### Upload CSV File")
    st.markdown("Expected columns: `text` (required), optionally `sentiment`, `date`")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! {len(df)} rows found.")

            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column.")
                return

            st.markdown("### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            if st.button("🚀 Process All Reviews", type="primary"):
                preprocessor = st.session_state["preprocessor"]
                model = st.session_state["baseline_model"]
                feature_ext = st.session_state["feature_extractor"]
                le = st.session_state["label_encoder"]

                progress = st.progress(0)
                status = st.empty()

                cleaned_texts = preprocessor.preprocess_batch(df["text"].tolist(), show_progress=False)
                X = feature_ext.transform_tfidf(cleaned_texts)
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)

                sentiment_labels = le.inverse_transform(predictions)

                df["predicted_sentiment"] = sentiment_labels
                df["confidence"] = [float(probabilities[i][predictions[i]]) for i in range(len(predictions))]
                df["cleaned_text"] = cleaned_texts

                for idx, label in enumerate(le.classes_):
                    df[f"prob_{label}"] = probabilities[:, idx]

                status.text("Running aspect-based sentiment analysis...")
                miner = OpinionMiner()
                all_aspects = []

                batch_size = 50
                for i in range(0, len(df), batch_size):
                    batch = df["text"].iloc[i : i + batch_size].tolist()
                    for text in batch:
                        triplets = miner.extract_opinion_triplets(text)
                        if triplets:
                            aspects = "; ".join(
                                [f"{t['target']}({t['expression']}:{t['polarity']})" for t in triplets]
                            )
                        else:
                            aspects = "none"
                        all_aspects.append(aspects)

                    progress.progress(min(0.5 + (i + batch_size) / len(df) * 0.5, 1.0))

                df["aspects"] = all_aspects

                progress.progress(1.0)
                status.success("Processing complete!")

                st.markdown("### Results")
                st.dataframe(df.head(20), use_container_width=True)

                st.markdown("### Sentiment Distribution")
                sentiment_counts = df["predicted_sentiment"].value_counts()
                st.bar_chart(sentiment_counts)

                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv,
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

    else:
        st.info("Upload a CSV file to get started.")
        st.markdown("#### Sample CSV format:")
        sample = pd.DataFrame({
            "text": [
                "I love this product!",
                "Terrible experience.",
                "It's okay, nothing special.",
            ],
            "sentiment": ["positive", "negative", "neutral"],
        })
        st.dataframe(sample)
        csv = sample.to_csv(index=False)
        st.download_button(
            "📥 Download Sample CSV Template",
            data=csv,
            file_name="sample_template.csv",
            mime="text/csv",
        )