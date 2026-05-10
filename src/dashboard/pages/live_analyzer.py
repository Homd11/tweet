import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.utils.helpers import is_arabic_text


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
            df = df[df["cleaned_text"].str.len() > 0]

            feature_ext = FeatureExtractor()
            X = feature_ext.fit_transform_tfidf(df["cleaned_text"].tolist())

            le = LabelEncoder()
            y = le.fit_transform(df["sentiment"].tolist())

            model = SentimentClassifier("logistic_regression")
            model.fit(X, y, cv_folds=3, use_calibration=False)

            st.session_state["baseline_model"] = model
            st.session_state["feature_extractor"] = feature_ext
            st.session_state["label_encoder"] = le
            st.session_state["preprocessor"] = preprocessor

        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    return True


def render():
    st.title("🏠 Live Sentiment Analyzer")
    st.markdown("Type any text and get real-time sentiment analysis with confidence scores and aspect breakdown.")

    models_loaded = _load_models()

    if not models_loaded:
        st.warning("Models are loading... Please wait.")
        return

    language = st.session_state.get("language", "english")

    if language == "arabic":
        text_input = st.text_area("أدخل النص هنا:", placeholder="اكتب تعليقك هنا...", height=150)
    else:
        text_input = st.text_area("Enter your text here:", placeholder="Type your review or tweet here...", height=150)

    if not text_input.strip():
        st.info("Please enter some text to analyze.")
        return

    preprocessor = st.session_state["preprocessor"]
    model = st.session_state["baseline_model"]
    feature_ext = st.session_state["feature_extractor"]
    le = st.session_state["label_encoder"]

    cleaned = preprocessor.preprocess_single(text_input)

    if not cleaned.strip():
        st.warning("Text became empty after preprocessing. Please enter more meaningful text.")
        return

    X_input = feature_ext.transform_tfidf([cleaned])
    prediction = model.predict(X_input)[0]
    probabilities = model.predict_proba(X_input)[0]

    sentiment_label = le.inverse_transform([prediction])[0]
    confidence = float(probabilities[prediction])

    sentiment_icons = {"positive": "😊", "negative": "😠", "neutral": "😐"}
    sentiment_colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f39c12"}

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Sentiment")
        st.markdown(
            f"<div style='text-align:center; font-size:3em;'>{sentiment_icons.get(sentiment_label, '❓')}</div>",
            unsafe_allow_html=True,
        )
        color = sentiment_colors.get(sentiment_label, "#333")
        st.markdown(
            f"<div style='text-align:center; font-size:1.5em; color:{color}; font-weight:bold;'>"
            f"{sentiment_label.upper()}</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### Confidence Score")
        st.progress(confidence)
        st.metric("Confidence", f"{confidence:.2%}")

    with col3:
        st.markdown("### All Probabilities")
        for idx, label in enumerate(le.classes_):
            prob = float(probabilities[idx])
            st.markdown(
                f"**{label}**: {prob:.2%} "
                f"{'█' * int(prob * 20)}{'░' * (20 - int(prob * 20))}",
            )

    st.markdown("---")

    from src.opinion_mining import OpinionMiner
    miner = OpinionMiner()
    triplets = miner.extract_opinion_triplets(text_input)

    if triplets:
        st.markdown("### 🔍 Aspect Breakdown")
        aspect_data = []
        for triplet in triplets:
            aspect_data.append({
                "Aspect": triplet["target"],
                "Expression": triplet["expression"],
                "Polarity": triplet["polarity"],
            })
        st.dataframe(aspect_data, use_container_width=False)
    else:
        st.info("No specific aspects detected in this text.")

    st.markdown("---")
    with st.expander("🔧 Preprocessing Details"):
        st.markdown(f"**Original Text:** `{text_input}`")
        st.markdown(f"**Cleaned Text:** `{cleaned}`")
        st.markdown(f"**Original Length:** {len(text_input)} chars | **Cleaned Length:** {len(cleaned)} chars")
        st.markdown(f"**Language Detected:** {'Arabic' if is_arabic_text(text_input) else 'English'}")