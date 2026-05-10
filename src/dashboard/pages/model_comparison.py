import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


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
            df_proc = preprocessor.preprocess_dataframe(df)
            df_proc = df_proc[df_proc["cleaned_text"].str.len() > 0]

            feature_ext = FeatureExtractor()
            X = feature_ext.fit_transform_tfidf(df_proc["cleaned_text"].tolist())

            le = LabelEncoder()
            y = le.fit_transform(df_proc["sentiment"].tolist())

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
    st.title("⚖️ Model Comparison")
    st.markdown("Compare predictions from Logistic Regression vs BERT side-by-side.")

    models_loaded = _load_models()
    if not models_loaded:
        st.warning("Baseline model is loading... Please wait.")
        return

    st.markdown("---")

    text_input = st.text_area(
        "Enter text for comparison:", placeholder="Type your review or tweet here...", height=100
    )

    st.markdown("---")

    sentiment_icons = {"positive": "😊", "negative": "😠", "neutral": "😐"}
    sentiment_colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f39c12"}

    if st.button("🔍 Analyze with Logistic Regression", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            preprocessor = st.session_state["preprocessor"]
            model = st.session_state["baseline_model"]
            feature_ext = st.session_state["feature_extractor"]
            le = st.session_state["label_encoder"]

            cleaned = preprocessor.preprocess_single(text_input)
            if not cleaned.strip():
                st.warning("Text became empty after preprocessing.")
            else:
                X_input = feature_ext.transform_tfidf([cleaned])
                lr_pred = model.predict(X_input)[0]
                lr_proba = model.predict_proba(X_input)[0]
                lr_label = le.inverse_transform([lr_pred])[0]
                lr_confidence = float(lr_proba[lr_pred])

                st.session_state["lr_result"] = {
                    "label": lr_label,
                    "confidence": lr_confidence,
                    "probabilities": lr_proba.tolist(),
                    "classes": le.classes_.tolist(),
                }

    if "lr_result" in st.session_state:
        lr = st.session_state["lr_result"]
        lr_label = lr["label"]
        lr_confidence = lr["confidence"]

        color = sentiment_colors.get(lr_label, "#333")

        st.markdown("### 🤖 Logistic Regression Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<div style='text-align:center; font-size:2.5em;'>{sentiment_icons.get(lr_label, '❓')}</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div style='text-align:center; font-size:1.5em; color:{color}; font-weight:bold;'>"
                f"{lr_label.upper()}</div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.metric("Confidence", f"{lr_confidence:.2%}")

        st.markdown("#### Probability Distribution")
        for idx, label in enumerate(lr["classes"]):
            prob = lr["probabilities"][idx]
            st.markdown(f"**{label}**: {prob:.2%} {'█' * int(prob * 20)}{'░' * (20 - int(prob * 20))}")

    st.markdown("---")

    use_bert = st.checkbox("🧠 Enable BERT comparison (requires transformers + torch, ~440MB download)")

    if use_bert:
        if not text_input.strip():
            st.warning("Please enter some text above to analyze with BERT.")
        else:
            try:
                from src.models.bert_model import BERTClassifier
                from src.utils.helpers import is_arabic_text

                language = st.session_state.get("language", "english")

                if language == "arabic" or is_arabic_text(text_input):
                    bert_model_inst = BERTClassifier(model_name="aubmindlab/bert-base-arabertv02", is_arabic=True)
                else:
                    bert_model_inst = BERTClassifier(model_name="bert-base-uncased")

                with st.spinner("Loading BERT model... This may take a while on first run."):
                    bert_model_inst.load_pretrained()

                bert_result = bert_model_inst.predict_with_aspects(text_input)

                bert_label = bert_result.get("sentiment", "unknown")
                bert_confidence = bert_result.get("confidence", 0.0)

                bert_color = sentiment_colors.get(bert_label, "#333")

                st.markdown("### 🧠 BERT Result")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f"<div style='text-align:center; font-size:2.5em;'>{sentiment_icons.get(bert_label, '❓')}</div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f"<div style='text-align:center; font-size:1.5em; color:{bert_color}; font-weight:bold;'>"
                        f"{bert_label.upper()}</div>",
                        unsafe_allow_html=True,
                    )
                with col3:
                    st.metric("Confidence", f"{bert_confidence:.2%}")

                if "aspects" in bert_result:
                    st.markdown("#### Aspect Breakdown")
                    for aspect in bert_result["aspects"]:
                        st.markdown(f"- **{aspect['word']}**: {aspect['sentiment']}")

                if "lr_result" in st.session_state:
                    st.markdown("---")
                    st.markdown("### 📊 Model Comparison Summary")
                    lr_label = st.session_state["lr_result"]["label"]
                    comparison = pd.DataFrame({
                        "Model": ["Logistic Regression", "BERT"],
                        "Sentiment": [lr_label, bert_label],
                        "Confidence": [f"{st.session_state['lr_result']['confidence']:.2%}", f"{bert_confidence:.2%}"],
                    })
                    st.dataframe(comparison, use_container_width=True)

                    if lr_label.lower() == bert_label.lower():
                        st.success("✅ Both models agree on the sentiment!")
                    else:
                        st.warning("⚠️ Models disagree on the sentiment. BERT typically handles context better.")

            except ImportError:
                st.error("BERT requires `transformers` and `torch` packages. Install with: `pip install transformers torch`")
            except Exception as e:
                st.error(f"BERT model error: {e}")
                st.info("Make sure you have internet access for model download (~440MB on first use).")
    else:
        st.info("Check the checkbox above to enable BERT comparison.")
        st.markdown("_Note: BERT requires the `transformers` and `torch` packages and will download ~440MB on first use._")

    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    st.markdown("""
    **Logistic Regression (Baseline)**:
    - Uses TF-IDF features (up to 10,000 features)
    - Fast prediction (~milliseconds)
    - Interpretable coefficients
    - Good for simple sentiment patterns

    **BERT (Deep Learning)**:
    - Uses contextual word embeddings
    - Slower prediction (~seconds)
    - Better at understanding nuance, sarcasm, and context
    - Can handle Arabic text (AraBERT)
    """)