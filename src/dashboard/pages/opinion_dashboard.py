import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import TweetDataLoader
from src.preprocessing import TweetPreprocessor
from src.opinion_mining import OpinionMiner


def render():
    st.title("💡 Opinion Dashboard")
    st.markdown("Explore aspect-level sentiment analysis with filterable heatmaps and top opinion phrases.")

    try:
        loader = TweetDataLoader()
        df = loader.load_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    miner = OpinionMiner()

    try:
        opinions_df = miner.extract_opinions_batch(df["text"].tolist())
    except Exception as e:
        st.error(f"Error extracting opinions: {e}")
        opinions_df = pd.DataFrame(columns=["text", "target", "expression", "polarity"])

    if len(opinions_df) == 0:
        st.warning("No opinions extracted from the sample data. The opinion miner uses spaCy for dependency parsing. Try installing it with: `python -m spacy download en_core_web_sm`")
        st.markdown("---")
        st.markdown("### Demo Mode: Rule-Based Opinion Extraction")

        positive_words = {"love", "great", "amazing", "excellent", "fantastic", "wonderful", "best", "good", "happy", "awesome", "incredible", "delicious", "phenomenal", "brilliant", "outstanding"}
        negative_words = {"hate", "terrible", "awful", "worst", "bad", "horrible", "disappointing", "angry", "poor", "frustrated", "useless", "annoying", "broken", "ugly", "disgusting"}

        rule_triplets = []
        for text in df["text"]:
            words = text.lower().split()
            words = [w.strip(".,!?;:") for w in words]
            for word in words:
                if word in positive_words:
                    rule_triplets.append({"text": text, "target": "general", "expression": word, "polarity": "positive"})
                elif word in negative_words:
                    rule_triplets.append({"text": text, "target": "general", "expression": word, "polarity": "negative"})

        if not rule_triplets:
            st.info("No opinions found even with rule-based extraction.")
            return

        opinions_df = pd.DataFrame(rule_triplets)

    st.markdown("---")

    st.sidebar.markdown("---")
    filter_sentiment = st.sidebar.multiselect(
        "Filter by Polarity", ["positive", "negative", "neutral"], default=["positive", "negative", "neutral"]
    )
    min_occurrences = st.sidebar.slider("Minimum Occurrences", 1, 10, 1)

    valid_filters = [f for f in filter_sentiment if f in opinions_df["polarity"].unique()]
    if not valid_filters:
        valid_filters = opinions_df["polarity"].unique().tolist()

    filtered_df = opinions_df[opinions_df["polarity"].isin(valid_filters)]

    if len(filtered_df) == 0:
        st.warning("No opinions match the filter criteria.")
        return

    st.markdown("### 🔥 Aspect-Sentiment Heatmap")

    target_counts = filtered_df.groupby(["target", "polarity"]).size().reset_index(name="count")
    target_counts = target_counts[target_counts["count"] >= min_occurrences]

    if len(target_counts) > 0:
        heatmap_data = target_counts.pivot_table(
            index="target", columns="polarity", values="count", fill_value=0
        )

        for col in ["positive", "negative", "neutral"]:
            if col not in heatmap_data.columns:
                heatmap_data[col] = 0

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values.tolist(),
                x=heatmap_data.columns.tolist(),
                y=heatmap_data.index.tolist(),
                colorscale="RdYlGn",
                text=heatmap_data.values.tolist(),
                texttemplate="%{text}",
                textfont={"size": 12},
            )
        )
        fig.update_layout(
            title="Aspect-Sentiment Heatmap",
            xaxis_title="Sentiment",
            yaxis_title="Aspect (Target)",
            height=max(400, len(heatmap_data) * 30),
        )
        st.plotly_chart(fig)
    else:
        st.info("Not enough data for heatmap visualization.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌟 Top Positive Opinion Phrases")
        positive_phrases = miner.get_top_opinion_phrases(df["text"].tolist(), polarity="positive", top_n=15)
        if positive_phrases:
            pos_df = pd.DataFrame(positive_phrases, columns=["Phrase", "Count"])
            st.dataframe(pos_df, use_container_width=False)
        else:
            pos_words = opinions_df[opinions_df["polarity"] == "positive"]["expression"].value_counts().head(15)
            if len(pos_words) > 0:
                st.dataframe(pos_words.reset_index().rename(columns={"index": "Phrase", "expression": "Phrase", "count": "Count"}), use_container_width=False)
            else:
                st.info("No positive phrases found.")

    with col2:
        st.markdown("### 😠 Top Negative Opinion Phrases")
        negative_phrases = miner.get_top_opinion_phrases(df["text"].tolist(), polarity="negative", top_n=15)
        if negative_phrases:
            neg_df = pd.DataFrame(negative_phrases, columns=["Phrase", "Count"])
            st.dataframe(neg_df, use_container_width=False)
        else:
            neg_words = opinions_df[opinions_df["polarity"] == "negative"]["expression"].value_counts().head(15)
            if len(neg_words) > 0:
                st.dataframe(neg_words.reset_index(), use_container_width=False)
            else:
                st.info("No negative phrases found.")

    st.markdown("---")
    st.markdown("### 📊 Polarity Distribution")

    polarity_counts = filtered_df["polarity"].value_counts()
    fig_pie = px.pie(
        values=polarity_counts.values,
        names=polarity_counts.index,
        color=polarity_counts.index,
        color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f39c12"},
        title="Overall Polarity Distribution",
    )
    st.plotly_chart(fig_pie)

    st.markdown("---")
    st.markdown("### 📋 All Extracted Opinions")
    display_cols = [c for c in ["text", "target", "expression", "polarity"] if c in filtered_df.columns]
    st.dataframe(filtered_df[display_cols], use_container_width=False)