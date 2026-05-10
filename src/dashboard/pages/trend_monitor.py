import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import TweetDataLoader
from src.preprocessing import TweetPreprocessor
from src.trend_analysis import TrendAnalyzer


def render():
    st.title("📈 Trend Monitor")
    st.markdown("Monitor sentiment trends over time with anomaly detection and event correlation.")

    loader = TweetDataLoader()
    df = loader.load_sample_data()

    if "date" not in df.columns:
        st.warning("Dataset does not contain date information. Using simulated dates.")
        df["date"] = [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=6 * i) for i in range(len(df))]

    preprocessor = TweetPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df)

    analyzer = TrendAnalyzer()
    
    st.sidebar.markdown("---")
    freq = st.sidebar.selectbox("Aggregation Frequency", ["h", "6h", "D", "W", "ME"], index=2, format_func=lambda x: {"h": "Hourly", "6h": "6-Hourly", "D": "Daily", "W": "Weekly", "ME": "Monthly"}.get(x, x))
    rolling_window = st.sidebar.slider("Rolling Average Window", 3, 30, 7)
    anomaly_threshold = st.sidebar.slider("Anomaly Threshold (σ)", 1.0, 4.0, 2.0, 0.5)

    trends = analyzer.compute_sentiment_trends(df, freq=freq)

    if len(trends) == 0:
        st.warning("No trend data available.")
        return

    trends_with_rolling = analyzer.compute_rolling_average(trends, window=rolling_window)
    anomalies = analyzer.detect_sentiment_shifts(trends, threshold=anomaly_threshold)

    st.markdown("### 📊 Sentiment Trend Over Time")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trends_with_rolling.index,
        y=trends_with_rolling["mean_sentiment"],
        mode="lines",
        name="Daily Sentiment",
        line=dict(color="rgba(100,100,100,0.3)", width=1),
    ))

    fig.add_trace(go.Scatter(
        x=trends_with_rolling.index,
        y=trends_with_rolling["mean_sentiment_rolling"],
        mode="lines",
        name=f"{rolling_window}-Day Rolling Average",
        line=dict(color="#2ecc71", width=3),
    ))

    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies["mean_sentiment"],
            mode="markers",
            name="Anomalies",
            marker=dict(color="red", size=12, symbol="star"),
        ))

    fig.update_layout(
        title="Sentiment Score Over Time",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        hovermode="x unified",
    )
    st.plotly_chart(fig)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📉 Sentiment Volume Over Time")
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=trends.index,
            y=trends["positive_count"],
            name="Positive",
            marker_color="#2ecc71",
        ))
        fig_volume.add_trace(go.Bar(
            x=trends.index,
            y=trends["negative_count"],
            name="Negative",
            marker_color="#e74c3c",
        ))
        fig_volume.update_layout(
            title="Daily Post Volume by Sentiment",
            barmode="stack",
            xaxis_title="Date",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_volume)

    with col2:
        st.markdown("### 📊 Sentiment Ratio")
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(
            x=trends_with_rolling.index,
            y=trends_with_rolling.get("positive_ratio", trends_with_rolling["mean_sentiment"]),
            mode="lines",
            name="Positive Ratio",
            line=dict(color="#2ecc71"),
        ))
        fig_ratio.update_layout(
            title="Positive Sentiment Ratio Over Time",
            xaxis_title="Date",
            yaxis_title="Ratio",
        )
        st.plotly_chart(fig_ratio)

    st.markdown("---")

    if len(anomalies) > 0:
        st.markdown("### 🚨 Detected Anomalies")
        anomaly_display = anomalies[["mean_sentiment", "z_score", "shift_direction"]].copy()
        anomaly_display.columns = ["Sentiment Score", "Z-Score", "Direction"]
        st.dataframe(anomaly_display, use_container_width=False)
    else:
        st.info("No anomalies detected with current threshold.")

    st.markdown("---")

    st.markdown("### 📋 Trend Summary")
    trend_report = analyzer.generate_trend_report(trends, anomalies)
    st.text(trend_report)

    emerging = analyzer.detect_emerging_topics(df, freq="W", top_n=10)
    if "negative_topics" in emerging and len(emerging["negative_topics"]) > 0:
        st.markdown("### 🔴 Emerging Negative Topics")
        st.dataframe(emerging["negative_topics"], use_container_width=False)