import streamlit as st
from src.config.settings import DASHBOARD_CONFIG

st.set_page_config(
    page_title=DASHBOARD_CONFIG["title"],
    page_icon=DASHBOARD_CONFIG["page_icon"],
    layout=DASHBOARD_CONFIG["layout"],
    initial_sidebar_state=DASHBOARD_CONFIG["initial_sidebar_state"],
)

st.sidebar.title("📊 Tweet Sentiment Analyzer")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Live Analyzer",
        "📁 Batch Processor",
        "💡 Opinion Dashboard",
        "📈 Trend Monitor",
        "⚖️ Model Comparison",
    ],
)

st.sidebar.markdown("---")

language = st.sidebar.radio("Language", ["English", "العربية (Arabic)"], index=0)
st.session_state["language"] = "arabic" if language == "العربية (Arabic)" else "english"

st.sidebar.markdown("---")
st.sidebar.info("Tweet Sentiment Analysis & Text Mining Dashboard\n\nAlamein University - AIE323")

if page == "🏠 Live Analyzer":
    from src.dashboard.pages.live_analyzer import render
    render()
elif page == "📁 Batch Processor":
    from src.dashboard.pages.batch_processor import render
    render()
elif page == "💡 Opinion Dashboard":
    from src.dashboard.pages.opinion_dashboard import render
    render()
elif page == "📈 Trend Monitor":
    from src.dashboard.pages.trend_monitor import render
    render()
elif page == "⚖️ Model Comparison":
    from src.dashboard.pages.model_comparison import render
    render()