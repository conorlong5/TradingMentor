# --- ensure project root is importable ---
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]  # -> .../DS440_Project
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ----------------------------------------

import streamlit as st

# Page config first
st.set_page_config(page_title="Trading Stock Mentor", layout="wide")

st.title("📈 Trading Stock Mentor")
st.markdown(
    "Welcome to **Trading Stock Mentor**, your AI-powered assistant for smarter investing."
)
st.write("Use the links below to explore:")

# Link-based navigation (no switch_page)
col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/Stock_Data.py", label="📊 Stock Data")
with col2:
    st.page_link("pages/Sentiment_Analysis.py", label="📰 Sentiment Analysis")
with col3:
    st.page_link("pages/Trading_Strategy.py", label="🤖 Trading Strategy")

st.markdown("---")
st.caption("Built with Streamlit • Powered by Gemini AI & Yahoo Finance")
