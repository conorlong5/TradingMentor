import streamlit as st

st.set_page_config(page_title="Trading Stock Mentor", layout="centered")

st.title("📈 Trading Stock Mentor")
st.markdown("""
Welcome to **Trading Stock Mentor**, your AI-powered assistant for smarter investing.  
Use the buttons below to explore:
""")

# Button Navigation
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Stock Data"):
        st.switch_page("pages/Stock_Data.py")

with col2:
    if st.button("📰 Sentiment Analysis"):
        st.switch_page("pages/Sentiment_Analysis.py")

with col3:
    if st.button("🤖 Trading Strategy"):
        st.switch_page("pages/Trading_Strategy.py")

st.markdown("---")
st.caption("Built with Streamlit • Powered by Gemini AI & Yahoo Finance")
