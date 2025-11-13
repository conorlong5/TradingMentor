import streamlit as st
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

col_back, col_title = st.columns([1, 11])
with col_back:
    if st.button("← Home"):
        st.switch_page("app.py")
with col_title:
    st.title("📊 Stock Data Viewer")

st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("📰 Stock Sentiment Analysis")

symbol = st.text_input("Enter a Stock Symbol:", "AAPL")

if st.button("Analyze Sentiment"):
    analyzer = SentimentIntensityAnalyzer()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    query = f"{symbol} stock"

    articles = newsapi.get_everything(
        q=query,
        from_param=start_date.strftime("%Y-%m-%d"),
        to=end_date.strftime("%Y-%m-%d"),
        language="en",
        sort_by="relevancy",
        page_size=10
    )

    sentiments = []
    for a in articles["articles"]:
        text = f"{a.get('title','')} {a.get('description','')}"
        score = analyzer.polarity_scores(text)["compound"]
        sentiments.append(score)

    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    label = "Positive 😀" if avg_sentiment > 0.05 else \
            "Negative 😞" if avg_sentiment < -0.05 else "Neutral 😐"

    st.metric("Sentiment", label, delta=f"{avg_sentiment:.2f}")
