import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import os
import google.generativeai as genai
from dotenv import load_dotenv
import plotly.graph_objects as go
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# =========================
# SETUP
# =========================
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="LLM Stock Strategy Generator", layout="wide")

st.title("📈 LLM-Powered Stock Strategy Generator")
st.markdown("Generate AI-driven trading strategies using live market data and indicators.")

st.markdown("""
<style>
    .stChatMessage-user {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage-assistant {
        background-color: #262730;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# USER INPUT
# =========================
col1, col2 = st.columns(2)
with col1:
    symbol = st.text_input("Enter a Stock Symbol (ex: AAPL, TSLA, NVDA):", "AAPL")
with col2:
    period = st.selectbox("Select Time Period", ["1d", "1mo", "6mo", "1y", "5y"], index=0)

if not symbol:
    st.stop()

# =========================
# FETCH DATA
# =========================
try:
    st.write(f"Fetching data for **{symbol}**...")
    data = yf.download(symbol, period=period, interval="1d")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    if data.empty:
        st.error("No data found. Try another symbol.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# =========================
# CALCULATE INDICATORS
# =========================
data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
data["MACD"] = ta.trend.MACD(data["Close"]).macd()
data["Signal"] = ta.trend.MACD(data["Close"]).macd_signal()
data["SMA_50"] = data["Close"].rolling(window=50).mean()
data["SMA_200"] = data["Close"].rolling(window=200).mean()
data["BB_High"] = ta.volatility.BollingerBands(data["Close"]).bollinger_hband()
data["BB_Low"] = ta.volatility.BollingerBands(data["Close"]).bollinger_lband()

latest = data.iloc[-1]

# =========================
# DISPLAY INDICATOR SUMMARY
# =========================
summary = f"""
Stock: {symbol}
Time period: {period}
Latest close price: {latest['Close']:.2f}
RSI: {latest['RSI']:.2f}
MACD: {latest['MACD']:.2f}
Signal line: {latest['Signal']:.2f}
50-day SMA: {latest['SMA_50']:.2f}
200-day SMA: {latest['SMA_200']:.2f}
Bollinger Band High: {latest['BB_High']:.2f}
Bollinger Band Low: {latest['BB_Low']:.2f}
"""

st.subheader("📊 Indicator Summary")
st.text(summary)

# =========================
# VISUALIZATION SECTION
# =========================
st.subheader("📉 Stock Price Visualization")

# Chart type selection
chart_type = st.radio(
    "Select chart type:",
    ["Candlestick", "Line Chart"],
    horizontal=True
)

# --- Candlestick Chart ---
candle_fig = go.Figure()

candle_fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Candlestick",
    increasing_line_color="green",
    decreasing_line_color="red"
))
candle_fig.add_trace(go.Scatter(
    x=data.index, y=data["SMA_50"],
    mode="lines", name="SMA 50", line=dict(color="blue", width=1.5)
))
candle_fig.add_trace(go.Scatter(
    x=data.index, y=data["SMA_200"],
    mode="lines", name="SMA 200", line=dict(color="orange", width=1.5)
))
candle_fig.update_layout(
    title=f"{symbol} Price Chart (Candlestick)",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# --- Line Chart ---
line_fig = go.Figure()
line_fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close", line=dict(color="white", width=2)))
line_fig.add_trace(go.Scatter(x=data.index, y=data["SMA_50"], mode="lines", name="SMA 50", line=dict(color="blue", width=1.5)))
line_fig.add_trace(go.Scatter(x=data.index, y=data["SMA_200"], mode="lines", name="SMA 200", line=dict(color="orange", width=1.5)))
line_fig.update_layout(
    title=f"{symbol} Price Chart (Line)",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    template="plotly_dark",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# --- Display Selected Chart ---
if chart_type == "Line Chart":
    st.plotly_chart(line_fig, use_container_width=True, key="line_chart")
else:
    st.plotly_chart(candle_fig, use_container_width=True, key="candlestick_chart")


# =========================
# SENTIMENT ANALYSIS SECTION
# =========================
st.write("Analyzing recent news sentiment...")

try:
    # Setup
    newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
    analyzer = SentimentIntensityAnalyzer()

    # Fetch news headlines about the stock
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

    # Analyze sentiment safely
    sentiments = []
    for article in articles["articles"]:
        title = article.get("title", "")
        description = article.get("description", "")
        if not isinstance(title, str):
            title = ""
        if not isinstance(description, str):
            description = ""
        text = f"{title}. {description}".strip()
        if text:
            score = analyzer.polarity_scores(text)["compound"]
            sentiments.append(score)

    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
        sentiment_label = "Positive 😀" if avg_sentiment > 0.05 else \
                          "Negative 😞" if avg_sentiment < -0.05 else "Neutral 😐"
    else:
        avg_sentiment = 0
        sentiment_label = "Neutral (No recent news found)"

    st.metric("📰 News Sentiment", sentiment_label, delta=f"{avg_sentiment:.2f}")

except Exception as e:
    st.error(f"Error analyzing sentiment: {e}")
    avg_sentiment = 0
    sentiment_label = "Neutral"



# =========================
# BUILD PROMPT FOR GEMINI
# =========================
prompt = f"""
You are a professional quantitative trading strategist.
You have access to both recent technical indicators and market sentiment data.

Data:
{summary}

Market Sentiment:
Overall sentiment score: {avg_sentiment:.2f}
Sentiment label: {sentiment_label}

Instructions:
- Suggest one clear trading strategy (long, short, or neutral)
- Include:
    • Strategy name
    • Description
    • Entry rules
    • Exit rules
    • Stop loss and take profit levels
    • Expected holding period
    • Risk level (Low/Moderate/High)
    • Backtesting code and results
Return the output in a clean, readable markdown format.
"""


# =========================
# GENERATE STRATEGY WITH GEMINI
# =========================
st.subheader("🤖 Generate Strategy")

if st.button("Generate Strategy"):
    with st.spinner("Thinking..."):
        try:
            model = genai.GenerativeModel("gemini-flash-latest")
            response = model.generate_content(prompt)
            st.markdown(response.text)
        except Exception as e:
            st.error(f"LLM Error: {e}")




# =========================
# INTERACTIVE Q&A SECTION
# =========================
st.subheader("💬 Interactive Q&A with the AI Strategist")

# --- Keep chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Display previous messages ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input ---
user_query = st.chat_input("Ask a question about this stock, its indicators, or sentiment...")

if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # --- Detect if user wants a graph ---
    keywords = ["plot", "graph", "show", "visualize", "chart"]
    if any(word in user_query.lower() for word in keywords):
        # Detect time range
        import re
        time_period = "1y"  # default
        if re.search(r"last\s+(\d+)\s*day", user_query.lower()):
            days = int(re.search(r"last\s+(\d+)\s*day", user_query.lower()).group(1))
            time_period = f"{days}d"
        elif re.search(r"last\s+(\d+)\s*month", user_query.lower()):
            months = int(re.search(r"last\s+(\d+)\s*month", user_query.lower()).group(1))
            time_period = f"{months}mo"
        elif re.search(r"last\s+(\d+)\s*year", user_query.lower()):
            years = int(re.search(r"last\s+(\d+)\s*year", user_query.lower()).group(1))
            time_period = f"{years}y"

        # --- Fetch data again with new period ---
        try:
            st.write(f"📈 Fetching {time_period} of data for **{symbol}**...")
            chart_data = yf.download(symbol, period=time_period, interval="1d")
            if isinstance(chart_data.columns, pd.MultiIndex):
                chart_data.columns = [col[0] for col in chart_data.columns]
        except Exception as e:
            st.error(f"Error fetching new data: {e}")
            chart_data = data  # fallback

        with st.chat_message("assistant"):
            st.markdown("📊 Sure! Here's a visualization based on your request:")

            fig = go.Figure()

            # --- RSI ---
            if "rsi" in user_query.lower():
                fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Close"], mode="lines", name="Close Price"))
                fig.add_trace(go.Scatter(x=chart_data.index, y=ta.momentum.RSIIndicator(chart_data["Close"], window=14).rsi(), mode="lines", name="RSI"))
                fig.update_layout(
                    title=f"{symbol} - RSI vs Close Price ({time_period})",
                    xaxis_title="Date", yaxis_title="Value",
                    template="plotly_dark", height=500
                )

            # --- MACD ---
            elif "macd" in user_query.lower():
                macd = ta.trend.MACD(chart_data["Close"])
                fig.add_trace(go.Scatter(x=chart_data.index, y=macd.macd(), mode="lines", name="MACD"))
                fig.add_trace(go.Scatter(x=chart_data.index, y=macd.macd_signal(), mode="lines", name="Signal Line"))
                fig.update_layout(
                    title=f"{symbol} - MACD and Signal Line ({time_period})",
                    xaxis_title="Date", yaxis_title="Value",
                    template="plotly_dark", height=500
                )

            # --- Bollinger Bands ---
            elif "bollinger" in user_query.lower():
                bb = ta.volatility.BollingerBands(chart_data["Close"])
                fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Close"], mode="lines", name="Close"))
                fig.add_trace(go.Scatter(x=chart_data.index, y=bb.bollinger_hband(), mode="lines", name="Upper Band"))
                fig.add_trace(go.Scatter(x=chart_data.index, y=bb.bollinger_lband(), mode="lines", name="Lower Band"))
                fig.update_layout(
                    title=f"{symbol} - Bollinger Bands ({time_period})",
                    xaxis_title="Date", yaxis_title="Price (USD)",
                    template="plotly_dark", height=500
                )

            # --- Default Close Chart ---
            else:
                fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["Close"], mode="lines", name="Close Price"))
                fig.update_layout(
                    title=f"{symbol} - Closing Price Trend ({time_period})",
                    xaxis_title="Date", yaxis_title="Price (USD)",
                    template="plotly_dark", height=500
                )

            st.plotly_chart(fig, use_container_width=True)
            st.session_state.chat_history.append({"role": "assistant", "content": f"Displayed {time_period} chart for {symbol}"})

    else:
        # --- Regular LLM Q&A ---
        contextual_prompt = f"""
        You are a professional quantitative trading strategist with access to:

        Technical Data:
        {summary}

        Sentiment:
        Score: {avg_sentiment:.2f} ({sentiment_label})

        The user asked:
        {user_query}

        Give a precise, data-based answer using both technical and sentiment insights.
        """

        try:
            model = genai.GenerativeModel("gemini-flash-latest")
            response = model.generate_content(contextual_prompt)
            ai_reply = response.text
        except Exception as e:
            ai_reply = f"LLM Error: {e}"

        with st.chat_message("assistant"):
            st.markdown(ai_reply)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
