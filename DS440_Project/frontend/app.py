import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title = "LLM Stock Strategy Generator", layout = "wide")

st.title("LLM-Powered Stock Strategy Generator")
st.markdown("Generate AI-driven trading strategies usin live market data and indicators.")

col1, col2 = st. columns(2)
with col1:
    symbol = st.text_input("Enter a Stock Symbol (ex: AAPL, TSLA, NVDA):", "AAPL")
with col2:
    period = st.selectbox("Select Time Period", ['5d', '1mo', '3mo', '6mo', '1y'], index = 0)

if not symbol:
    st.stop()

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

data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
data["MACD"] = ta.trend.MACD(data["Close"]).macd()
data["Signal"] = ta.trend.MACD(data["Close"]).macd_signal()
data["SMA_50"] = data["Close"].rolling(window=50).mean()
data["SMA_200"] = data["Close"].rolling(window=200).mean()
data["BB_High"] = ta.volatility.BollingerBands(data["Close"]).bollinger_hband()
data["BB_Low"] = ta.volatility.BollingerBands(data["Close"]).bollinger_lband()

latest = data.iloc[-1]

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

st.subheader("Indicator Summary")
st.text(summary)

prompt = f"""
You are a professional quantitative trading strategist.
Given this recent market data, create a trading strategy for the stock below.

Data:
{summary}

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
    • Optional: simple pseudocode for backtesting
Return the output in a clean, readable markdown format.
"""

st.subheader("Generate Strategy")

if st.button("Generate Strategy"):
    with st.spinner("Thinking..."):
        try:
            model = genai.GenerativeModel("gemini-flash-latest")

            try:
                response = model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"LLM Error: {e}")

        except Exception as e:
            st.error(f"LLM Error: {e}")

st.subheader("Price Chart")
st.line_chart(data[["Close"]])
