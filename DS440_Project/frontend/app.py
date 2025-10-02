import yfinance as yf
import streamlit as st

st.title("📈 Trading Mentor - Stage 1 Demo")

# Input from user
symbol = st.text_input("Enter a stock symbol:", "AAPL")

if symbol:
    # Fetch stock data
    data = yf.download(symbol, period="1mo", interval="1d")
    
    st.subheader(f"Closing Prices for {symbol}")
    st.line_chart(data["Close"])
