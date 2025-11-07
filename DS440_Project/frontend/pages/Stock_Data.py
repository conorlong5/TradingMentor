import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta

st.set_page_config(page_title="Stock Data", layout="wide")

st.title("📊 Stock Data Viewer")



# ----------------------------
# Inputs
# ----------------------------
col1, col2 = st.columns(2)
with col1:
    symbol = st.text_input("Enter a Stock Symbol (e.g. AAPL, TSLA, NVDA):", "AAPL")
with col2:
    period = st.selectbox("Select Time Period", ["1d", "1mo", "6mo", "1y", "5y"], index=1)

# ----------------------------
# Session state init
# ----------------------------
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None
if "stock_symbol" not in st.session_state:
    st.session_state.stock_symbol = None

# ----------------------------
# Helper function
# ----------------------------
def get_interval(p):
    if p == "1d":
        return "30m"
    elif p == "1mo":
        return "1h"
    return "1d"

# ----------------------------
# Fetch Button
# ----------------------------
if st.button("Fetch Data"):
    try:
        interval = get_interval(period)
        data = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        if data.empty:
            st.error("⚠️ No data found. Try a different symbol or longer period.")
            st.stop()

        # Compute SMAs
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["SMA_200"] = data["Close"].rolling(window=200).mean()

        st.session_state.stock_data = data
        st.session_state.stock_symbol = symbol
        st.success(f"✅ Data fetched for {symbol}")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# ----------------------------
# Display Section
# ----------------------------
if st.session_state.stock_data is not None:
    data = st.session_state.stock_data
    symbol = st.session_state.stock_symbol

    # -----------------------------------
    # 📈 1. Graph Section (moved first)
    # -----------------------------------
    st.markdown("### 📈 Stock Chart")

    # Candlestick
    candle_fig = go.Figure()
    candle_fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Candlestick",
        increasing_line_color="green",
        decreasing_line_color="red",
    ))
    candle_fig.add_trace(go.Scatter(
        x=data.index, y=data["SMA_50"], mode="lines", name="SMA 50", line=dict(width=1.5)
    ))
    candle_fig.add_trace(go.Scatter(
        x=data.index, y=data["SMA_200"], mode="lines", name="SMA 200", line=dict(width=1.5)
    ))
    candle_fig.update_layout(
        title=f"{symbol} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=560,
    )

    # Line Chart
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(
        x=data.index, y=data["Close"], mode="lines", name="Close", line=dict(width=2)
    ))
    line_fig.add_trace(go.Scatter(
        x=data.index, y=data["SMA_50"], mode="lines", name="SMA 50", line=dict(width=1.5)
    ))
    line_fig.add_trace(go.Scatter(
        x=data.index, y=data["SMA_200"], mode="lines", name="SMA 200", line=dict(width=1.5)
    ))
    line_fig.update_layout(
        title=f"{symbol} Line Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=560,
    )

    # Toggle
    chart_type = st.radio(
        "Select Chart Type:",
        ["Candlestick", "Line Chart"],
        horizontal=True,
        key="chart_toggle"
    )
    if chart_type == "Line Chart":
        st.plotly_chart(line_fig, use_container_width=True)
    else:
        st.plotly_chart(candle_fig, use_container_width=True)

    # -----------------------------------
    # 📄 2. Latest Snapshot
    # -----------------------------------
    st.markdown("### 📄 Latest Snapshot")
    latest = data.iloc[-1]
    sma50 = latest["SMA_50"] if not pd.isna(latest["SMA_50"]) else "N/A"
    sma200 = latest["SMA_200"] if not pd.isna(latest["SMA_200"]) else "N/A"

    row1 = st.columns(4)
    row1[0].metric("Open", f"{latest['Open']:.2f}")
    row1[1].metric("Close", f"{latest['Close']:.2f}")
    row1[2].metric("High", f"{latest['High']:.2f}")
    row1[3].metric("Low", f"{latest['Low']:.2f}")

    row2 = st.columns(4)
    row2[0].metric("Volume", f"{int(latest['Volume']):,}")
    row2[1].metric("SMA 50", "N/A" if sma50 == "N/A" else f"{sma50:.2f}")
    row2[2].metric("SMA 200", "N/A" if sma200 == "N/A" else f"{sma200:.2f}")

    # -----------------------------------
    # 📊 3. Technical Indicators
    # -----------------------------------
    st.markdown("### 📊 Technical Indicators")

    try:
        rsi = ta.momentum.RSIIndicator(data["Close"]).rsi().iloc[-1]
        macd = ta.trend.MACD(data["Close"]).macd().iloc[-1]
        signal = ta.trend.MACD(data["Close"]).macd_signal().iloc[-1]
        bb_high = ta.volatility.BollingerBands(data["Close"]).bollinger_hband().iloc[-1]
        bb_low = ta.volatility.BollingerBands(data["Close"]).bollinger_lband().iloc[-1]
        adx = ta.trend.ADXIndicator(data["High"], data["Low"], data["Close"]).adx().iloc[-1]

        colA, colB, colC = st.columns(3)
        colA.metric("RSI (14)", f"{rsi:.2f}")
        colB.metric("MACD", f"{macd:.2f}")
        colC.metric("Signal Line", f"{signal:.2f}")

        colD, colE, colF = st.columns(3)
        colD.metric("Bollinger High", f"{bb_high:.2f}")
        colE.metric("Bollinger Low", f"{bb_low:.2f}")
        colF.metric("ADX", f"{adx:.2f}")
    except Exception:
        st.info("⚠️ Some indicators not available for this range.")

    # -----------------------------------
    # 🏢 4. Company Fundamentals
    # -----------------------------------
    st.markdown("### 🏢 Company Fundamentals")

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        col1, col2, col3 = st.columns(3)
        col1.metric("Market Cap", f"${info.get('marketCap', 0):,}")
        col2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
        col3.metric("Beta", f"{info.get('beta', 'N/A')}")

        col4, col5, col6 = st.columns(3)
        col4.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        col5.metric("52-Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")
        col6.metric("Dividend Yield", f"{round(info.get('dividendYield', 0)*100,2)}%")

        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}  |  **Industry:** {info.get('industry', 'N/A')}")
        st.markdown(f"**Profit Margin:** {info.get('profitMargins', 'N/A')}")
    except Exception:
        st.info("Some company data unavailable for this ticker.")


