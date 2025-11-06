import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Data", layout="wide")

st.title("📊 Stock Data Viewer")

# --- User Input ---
col1, col2 = st.columns(2)
with col1:
    symbol = st.text_input("Enter a Stock Symbol (e.g. AAPL, TSLA, NVDA):", "AAPL")
with col2:
    period = st.selectbox("Select Time Period", ["1d", "1mo", "6mo", "1y", "5y"], index=1)

# --- Session State Setup ---
if "data" not in st.session_state:
    st.session_state.data = None
if "symbol" not in st.session_state:
    st.session_state.symbol = None

# --- Fetch Button ---
if st.button("Fetch Data"):
    if period == "1d":
        interval = "30m"
    elif period == "1mo":
        interval = "1h"
    else:
        interval = "1d"

    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        if data.empty:
            st.error("⚠️ No data found. Try a different symbol or longer period.")
            st.stop()

        # Save to session
        st.session_state.data = data
        st.session_state.symbol = symbol

        st.success(f"✅ Data fetched for {symbol}")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# --- Display Data and Chart ---
if st.session_state.data is not None:
    data = st.session_state.data
    symbol = st.session_state.symbol

    st.markdown(f"### 📄 Latest Data for {symbol}")
    st.dataframe(data.tail())

    # --- Moving Averages ---
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()

    # --- Charts ---
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
        x=data.index, y=data["SMA_50"], mode="lines", name="SMA 50", line=dict(color="blue", width=1.5)
    ))
    candle_fig.add_trace(go.Scatter(
        x=data.index, y=data["SMA_200"], mode="lines", name="SMA 200", line=dict(color="orange", width=1.5)
    ))
    candle_fig.update_layout(
        title=f"{symbol} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(
        x=data.index, y=data["Close"], mode="lines", name="Close", line=dict(color="white", width=2)
    ))
    line_fig.add_trace(go.Scatter(
        x=data.index, y=data["SMA_50"], mode="lines", name="SMA 50", line=dict(color="blue", width=1.5)
    ))
    line_fig.add_trace(go.Scatter(
        x=data.index, y=data["SMA_200"], mode="lines", name="SMA 200", line=dict(color="orange", width=1.5)
    ))
    line_fig.update_layout(
        title=f"{symbol} Line Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- Chart Toggle ---
    chart_type = st.radio(
        "Select Chart Type:",
        ["Candlestick", "Line Chart"],
        horizontal=True,
        key="chart_type_toggle"
    )

    if chart_type == "Line Chart":
        st.plotly_chart(line_fig, use_container_width=True, key="line_chart")
    else:
        st.plotly_chart(candle_fig, use_container_width=True, key="candle_chart")
