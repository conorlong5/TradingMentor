import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta

st.set_page_config(page_title="Stock Data", layout="wide")

# ----------------------------
# CUSTOM STYLING
# ----------------------------
st.markdown("""
<style>
/* Card-style containers for sections */
.metric-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}

/* Enhance metric displays */
[data-testid="stMetricValue"] {
    font-size: 24px;
    font-weight: 600;
}

/* Button hover effects */
button[data-testid="baseButton-secondary"]:hover {
    background-color: rgba(0, 191, 255, 0.2);
    border-color: #00BFFF;
    transition: all 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER WITH BACK BUTTON
# ----------------------------
col_back, col_title = st.columns([1, 11])
with col_back:
    if st.button("← Home"):
        st.switch_page("Home.py")
with col_title:
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
    with st.spinner("Fetching data..."):
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
    # SIDEBAR WITH QUICK STATS
    # -----------------------------------
    with st.sidebar:
        st.markdown("### 📊 Quick Stats")
        latest = data.iloc[-1]
        st.metric("Current Price", f"${latest['Close']:.2f}")
        st.metric("Day Range", f"${latest['Low']:.2f} - ${latest['High']:.2f}")
        st.metric("Volume", f"{int(latest['Volume']):,}")
        
        # Price change
        if len(data) >= 2:
            prev_close = data.iloc[-2]["Close"]
            change = latest["Close"] - prev_close
            change_pct = (change / prev_close) * 100
            st.metric("Change", f"${change:.2f}", f"{change_pct:+.2f}%")

    # -----------------------------------
    # 📈 STOCK CHART WITH VOLUME
    # -----------------------------------
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Stock Chart with Volume")

    # Toggle for chart type
    chart_type = st.radio(
        "Select Chart Type:",
        ["Candlestick", "Line Chart"],
        horizontal=True,
        key="chart_toggle"
    )

    # Import for subplots
    from plotly.subplots import make_subplots
    
    # Create volume bar colors
    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' 
              for i in range(len(data))]

    # Create figure with subplots (2 rows: price and volume)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f"{symbol} Price", "Volume")
    )

    if chart_type == "Candlestick":
        # Add candlestick to top subplot
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color="green",
            decreasing_line_color="red",
        ), row=1, col=1)
    else:
        # Add line chart to top subplot
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Close",
            line=dict(width=2, color="#00BFFF")
        ), row=1, col=1)

    # Add SMAs to top subplot
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["SMA_50"],
        mode="lines",
        name="SMA 50",
        line=dict(width=1.5, color="orange")
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["SMA_200"],
        mode="lines",
        name="SMA 200",
        line=dict(width=1.5, color="cyan")
    ), row=1, col=1)

    # Add volume bars to bottom subplot
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=colors,
        name='Volume',
        opacity=0.7,
        showlegend=False
    ), row=2, col=1)

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------
    # 📄 LATEST SNAPSHOT
    # -----------------------------------
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Latest Snapshot")
    
    latest = data.iloc[-1]
    
    # Price change display
    if len(data) >= 2:
        prev_close = data.iloc[-2]["Close"]
        current_close = latest["Close"]
        change = current_close - prev_close
        change_pct = (change / prev_close) * 100
        
        st.markdown(f"""
            <div style='text-align:center; padding:15px; border-radius:8px; margin-bottom:20px;
                 background-color: {"rgba(0, 200, 83, 0.2)" if change >= 0 else "rgba(229, 57, 53, 0.2)"};
                 border: 1px solid {"rgba(0, 200, 83, 0.4)" if change >= 0 else "rgba(229, 57, 53, 0.4)"};'>
                <span style='font-size:28px; font-weight:bold; color:{"#00C853" if change >= 0 else "#E53935"};'>
                    {"▲" if change >= 0 else "▼"} ${abs(change):.2f} ({change_pct:+.2f}%)
                </span>
                <br>
                <span style='font-size:12px; color:gray;'>Change from previous close</span>
            </div>
        """, unsafe_allow_html=True)
    
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
    
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------
    # 🎯 TRADING SIGNALS
    # -----------------------------------
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Trading Signals")
    
    signals = []
    if not pd.isna(latest["SMA_50"]) and not pd.isna(latest["SMA_200"]):
        if latest["SMA_50"] > latest["SMA_200"]:
            signals.append("🟢 **Golden Cross:** SMA 50 > SMA 200 (Bullish)")
        else:
            signals.append("🔴 **Death Cross:** SMA 50 < SMA 200 (Bearish)")
    
    try:
        rsi = ta.momentum.RSIIndicator(data["Close"]).rsi().iloc[-1]
        if rsi > 70:
            signals.append("⚠️ **RSI Overbought** (>70) - Potential reversal")
        elif rsi < 30:
            signals.append("⚠️ **RSI Oversold** (<30) - Potential bounce")
        else:
            signals.append("✅ **RSI Neutral** - No extreme conditions")
    except:
        pass
    
    for signal in signals:
        st.markdown(f"- {signal}")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------
    # 📊 TECHNICAL INDICATORS
    # -----------------------------------
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
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
    
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------
    # 🏢 COMPANY FUNDAMENTALS
    # -----------------------------------
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
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
    
    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------
    # 💾 EXPORT DATA
    # -----------------------------------
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### 💾 Export Data")
    
    csv = data.to_csv()
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"{symbol}_data_{period}.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)