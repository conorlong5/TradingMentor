import streamlit as st
import yfinance as yf
import random
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import base64
from io import BytesIO
import time

# -------------------------------
# 🎨 PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Trading Mentor Pro",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Simple auto-refresh of the top timestamp
if "last_update" not in st.session_state:
    st.session_state.last_update = time.time()

current_time = time.time()
if current_time - st.session_state.last_update > 60:
    st.session_state.last_update = current_time
    st.rerun()

st.markdown(
    f"""
    <p style='text-align:center; color:gray; font-size:12px; margin-bottom:10px;'>
        Last updated: {datetime.now().strftime('%I:%M:%S %p')}
    </p>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# GLOBAL STYLING
# -------------------------------
st.markdown(
    """
<style>
/* Metric styling */
[data-testid="stMetricLabel"] {
    font-weight: 700;
    font-size: 15px;
    text-align: center;
}
[data-testid="stMetricValue"] {
    font-size: 26px;
    text-align: center;
}

/* Remove extra padding at very top */
.block-container {
    padding-top: 1.5rem;
}

/* ===== Explore cards ===== */
.feature-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px 18px 16px 18px;
    text-align: center;
    transition: all 0.18s ease-in-out;
    box-shadow: 0 0 0 rgba(0,0,0,0);
}
.feature-card:hover {
    border-color: rgba(39, 238, 245, 0.7);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.35);
    transform: translateY(-3px);
}
.feature-icon {
    width: 56px;
    height: 56px;
    margin-bottom: 10px;
}
.feature-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 4px;
}
.feature-desc {
    font-size: 12px;
    color: #b0b0b0;
    margin-bottom: 10px;
}
.feature-btn {
    display: inline-block;
    margin-top: 4px;
}

/* Make the “Open” buttons compact */
button[data-testid="baseButton-secondary"] {
    padding: 4px 18px !important;
}

/* Center the quote text */
.footer-quote {
    text-align:center;
    color:#b0b0b0;
    font-style:italic;
    font-size: 11px;
    margin-top: 25px;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# HEADER / BRANDING
# -------------------------------
st.markdown(
    """
<h1 style='text-align:center; color:#27EEF5; margin-bottom:4px;'>
    🟢 Trading Mentor Pro 🟢
</h1>
<p style='text-align:center; font-size:16px; margin-top:0; color:#e0e0e0;'>
    AI-powered insights for smarter trading decisions
</p>
<hr style="margin-top:10px; margin-bottom:25px;">
""",
    unsafe_allow_html=True,
)

# -------------------------------
# PERSONALIZED GREETING
# -------------------------------
hour = datetime.now().hour
if hour < 12:
    greet = "Good morning"
elif hour < 18:
    greet = "Good afternoon"
else:
    greet = "Good evening"

st.markdown(
    f"<h4 style='text-align:center; margin-bottom:2px;'>👋 {greet}, Trader!</h4>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; font-size:12px; color:#b0b0b0;'>"
    "Welcome to your personal AI trading assistant — explore data, sentiment, and backtest your ideas below."
    "</p>",
    unsafe_allow_html=True,
)

# -------------------------------
# MARKET SNAPSHOT
# -------------------------------
def market_snapshot():
    st.markdown(
        "<h3 style='text-align:center; font-size:20px; margin-bottom:20px; color:#27EEF5;'>📊 Market Overview 📊</h3>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    def render_card(name, symbol, col):
        with col:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="7d")

                if len(hist) >= 2:
                    last_close = hist["Close"][-1]
                    prev_close = hist["Close"][-2]
                    change = ((last_close - prev_close) / prev_close) * 100

                    color = "#00C853" if change > 0 else "#E53935"
                    arrow = "▲" if change > 0 else "▼"
                    border_color = (
                        "rgba(0, 200, 83, 0.4)"
                        if change > 0
                        else "rgba(229, 57, 53, 0.4)"
                    )

                    prices = gaussian_filter1d(hist["Close"].values, sigma=1)
                    fig, ax = plt.subplots(figsize=(3.2, 0.9))
                    fig.patch.set_alpha(0.0)
                    ax.set_facecolor("none")
                    ax.plot(prices, color=color, linewidth=2.2)
                    ax.fill_between(
                        range(len(prices)),
                        prices,
                        min(prices),
                        color=color,
                        alpha=0.18,
                    )
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    ax.tick_params(
                        left=False,
                        bottom=False,
                        labelleft=False,
                        labelbottom=False,
                    )
                    ax.grid(False)

                    buf = BytesIO()
                    fig.savefig(
                        buf, format="png", transparent=True, bbox_inches="tight", dpi=130
                    )
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode()
                    plt.close(fig)

                    st.markdown(
                        f"""
                        <div style='
                            border: 2px solid {border_color};
                            border-radius: 16px;
                            padding: 18px 15px 14px 15px;
                            background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
                            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25), 0 0 15px {border_color.replace("0.4", "0.15")};
                            text-align: center;
                            margin-bottom: 10px;
                        '>
                            <div style='text-align:center; line-height:1.4; margin-top:3px;'>
                                <div style='font-weight:600; font-size:14px; color: #b0b0b0; text-transform: uppercase; letter-spacing: 0.05em;'>{name}</div>
                                <div style='font-size:24px; font-weight:700; margin: 6px 0;'>{last_close:,.2f}</div>
                                <div style='font-size:13px; color:{color}; font-weight: 600; margin-bottom: 8px;'>{arrow} {change:.2f}%</div>
                            </div>
                            <img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 260px; margin-top: 4px;">
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='text-align:center; font-size:14px; color:gray; "
                        f"border: 1px solid rgba(255,255,255,0.1); border-radius:12px; padding:20px;'>{name}<br>N/A</div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.markdown(
                    f"<div style='text-align:center; font-size:14px; color:gray; "
                    f"border: 1px solid rgba(255,255,255,0.1); border-radius:12px; padding:20px;'>{name}<br>⚠️ Error</div>",
                    unsafe_allow_html=True,
                )

    render_card("S&P 500", "^GSPC", col1)
    render_card("NASDAQ", "^IXIC", col2)
    render_card("DOW JONES", "^DJI", col3)


market_snapshot()
st.divider()

# -------------------------------
# 🚀 EXPLORE THE PLATFORM – 2×2 GRID
# -------------------------------
st.markdown(
    """
    <h3 style='text-align:center; font-size:20px; color:#27EEF5; margin-bottom:4px;'>🚀 Explore the Platform 🚀</h3>
    <p style='text-align:center; font-size:12px; color:#b0b0b0; margin-bottom:20px;'>
        Choose a section to begin your analysis
    </p>
    """,
    unsafe_allow_html=True,
)

# Row 1
row1_col1, row1_col2 = st.columns(2)
# Row 2
row2_col1, row2_col2 = st.columns(2)

# --- Card 1: Stock Data ---
with row1_col1:
    st.markdown(
        """
        <div class="feature-card">
            <img class="feature-icon" src="https://cdn-icons-png.flaticon.com/512/3132/3132086.png">
            <div class="feature-title">Stock Data</div>
            <div class="feature-desc">
                View charts, fundamentals,<br>and key ratios.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open", key="btn_stock_data"):
        st.switch_page("pages/Stock_Data.py")

# --- Card 2: Sentiment Analysis ---
with row1_col2:
    st.markdown(
        """
        <div class="feature-card">
            <img class="feature-icon" src="https://cdn-icons-png.flaticon.com/512/3059/3059997.png">
            <div class="feature-title">Sentiment Analysis</div>
            <div class="feature-desc">
                Analyze recent news sentiment<br>around your favorite stocks.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open", key="btn_sentiment"):
        st.switch_page("pages/Sentiment_Analysis.py")

# --- Card 3: Trading Strategy ---
with row2_col1:
    st.markdown(
        """
        <div class="feature-card">
            <img class="feature-icon" src="https://cdn-icons-png.flaticon.com/512/4230/4230634.png">
            <div class="feature-title">Trading Strategy</div>
            <div class="feature-desc">
                Generate AI-powered trading ideas<br>tailored to your style.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open", key="btn_trading_strategy"):
        st.switch_page("pages/Trading_Strategy.py")

# --- Card 4: Backtest Strategy ---
with row2_col2:
    st.markdown(
        """
        <div class="feature-card">
            <img class="feature-icon" src="https://cdn-icons-png.flaticon.com/512/9841/9841552.png">
            <div class="feature-title">Backtest Strategy</div>
            <div class="feature-desc">
                Backtest saved or custom rules<br>on historical data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Open", key="btn_backtest_strategy"):
        st.switch_page("pages/Backtest_Strategy.py")

# -------------------------------
# 💬 QUOTE / TIP OF THE DAY
# -------------------------------
quotes = [
    "“The key to making money in stocks is not to get scared out of them.” – Peter Lynch",
    "“Amateurs want to be right. Professionals want to make money.” – Alan Greenspan",
    "“Risk comes from not knowing what you’re doing.” – Warren Buffett",
    "“The four most dangerous words in investing are: This time it's different.” – Sir John Templeton",
    "“An investment in knowledge pays the best interest.” – Benjamin Franklin",
]

st.markdown(
    f"<p class='footer-quote'>{random.choice(quotes)}</p>",
    unsafe_allow_html=True,
)
