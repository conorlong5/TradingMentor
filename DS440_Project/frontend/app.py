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

# Optional simple auto-refresh
if "last_update" not in st.session_state:
    st.session_state["last_update"] = time.time()

current_time = time.time()
if current_time - st.session_state["last_update"] > 60:
    st.session_state["last_update"] = current_time
    st.rerun()

st.markdown(
    f"""
    <p style='text-align:center; color:gray; font-size:12px; margin-bottom:10px;'>
        Last updated: {datetime.now().strftime("%I:%M:%S %p")}
    </p>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# GLOBAL STYLES
# -------------------------------
st.markdown(
    """
<style>
/* Metric tweaks */
[data-testid="stMetricLabel"] {
    font-weight: 700;
    font-size: 15px;
    text-align: center;
}
[data-testid="stMetricValue"] {
    font-size: 26px;
    text-align: center;
}

/* Remove a bit of top padding */
.block-container {
    padding-top: 1.5rem;
}

/* Market index cards */
.index-card {
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.03);
    padding: 18px 20px 12px 20px;
    text-align: center;
}

/* App cards (for Explore section) */
.app-card {
    width: 100%;
    text-align: center;
    padding: 22px 10px 14px 10px;
    border-radius: 18px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.2s ease;
}

.app-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 22px rgba(39,238,245,0.28);
    border-color: rgba(39,238,245,0.7);
}

/* Caption text under icon/title */
.card-caption {
    text-align: center;
    font-size: 12px;
    color: #b0b0b0;
    margin-top: 4px;
}

/* Make our "Open" buttons full width and tidy */
.app-open-button > button {
    width: 100% !important;
    border-radius: 10px !important;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# 🏷️ HEADER / BRANDING
# -------------------------------
st.markdown(
    """
<h1 style='text-align:center; color:#27EEF5; margin-bottom:0;'>
    🟢 Trading Mentor Pro 🟢
</h1>
<p style='text-align:center; font-size:16px; margin-top:4px;'>
    AI-powered insights for smarter trading decisions
</p>
<hr style="margin-top:15px; margin-bottom:25px;">
""",
    unsafe_allow_html=True,
)

# -------------------------------
# 👋 GREETING
# -------------------------------
hour = datetime.now().hour
if hour < 12:
    greet = "Good morning"
elif hour < 18:
    greet = "Good afternoon"
else:
    greet = "Good evening"

st.markdown(
    f"<h3 style='text-align:center;'>👋 {greet}, Trader!</h3>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; font-size:12px; opacity:0.8;'>"
    "Welcome to your personal AI trading assistant — explore data, sentiment, and backtest your ideas below."
    "</p>",
    unsafe_allow_html=True,
)

# -------------------------------
# 📈 MARKET SNAPSHOT
# -------------------------------
def render_index_card(name: str, symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="7d")

        if len(hist) >= 2:
            last_close = hist["Close"].iloc[-1]
            prev_close = hist["Close"].iloc[-2]
            change = ((last_close - prev_close) / prev_close) * 100

            color = "#00C853" if change > 0 else "#E53935"
            arrow = "▲" if change > 0 else "▼"
            border_color = (
                "rgba(0, 200, 83, 0.4)" if change > 0 else "rgba(229, 57, 53, 0.4)"
            )

            prices = gaussian_filter1d(hist["Close"].values, sigma=1)
            fig, ax = plt.subplots(figsize=(3.2, 0.9))
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")
            ax.plot(prices, color=color, linewidth=2.4)
            ax.fill_between(
                range(len(prices)), prices, min(prices), color=color, alpha=0.25
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
            fig.savefig(buf, format="png", transparent=True, bbox_inches="tight")
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode()
            plt.close(fig)

            st.markdown(
                f"""
                <div class="index-card" style="border-color:{border_color};">
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.12em; color:#aaaaaa;">
                        {name}
                    </div>
                    <div style="font-size:24px; font-weight:700; margin-top:8px;">
                        {last_close:,.2f}
                    </div>
                    <div style="font-size:13px; color:{color}; margin-bottom:6px; font-weight:600;">
                        {arrow} {change:.2f}%
                    </div>
                    <img src="data:image/png;base64,{img_base64}" style="width:100%; max-width:260px;"/>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="index-card">
                    <div>{name}</div>
                    <div style="font-size:14px; color:gray;">N/A</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            f"""
            <div class="index-card">
                <div>{name}</div>
                <div style="font-size:14px; color:gray;">⚠️ Error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    "<h3 style='text-align:center; margin-top:16px;'>📊 Market Overview 📊</h3>",
    unsafe_allow_html=True,
)

idx_col1, idx_col2, idx_col3 = st.columns(3)
with idx_col1:
    render_index_card("S&P 500", "^GSPC")
with idx_col2:
    render_index_card("NASDAQ", "^IXIC")
with idx_col3:
    render_index_card("DOW JONES", "^DJI")

st.divider()

# -------------------------------
# 🚀 EXPLORE THE PLATFORM (2×2)
# -------------------------------
st.markdown(
    """
    <h3 style='text-align:center; margin-bottom:0;'>🚀 Explore the Platform 🚀</h3>
    <p style='text-align:center; font-size:12px; opacity:0.8; margin-top:4px;'>
        Choose a section to begin your analysis
    </p>
    """,
    unsafe_allow_html=True,
)

# Row 1: Stock Data | Sentiment Analysis
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# ---- Card 1: Stock Data ----
with row1_col1:
    st.markdown(
        """
        <div class="app-card">
            <img src="https://cdn-icons-png.flaticon.com/512/3132/3132086.png"
                 style="width:70px; margin-bottom:10px;"/>
            <h4 style="margin:4px 0 2px 0;">Stock Data</h4>
            <p class="card-caption">View charts, fundamentals, and key ratios.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # real Streamlit button, full width via CSS above
    btn_container = st.container()
    with btn_container:
        if st.button("Open", key="open_stock", use_container_width=True):
            st.switch_page("pages/Stock_Data.py")
    btn_container.markdown('<div class="app-open-button"></div>', unsafe_allow_html=True)

# ---- Card 2: Sentiment Analysis ----
with row1_col2:
    st.markdown(
        """
        <div class="app-card">
            <img src="https://cdn-icons-png.flaticon.com/512/3059/3059997.png"
                 style="width:70px; margin-bottom:10px;"/>
            <h4 style="margin:4px 0 2px 0;">Sentiment Analysis</h4>
            <p class="card-caption">Analyze recent market news sentiment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    btn_container = st.container()
    with btn_container:
        if st.button("Open", key="open_sentiment", use_container_width=True):
            st.switch_page("pages/Sentiment_Analysis.py")
    btn_container.markdown('<div class="app-open-button"></div>', unsafe_allow_html=True)

# ---- Card 3: Trading Strategy ----
with row2_col1:
    st.markdown(
        """
        <div class="app-card">
            <img src="https://cdn-icons-png.flaticon.com/512/4230/4230634.png"
                 style="width:70px; margin-bottom:10px;"/>
            <h4 style="margin:4px 0 2px 0;">Trading Strategy</h4>
            <p class="card-caption">Generate AI-powered trading ideas.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    btn_container = st.container()
    with btn_container:
        if st.button("Open", key="open_trading_strategy", use_container_width=True):
            st.switch_page("pages/Trading_Strategy.py")
    btn_container.markdown('<div class="app-open-button"></div>', unsafe_allow_html=True)

# ---- Card 4: Backtest Strategy ----
with row2_col2:
    st.markdown(
        """
        <div class="app-card">
            <img src="https://cdn-icons-png.flaticon.com/512/10913/10913166.png"
                 style="width:70px; margin-bottom:10px;"/>
            <h4 style="margin:4px 0 2px 0;">Backtest Strategy</h4>
            <p class="card-caption">Backtest saved or custom rules.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    btn_container = st.container()
    with btn_container:
        if st.button("Open", key="open_backtest", use_container_width=True):
            st.switch_page("pages/Backtest_Strategy.py")
    btn_container.markdown('<div class="app-open-button"></div>', unsafe_allow_html=True)

st.divider()

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
    f"<p style='text-align:center; color:white; font-style:italic; font-size:15px; margin-top:10px;'>"
    f"{random.choice(quotes)}</p>",
    unsafe_allow_html=True,
)
