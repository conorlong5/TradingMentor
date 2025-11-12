import streamlit as st
import yfinance as yf
import random
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import gaussian_filter1d


# -------------------------------
# 🎨 PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Trading Mentor Pro",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="stMetricLabel"] {
    font-weight: 700;
    font-size: 15px;
    text-align: center;
}
[data-testid="stMetricValue"] {
    font-size: 26px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Center all Streamlit page_link buttons */
div[data-testid="stPageLink"] {
    display: flex;
    justify-content: center; /* centers horizontally */
    align-items: center;      /* centers vertically */
    margin-top: 6px;
    margin-bottom: 4px;
}

/* Optional: style button to match your dark theme */
div[data-testid="stPageLink"] button {
    background-color: rgba(255, 255, 255, 0.06);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 6px 18px;
    font-weight: 500;
    transition: all 0.2s ease-in-out;
}
div[data-testid="stPageLink"] button:hover {
    background-color: rgba(255, 255, 255, 0.15);
}
</style>
""", unsafe_allow_html=True)



# -------------------------------
# ✏️ GLOBAL STYLING (metrics + center text)
# -------------------------------
st.markdown("""
<style>
/* Make metrics larger & cleaner */
[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: 600;
}
[data-testid="stMetricLabel"] {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Center captions used in cards */
.card-caption {
    text-align: center;
    font-size: 12px;
    color: #b0b0b0;
    margin-top: 4px;
}

/* Remove extra padding at very top */
.block-container {
    padding-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🏷️ HEADER / BRANDING
# -------------------------------
st.markdown("""
<h1 style='text-align:center; color:#00BFFF; margin-bottom:0;'>
    Trading Mentor Pro 💼
</h1>
<p style='text-align:center; font-size:30px; margin-top:0;'>
    AI-powered insights for smarter trading decisions
</p>
<hr style="margin-top:15px; margin-bottom:25px;">
""", unsafe_allow_html=True)

# -------------------------------
# 👋 PERSONALIZED GREETING
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
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Welcome to your personal AI trading assistant — explore data, sentiment, and backtest your ideas below.</p>",
    unsafe_allow_html=True
)
st.divider()

# -------------------------------
# 📈 MARKET SNAPSHOT (Centered + Larger)
# -------------------------------
def market_snapshot():
    st.markdown(
        "<h3 style='text-align:center;'>📊 Market Overview</h3>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([0.5, 0.5, 4, 2, 4, 2, 4, 0.5, 0.5])

    def render_card(name, symbol):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="7d")

            if len(hist) >= 2:
                last_close = hist["Close"][-1]
                prev_close = hist["Close"][-2]
                change = ((last_close - prev_close) / prev_close) * 100

                color = "#00C853" if change > 0 else "#E53935"
                arrow = "▲" if change > 0 else "▼"

                # --- Main Info Card ---
                st.markdown(
                    f"""
                    <div style='text-align:center; line-height:1.4; margin-top:10px;'>
                        <div style='font-weight:600; font-size:16px;'>{name}</div>
                        <div style='font-size:24px; font-weight:700;'>{last_close:,.2f}</div>
                        <div style='font-size:14px; color:{color};'>{arrow} {change:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # --- Enhanced Stock-Like Sparkline ---
                prices = gaussian_filter1d(hist["Close"].values, sigma=1)
                fig, ax = plt.subplots(figsize=(3, 0.8))
                fig.patch.set_alpha(0.0)  # Transparent figure background
                ax.set_facecolor("none")  # Transparent plot background

                prices = hist["Close"].values
                ax.plot(prices, color=color, linewidth=2.0)

                # Add subtle filled gradient under the line
                ax.fill_between(
                    range(len(prices)),
                    prices,
                    min(prices),
                    color=color,
                    alpha=0.15
                )

                # Remove borders, ticks, and grid
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax.grid(False)

                st.pyplot(fig, transparent=True)

            else:
                st.markdown(
                    f"<div style='text-align:center; font-size:14px; color:gray;'>{name}<br>N/A</div>",
                    unsafe_allow_html=True
                )

        except Exception:
            st.markdown(
                f"<div style='text-align:center; font-size:14px; color:gray;'>{name}<br>⚠️ Error</div>",
                unsafe_allow_html=True
            )

    with col3:
        render_card("S&P 500", "^GSPC")

    with col5:
        render_card("NASDAQ", "^IXIC")

    with col7:
        render_card("DOW JONES", "^DJI")
market_snapshot()
st.divider()

# -------------------------------
# 🚀 EXPLORE PLATFORM SECTION (final version)
# -------------------------------

st.markdown(
    """
    <h2 style='text-align:center;'>🚀 Explore the Platform</h2>
    <p style='text-align:center;'>Choose a section to begin your analysis</p>
    """,
    unsafe_allow_html=True
)

col1, col2, col3, col4, col5 = st.columns([0.5, 2, 2, 2, 0.5])


def centered_card(image_url, page_path, label, caption):
    st.markdown(
        f"""
        <div style='
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            line-height: 1.4;
            margin-top: 10px;
            border-radius: 12px;
            padding: 14px 0;
            transition: all 0.25s ease-in-out;
        ' onmouseover="this.style.background='rgba(255,255,255,0.05)';" onmouseout="this.style.background='transparent';">
            <img src="{image_url}" style="width:70px; margin-bottom:10px;"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Use Streamlit's internal page navigation — this is what fixes your redirect issue
    st.page_link(page_path, label=label)
    st.markdown(
        f"<p style='text-align:center; font-size:12px; color:gray; max-width:200px; margin:auto;'>{caption}</p>",
        unsafe_allow_html=True
    )

# --- STOCK DATA ---
with col2:
    centered_card(
        "https://cdn-icons-png.flaticon.com/512/3132/3132086.png",
        "pages/Stock_Data.py",
        "Stock Data",
        "View technical charts, fundamentals, and key ratios."
    )

# --- SENTIMENT ANALYSIS ---
with col3:
    centered_card(
        "https://cdn-icons-png.flaticon.com/512/3059/3059997.png",
        "pages/Sentiment_Analysis.py",
        "Sentiment Analysis",
        "Analyze recent news sentiment around your favorite stocks."
    )

# --- TRADING STRATEGY ---
with col4:
    centered_card(
        "https://cdn-icons-png.flaticon.com/512/4230/4230634.png",
        "pages/Trading_Strategy.py",
        "Trading Strategy",
        "Generate and backtest AI-powered trading ideas."
    )






# -------------------------------
# 💬 QUOTE / TIP OF THE DAY
# -------------------------------
quotes = [
    "“The key to making money in stocks is not to get scared out of them.” – Peter Lynch",
    "“Amateurs want to be right. Professionals want to make money.” – Alan Greenspan",
    "“Risk comes from not knowing what you’re doing.” – Warren Buffett",
    "“The four most dangerous words in investing are: This time it's different.” – Sir John Templeton",
    "“An investment in knowledge pays the best interest.” – Benjamin Franklin"
]
st.markdown(
    f"<p style='text-align:center; color:lightgray; font-style:italic;'>{random.choice(quotes)}</p>",
    unsafe_allow_html=True
)

# -------------------------------
# ⚙️ FOOTER
# -------------------------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
Built with ❤️ using Streamlit | © 2025 Trading Mentor Pro
</p>
""", unsafe_allow_html=True)
