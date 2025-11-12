import streamlit as st
import yfinance as yf
import random
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    initial_sidebar_state="collapsed"
)

if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# Check if 60 seconds have passed since last update
current_time = time.time()
if current_time - st.session_state.last_update > 60:  # Change 60 to your preferred seconds
    st.session_state.last_update = current_time
    st.rerun()

# Optional: Show last update time
st.markdown(f"""
    <p style='text-align:center; color:gray; font-size:12px; margin-bottom:10px;'>
        Last updated: {datetime.now().strftime('%I:%M:%S %p')}
    </p>
""", unsafe_allow_html=True)

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
<h1 style='text-align:center; color:#27EEF5; margin-bottom:0;'>
    💹 Trading Mentor Pro 💹
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
        "<h3 style='text-align:center; font-size:50px; margin-bottom:30px; color:#27EEF5; '>📊 Market Overview 📊</h3>",
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
                
                # Dynamic border color based on stock performance
                border_color = "rgba(0, 200, 83, 0.4)" if change > 0 else "rgba(229, 57, 53, 0.4)"

                # --- Create the sparkline chart ---
                prices = gaussian_filter1d(hist["Close"].values, sigma=1)
                fig, ax = plt.subplots(figsize=(3, 0.8))
                fig.patch.set_alpha(0.0)
                ax.set_facecolor("none")
                ax.plot(prices, color=color, linewidth=2.5)
                ax.fill_between(range(len(prices)), prices, min(prices), color=color, alpha=0.2)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax.grid(False)
                
                # Convert plot to base64 image
                buf = BytesIO()
                fig.savefig(buf, format='png', transparent=True, bbox_inches='tight')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode()
                plt.close(fig)

                # 🟢 Complete bordered box with embedded chart
                st.markdown(f"""
                    <div style='
                        border: 2px solid {border_color};
                        border-radius: 16px;
                        padding: 20px 15px 15px 15px;
                        background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
                        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25), 0 0 15px {border_color.replace("0.4", "0.15")};
                        text-align: center;
                        margin-bottom: 10px;
                    '>
                        <div style='text-align:center; line-height:1.4; margin-top:5px;'>
                            <div style='font-weight:600; font-size:16px; color: #b0b0b0; text-transform: uppercase; letter-spacing: 0.05em;'>{name}</div>
                            <div style='font-size:28px; font-weight:700; margin: 8px 0;'>{last_close:,.2f}</div>
                            <div style='font-size:15px; color:{color}; font-weight: 600; margin-bottom: 10px;'>{arrow} {change:.2f}%</div>
                        </div>
                        <img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 250px; margin-top: 5px;">
                    </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(
                    f"<div style='text-align:center; font-size:14px; color:gray; border: 1px solid rgba(255,255,255,0.1); border-radius:12px; padding:20px;'>{name}<br>N/A</div>",
                    unsafe_allow_html=True
                )

        except Exception:
            st.markdown(
                f"<div style='text-align:center; font-size:14px; color:gray; border: 1px solid rgba(255,255,255,0.1); border-radius:12px; padding:20px;'>{name}<br>⚠️ Error</div>",
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

st.markdown("""
<style>
/* Make the platform navigation buttons more compact */
button[data-testid="baseButton-secondary"] {
    padding: 8px 20px !important;
    width: auto !important;
    min-width: 120px !important;
    max-width: 200px !important;
    margin: 0 auto !important;
    display: block !important;
}

/* Center the button container */
[data-testid="column"] > div > div > div > button {
    margin-left: auto !important;
    margin-right: auto !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 🚀 EXPLORE PLATFORM SECTION
# -------------------------------

st.markdown(
    """
    <h2 style='text-align:center; font-size:50px; color:#27EEF5; '>🚀 Explore the Platform 🚀</h2>
    <p style='text-align:center; margin-bottom:30px;'>Choose a section to begin your analysis</p>
    """,
    unsafe_allow_html=True
)

# Create hidden navigation buttons
col1, col2, col3 = st.columns(3)

nav_choice = None

# Style the buttons and add the cards on top
st.markdown("""
<style>
/* Move buttons up and add cards styling */
[data-testid="column"] {
    position: relative;
}

/* Style the actual Streamlit buttons */
button[data-testid="baseButton-secondary"] {
    margin-top: -20px;
}
</style>
""", unsafe_allow_html=True)

# Add visual cards above buttons using separate markdown calls
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Stock Data", key="stock_data_btn", use_container_width=True):
        st.switch_page("pages/Stock_Data.py")
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <img src="https://cdn-icons-png.flaticon.com/512/3132/3132086.png" style="width:70px; margin-bottom:10px;"/>
            <p style='font-size:20px; color:white; margin-top:10px;'>View technical charts, fundamentals,<br>and key ratios.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    if st.button("Sentiment Analysis", key="sentiment_btn", use_container_width=True):
        st.switch_page("pages/Sentiment_Analysis.py")
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <img src="https://cdn-icons-png.flaticon.com/512/3059/3059997.png" style="width:70px; margin-bottom:10px;"/>
            <p style='font-size:20px; color:white; margin-top:10px;'>Analyze recent news sentiment around<br>your favorite stocks.</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    if st.button("Trading Strategy", key="strategy_btn", use_container_width=True):
        st.switch_page("pages/Trading_Strategy.py")
    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <img src="https://cdn-icons-png.flaticon.com/512/4230/4230634.png" style="width:70px; margin-bottom:10px;"/>
            <p style='font-size:20px; color:white; margin-top:10px;'>Generate and backtest AI-powered<br>trading ideas.</p>
        </div>
    """, unsafe_allow_html=True)

st.divider()

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
    f"<p style='text-align:center; color:white; font-style:italic;'>{random.choice(quotes)}</p>",
    unsafe_allow_html=True
)


