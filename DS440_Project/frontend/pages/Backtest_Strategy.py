import os
import sys

# ------------------------------
# LOAD .env FIRST (before ANY AI imports)
# ------------------------------
from dotenv import load_dotenv
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
load_dotenv(ENV_PATH)

# Now your Google API key is available
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# ------------------------------
# Now import AI libraries
# ------------------------------
import google.generativeai as genai
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ------------------------------
# Now import the rest
# ------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import json
import yfinance as yf

# backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from backend.backtest_engine import run_backtest, fetch_ohlcv


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Backtest Strategy", layout="wide")

# -------------------------------------------------
# STYLES
# -------------------------------------------------
st.markdown(
    """
<style>
.section-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.result-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(39,238,245,0.35);
    padding: 15px;
    border-radius: 10px;
    margin-top: 8px;
}
.trade-item {
    padding: 8px 12px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.02);
    border-left: 3px solid #27EEF5;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("""
<style>
/* Hide the entire sidebar */
[data-testid="stSidebar"] {
    display: none;
}

/* Hide the sidebar toggle (arrow) */
[data-testid="stSidebarNav"] {
    display: none;
}

button[kind="header"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# HEADER
# -------------------------------------------------
col_back, _ = st.columns([1, 10])
with col_back:
    if st.button("← Home", use_container_width=True):
        st.switch_page("app.py")

st.markdown("<h1 style='color:#27EEF5;'>📈 Strategy Backtester</h1>", unsafe_allow_html=True)
st.divider()

if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY not set in your .env file. AI summaries will be disabled.")

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def extract_json_block(text: str):
    """
    Try to extract JSON from LLM output.
    Handles ```json ... ``` or plain JSON.
    """
    if not text:
        return None
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") or p.startswith("["):
                try:
                    return json.loads(p)
                except Exception:
                    continue
    try:
        return json.loads(text)
    except Exception:
        return None


def default_spec(symbol: str, cash: float):
    """
    Simple fallback SMA/RSI spec if the LLM fails.
    """
    return {
        "name": "Default SMA Cross Strategy",
        "stock_name": symbol,
        "cash": cash,
        "params": {
            "fast": 10,
            "slow": 30,
            "rsi": 14,
            "rsi_buy": 30,
            "rsi_sell": 70,
        },
        "entry": "crossover(SMA_fast, SMA_slow) and RSI < rsi_buy",
        "exit": "crossover(SMA_slow, SMA_fast) or RSI > rsi_sell",
    }


def text_to_spec_llm(strategy_text: str, symbol: str, cash: float):
    """
    Use Gemini to convert plain-English strategy text into
    a JSON spec compatible with backtest_engine.build_strategy_from_spec.
    """
    if not GOOGLE_API_KEY:
        return default_spec(symbol, cash)

    prompt = f"""
You are a trading-strategy JSON generator for a backtesting engine.

Convert the following trading strategy description into a JSON object
with the following structure ONLY (no extra text, no markdown):

{{
  "name": "short descriptive name",
  "stock_name": "{symbol}",
  "cash": {cash},
  "params": {{
    "fast": 10,
    "slow": 30,
    "rsi": 14,
    "rsi_buy": 30,
    "rsi_sell": 70
  }},
  "entry": "crossover(SMA_fast, SMA_slow) and RSI < rsi_buy",
  "exit": "crossover(SMA_slow, SMA_fast) or RSI > rsi_sell"
}}

Notes:
- "fast" and "slow" are SMA periods used in SMA_fast and SMA_slow.
- "rsi" is the RSI lookback length.
- "rsi_buy" is the RSI threshold or lower bound to consider oversold.
- "rsi_sell" is the RSI threshold or upper bound to consider overbought.
- "entry" and "exit" are Python-like expressions using:
  - crossover(SMA_fast, SMA_slow)
  - SMA_fast, SMA_slow
  - RSI
  - rsi_buy, rsi_sell
  - logical operators: and, or, <, >, <=, >=

The strategy description is:

\"\"\"{strategy_text}\"\"\""""

    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        resp = model.generate_content(prompt)
        spec = extract_json_block(resp.text)
        if not spec:
            return default_spec(symbol, cash)

        spec.setdefault("stock_name", symbol)
        spec.setdefault("cash", cash)

        if "params" not in spec:
            spec["params"] = default_spec(symbol, cash)["params"]

        fallback = default_spec(symbol, cash)
        spec.setdefault("entry", fallback["entry"])
        spec.setdefault("exit", fallback["exit"])
        return spec
    except Exception:
        return default_spec(symbol, cash)


def build_trade_summary(trades_df: pd.DataFrame, initial_capital: float):
    """
    Convert Backtesting.py trades DataFrame into a list of dicts with normalized keys.
    """
    if trades_df is None or trades_df.empty:
        return []

    cols_lower = [c.lower() for c in trades_df.columns]

    # PnL column
    pnl_col = None
    for cand in ["pnl", "pnl$", "pnl (inc. commission)"]:
        if cand.lower() in cols_lower:
            pnl_col = trades_df.columns[cols_lower.index(cand.lower())]
            break

    # Return % column
    ret_col = None
    for cand in ["return[%]", "return [%]", "returnpct", "return"]:
        if cand.lower().replace(" ", "") in [c.replace(" ", "") for c in cols_lower]:
            for orig, lowered in zip(trades_df.columns, cols_lower):
                if cand.lower().replace(" ", "") == lowered.replace(" ", ""):
                    ret_col = orig
                    break
            if ret_col:
                break

    # Size / shares
    size_col = None
    for cand in ["size", "shares", "qty"]:
        if cand.lower() in cols_lower:
            size_col = trades_df.columns[cols_lower.index(cand.lower())]
            break

    detailed = []
    for i, (_, row) in enumerate(trades_df.iterrows()):
        pnl = float(row[pnl_col]) if pnl_col else 0.0
        if ret_col:
            return_pct = float(row[ret_col])
        else:
            return_pct = (pnl / initial_capital) * 100 if initial_capital != 0 else 0.0
        shares = int(row[size_col]) if size_col and not pd.isna(row[size_col]) else 0

        detailed.append(
            {
                "trade_num": i + 1,
                "capital_used": float(initial_capital),
                "shares": shares,
                "return_dollars": pnl,
                "return_pct": return_pct,
            }
        )

    return detailed


def render_trade_lists(detailed_trades):
    """
    Show top 3 and bottom 3 trades by return %.
    """
    if not detailed_trades:
        st.info("No trades were made with this strategy.")
        return

    df = pd.DataFrame(detailed_trades)

    top3 = df.sort_values("return_pct", ascending=False).head(3)
    bottom3 = df.sort_values("return_pct", ascending=True).head(3)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### 🥇 Top 3 Trades (Highest Return %)")
        for _, r in top3.iterrows():
            st.markdown(
                f"""
                <div class="trade-item" style="border-left-color:#00C853;">
                    <b>Trade {int(r.trade_num)}</b><br>
                    Capital Used: ${r.capital_used:,.2f}<br>
                    Shares: {r.shares}<br>
                    Return: <b>${r.return_dollars:,.2f}</b> ({r.return_pct:.2f}%)
                </div>
                """,
                unsafe_allow_html=True,
            )

    with colB:
        st.markdown("### ⚠️ Bottom 3 Trades (Lowest Return %)")
        for _, r in bottom3.iterrows():
            st.markdown(
                f"""
                <div class="trade-item" style="border-left-color:#E53935;">
                    <b>Trade {int(r.trade_num)}</b><br>
                    Capital Used: ${r.capital_used:,.2f}<br>
                    Shares: {r.shares}<br>
                    Return: <b>${r.return_dollars:,.2f}</b> ({r.return_pct:.2f}%)
                </div>
                """,
                unsafe_allow_html=True,
            )


# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_saved, tab_custom = st.tabs(
    ["📂 Backtest Saved Strategies", "✍️ Backtest Your Own Strategy"]
)

# =================================================
# TAB 1 — BACKTEST SAVED STRATEGIES
# =================================================
with tab_saved:
    st.markdown("### 📁 Choose a Saved Strategy")

    saved_strategies = st.session_state.get("saved_strategies", {})

    if not saved_strategies:
        st.info("No saved strategies yet. Generate and save a strategy first.")
    else:
        strategy_names = list(saved_strategies.keys())
        chosen_name = st.selectbox("Select Strategy:", strategy_names)

        chosen_strategy = saved_strategies[chosen_name]
        strategy_text = chosen_strategy.get("text", "")
        base_json = chosen_strategy.get("json", {})

        # Only show name + small note (no full description here)
        st.markdown(
            f"""
            <div class="section-box">
                <h4 style='margin-bottom:5px;'>Selected Strategy: <code>{chosen_name}</code></h4>
                <p style='opacity:0.65; margin-bottom:0;'>
                    Using the configuration from the saved strategy. Full description is visible on the Strategy page.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input(
                "Stock Symbol",
                value=base_json.get("stock_name", "AAPL"),
                key="saved_symbol",
            )
        with col2:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                value=int(float(base_json.get("total_capital", "10000"))),
                step=500,
                key="saved_capital",
            )
        with col3:
            period = st.selectbox(
                "Data Period",
                ["6mo", "1y", "2y", "5y"],
                index=2,
                key="saved_period",
            )

        if st.button("Run Backtest (Saved Strategy)", type="primary"):
            with st.spinner("Running backtest..."):
                try:
                    # Convert saved plain-English text → structured spec for engine
                    spec = text_to_spec_llm(strategy_text, symbol, initial_capital)

                    # Fetch data to know number of points used
                    data = fetch_ohlcv(symbol, period=period, interval="1d")

                    bt_result, trades_df, equity_df = run_backtest(
                        symbol=symbol,
                        mode="spec",
                        strategy_spec_or_code=spec,
                        cash=initial_capital,
                        period=period,
                        interval="1d",
                    )

                    data_points = len(data)
                    total_trades = bt_result.get("trades", 0)
                    win_rate = bt_result.get("win_rate", 0.0)

                    st.markdown("## 📊 Backtest Results (Saved Strategy)")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Data Points Used", data_points)
                    c2.metric("Total Trades", total_trades)
                    c3.metric("Win Rate", f"{win_rate:.2f}%")

                    detailed_trades = build_trade_summary(trades_df, initial_capital)
                    render_trade_lists(detailed_trades)

                    # AI explanation
                    st.markdown("## 🧠 Backtest Summary (AI)")
                    if not GOOGLE_API_KEY:
                        st.warning("Set GOOGLE_API_KEY in your .env file to enable AI summaries.")
                    else:
                        summary_prompt = f"""
Summarize this trading backtest for a beginner trader.

Explain:
- What these results roughly mean
- Whether the strategy appears strong or weak compared to buy & hold
- Main risks or weaknesses to be aware of
- One or two ideas to improve the strategy

Strategy name: {spec.get('name', 'Unnamed Strategy')}
Symbol: {symbol}
Data Points: {data_points}
Total Trades: {total_trades}
Win Rate: {win_rate:.2f}%
Strategy Return [%]: {bt_result.get('return_pct', 0.0):.2f}
Buy & Hold Return [%]: {bt_result.get('buy_hold_return_pct', 0.0):.2f}
Max Drawdown [%]: {bt_result.get('max_drawdown_pct', 0.0):.2f}
"""
                        try:
                            model = genai.GenerativeModel("gemini-flash-latest")
                            resp = model.generate_content(summary_prompt)
                            st.write(resp.text)
                        except Exception as e:
                            st.error(f"AI summary failed: {e}")

                except Exception as e:
                    st.error(f"Backtest failed: {e}")

# =================================================
# TAB 2 — USER-WRITTEN STRATEGY (PLAIN ENGLISH)
# =================================================
with tab_custom:
    st.markdown("### ✍️ Write a Strategy to Backtest")

    user_text = st.text_area(
        "Describe your strategy in plain English:",
        height=180,
        placeholder=(
            "Example: Buy when the 10-day SMA crosses above the 30-day SMA and RSI is below 40.\n"
            "Sell when RSI goes above 70 or when the fast SMA crosses back below the slow SMA."
        ),
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        custom_symbol = st.text_input("Stock Symbol", "AAPL", key="custom_symbol")
    with col2:
        custom_capital = st.number_input(
            "Initial Capital ($)", min_value=1000, value=10000, step=500, key="custom_capital"
        )
    with col3:
        custom_period = st.selectbox(
            "Data Period", ["6mo", "1y", "2y", "5y"], index=1, key="custom_period"
        )

    if st.button("Run Backtest (Custom Strategy)", type="primary", key="btn_custom_bt"):
        if not user_text.strip():
            st.warning("Please enter a strategy description first.")
        else:
            with st.spinner("Converting strategy and running backtest..."):
                try:
                    custom_spec = text_to_spec_llm(user_text, custom_symbol, custom_capital)

                    data2 = fetch_ohlcv(custom_symbol, period=custom_period, interval="1d")

                    bt_res2, trades2_df, eq2_df = run_backtest(
                        symbol=custom_symbol,
                        mode="spec",
                        strategy_spec_or_code=custom_spec,
                        cash=custom_capital,
                        period=custom_period,
                        interval="1d",
                    )

                    data_points2 = len(data2)
                    total_trades2 = bt_res2.get("trades", 0)
                    win_rate2 = bt_res2.get("win_rate", 0.0)

                    st.markdown("## 📊 Backtest Results (Custom Strategy)")
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Data Points Used", data_points2)
                    cc2.metric("Total Trades", total_trades2)
                    cc3.metric("Win Rate", f"{win_rate2:.2f}%")

                    detailed_custom = build_trade_summary(trades2_df, custom_capital)
                    render_trade_lists(detailed_custom)

                    st.markdown("## 🧠 Backtest Summary (AI)")
                    if not GOOGLE_API_KEY:
                        st.warning("Set GOOGLE_API_KEY in your .env file to enable AI summaries.")
                    else:
                        summary_prompt2 = f"""
Explain these backtest results to a beginner trader in a friendly, clear way.

Original strategy description:
\"\"\"{user_text}\"\"\"

Symbol: {custom_symbol}
Data Points: {data_points2}
Total Trades: {total_trades2}
Win Rate: {win_rate2:.2f}%
Strategy Return [%]: {bt_res2.get('return_pct', 0.0):.2f}
Buy & Hold Return [%]: {bt_res2.get('buy_hold_return_pct', 0.0):.2f}
Max Drawdown [%]: {bt_res2.get('max_drawdown_pct', 0.0):.2f}

Give:
- A high-level summary of how the strategy performed
- Main strengths and weaknesses
- One or two suggestions to improve the rules
- A reminder that backtests do not guarantee future results.
"""
                        try:
                            model = genai.GenerativeModel("gemini-flash-latest")
                            resp2 = model.generate_content(summary_prompt2)
                            st.write(resp2.text)
                        except Exception as e:
                            st.error(f"AI summary failed: {e}")
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
