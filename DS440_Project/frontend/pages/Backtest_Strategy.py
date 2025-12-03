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
from typing import Union
from components.ai_drawer import render_ai_drawer

# backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from backend.backtest_engine import run_backtest, fetch_ohlcv

# Helper function for safe JSON serialization
def _safe_json(obj):
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        if isinstance(obj, dict):
            return json.dumps({k: str(v) for k, v in obj.items()}, indent=2)
        return str(obj)


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
    Try to extract a JSON object from a Gemini response.
    - Strips markdown fences
    - Takes the substring from the first '{' to the last '}'.
    Returns None if no JSON-looking block is found.
    """
    if not text:
        return None
    
    # Strip code fences if present
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("` \n")
        if t.lower().startswith("json"):
            t = t[4:].strip()
    
    start = t.find("{")
    end = t.rfind("}")
    
    if start == -1 or end == -1 or end <= start:
        return None
    
    try:
        return json.loads(t[start : end + 1].strip())
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


def get_min_trades_for_period(period: str) -> int:
    """
    Returns minimum expected trades based on the data period.
    """
    min_trades_map = {
        "6mo": 5,
        "1y": 10,
        "2y": 15,
        "5y": 25,
        "max": 30,  # For maximum historical data, expect at least 30 trades
    }
    return min_trades_map.get(period, 10)


def adjust_spec_for_more_trades(spec: dict, adjustment_level: int = 1) -> dict:
    """
    Makes strategy more sensitive to generate more trades by adjusting parameters.
    adjustment_level: 1 = mild, 2 = moderate, 3 = aggressive
    """
    adjusted_spec = spec.copy()
    params = adjusted_spec.get("params", {}).copy()
    
    # Adjust RSI thresholds to be more sensitive (wider range = more signals)
    if adjustment_level == 1:
        # Mild: Expand RSI buy range, tighten sell range
        params["rsi_buy"] = max(20, params.get("rsi_buy", 30) - 10)
        params["rsi_sell"] = min(80, params.get("rsi_sell", 70) + 10)
        # Make SMAs closer together for more crossovers
        params["fast"] = max(5, params.get("fast", 10) - 2)
        params["slow"] = min(50, params.get("slow", 30) + 5)
    elif adjustment_level == 2:
        # Moderate: More aggressive adjustments
        params["rsi_buy"] = max(15, params.get("rsi_buy", 30) - 15)
        params["rsi_sell"] = min(85, params.get("rsi_sell", 70) + 15)
        params["fast"] = max(5, params.get("fast", 10) - 3)
        params["slow"] = min(40, params.get("slow", 30) + 3)
    else:
        # Aggressive: Very sensitive
        params["rsi_buy"] = max(10, params.get("rsi_buy", 30) - 20)
        params["rsi_sell"] = min(90, params.get("rsi_sell", 70) + 20)
        params["fast"] = max(5, params.get("fast", 10) - 4)
        params["slow"] = min(35, params.get("slow", 30) + 2)
    
    adjusted_spec["params"] = params
    
    # Adjust entry/exit conditions to be less restrictive
    entry = adjusted_spec.get("entry", "")
    exit_expr = adjusted_spec.get("exit", "")
    
    # Make entry easier (lower RSI threshold in condition)
    if "RSI < rsi_buy" in entry:
        entry = entry.replace("RSI < rsi_buy", "RSI <= rsi_buy")
    elif "RSI > rsi_buy" in entry:
        entry = entry.replace("RSI > rsi_buy", "RSI >= rsi_buy")
    
    # Make exit easier (higher RSI threshold in condition)
    if "RSI > rsi_sell" in exit_expr:
        exit_expr = exit_expr.replace("RSI > rsi_sell", "RSI >= rsi_sell")
    elif "RSI < rsi_sell" in exit_expr:
        exit_expr = exit_expr.replace("RSI < rsi_sell", "RSI <= rsi_sell")
    
    adjusted_spec["entry"] = entry
    adjusted_spec["exit"] = exit_expr
    
    return adjusted_spec


def analyze_time_based_performance(trades_df: pd.DataFrame, equity_df: pd.DataFrame, data_start_date, data_end_date) -> str:
    """
    Analyzes strategy performance over different time periods.
    Returns a description of how the strategy performed in early vs recent periods.
    """
    if trades_df is None or trades_df.empty or equity_df is None or equity_df.empty:
        return "Insufficient data for time-based analysis."
    
    try:
        # Get date range
        if hasattr(data_start_date, 'strftime'):
            start_year = data_start_date.year
        else:
            start_year = pd.to_datetime(data_start_date).year if data_start_date else None
            
        if hasattr(data_end_date, 'strftime'):
            end_year = data_end_date.year
        else:
            end_year = pd.to_datetime(data_end_date).year if data_end_date else None
        
        # Calculate midpoint
        if start_year and end_year:
            total_years = end_year - start_year
            midpoint_year = start_year + (total_years // 2)
        else:
            midpoint_year = None
        
        # Analyze trades by time period
        # Try to get date column from trades
        trade_dates = None
        if 'EntryTime' in trades_df.columns:
            trade_dates = pd.to_datetime(trades_df['EntryTime'])
        elif 'ExitTime' in trades_df.columns:
            trade_dates = pd.to_datetime(trades_df['ExitTime'])
        elif trades_df.index.name == 'Date' or isinstance(trades_df.index, pd.DatetimeIndex):
            trade_dates = trades_df.index
        
        if trade_dates is not None and midpoint_year:
            # Split into first half and second half
            midpoint_date = pd.Timestamp(f"{midpoint_year}-01-01")
            first_half = trades_df[trade_dates < midpoint_date]
            second_half = trades_df[trade_dates >= midpoint_date]
            
            if len(first_half) > 0 and len(second_half) > 0:
                # Calculate win rates for each period
                # Try to find PnL or return column
                pnl_col = None
                for col in trades_df.columns:
                    if 'pnl' in col.lower() or 'return' in col.lower():
                        pnl_col = col
                        break
                
                if pnl_col:
                    first_half_wins = (first_half[pnl_col] > 0).sum()
                    first_half_total = len(first_half)
                    first_half_win_rate = (first_half_wins / first_half_total * 100) if first_half_total > 0 else 0
                    
                    second_half_wins = (second_half[pnl_col] > 0).sum()
                    second_half_total = len(second_half)
                    second_half_win_rate = (second_half_wins / second_half_total * 100) if second_half_total > 0 else 0
                    
                    # Calculate average returns
                    first_half_avg_return = first_half[pnl_col].mean() if len(first_half) > 0 else 0
                    second_half_avg_return = second_half[pnl_col].mean() if len(second_half) > 0 else 0
                    
                    # Build description
                    description = f"""
Time Period Analysis:
- First Half ({start_year}-{midpoint_year-1}): {first_half_total} trades, {first_half_win_rate:.1f}% win rate, avg return: ${first_half_avg_return:.2f}
- Second Half ({midpoint_year}-{end_year}): {second_half_total} trades, {second_half_win_rate:.1f}% win rate, avg return: ${second_half_avg_return:.2f}
"""
                    
                    # Add interpretation
                    if first_half_win_rate > second_half_win_rate + 5 and first_half_avg_return > second_half_avg_return:
                        description += "\nThe strategy performed better in the earlier period (first half) compared to recent years."
                    elif second_half_win_rate > first_half_win_rate + 5 and second_half_avg_return > first_half_avg_return:
                        description += "\nThe strategy performed better in the recent period (second half) compared to earlier years."
                    else:
                        description += "\nThe strategy showed relatively consistent performance across both time periods."
                    
                    return description
        
        return "Time-based analysis: Data spans from {} to {}. Strategy performance appears relatively consistent across the period.".format(
            start_year if start_year else "unknown",
            end_year if end_year else "unknown"
        )
    except Exception as e:
        return f"Time-based analysis unavailable: {str(e)}"


def describe_parameter_changes(original_spec: dict, adjusted_spec: dict) -> str:
    """
    Creates a human-readable description of what parameters changed.
    """
    orig_params = original_spec.get("params", {})
    adj_params = adjusted_spec.get("params", {})
    changes = []
    
    # Compare parameters
    if orig_params.get("fast") != adj_params.get("fast"):
        changes.append(f"SMA Fast period: {orig_params.get('fast')} → {adj_params.get('fast')}")
    
    if orig_params.get("slow") != adj_params.get("slow"):
        changes.append(f"SMA Slow period: {orig_params.get('slow')} → {adj_params.get('slow')}")
    
    if orig_params.get("rsi_buy") != adj_params.get("rsi_buy"):
        changes.append(f"RSI Buy threshold: {orig_params.get('rsi_buy')} → {adj_params.get('rsi_buy')}")
    
    if orig_params.get("rsi_sell") != adj_params.get("rsi_sell"):
        changes.append(f"RSI Sell threshold: {orig_params.get('rsi_sell')} → {adj_params.get('rsi_sell')}")
    
    # Check entry/exit condition changes
    orig_entry = original_spec.get("entry", "")
    adj_entry = adjusted_spec.get("entry", "")
    if orig_entry != adj_entry:
        changes.append("Entry conditions: Made less restrictive (changed strict comparisons to inclusive)")
    
    orig_exit = original_spec.get("exit", "")
    adj_exit = adjusted_spec.get("exit", "")
    if orig_exit != adj_exit:
        changes.append("Exit conditions: Made less restrictive (changed strict comparisons to inclusive)")
    
    if changes:
        return "The following parameters were adjusted to generate more trades:\n" + "\n".join(f"- {change}" for change in changes)
    return "No parameter changes detected."


def text_to_spec_llm(strategy_text: str, symbol: str, cash: float):
    """
    Use Gemini to convert plain-English strategy text into
    a JSON spec compatible with backtest_engine.build_strategy_from_spec.
    Uses caching to ensure the same strategy always produces the same spec.
    """
    if not GOOGLE_API_KEY:
        return default_spec(symbol, cash)

    # Create cache key from strategy text, symbol, and cash
    import hashlib
    cache_key = hashlib.md5(f"{strategy_text.strip()}_{symbol}_{cash}".encode()).hexdigest()
    
    # Initialize cache in session state if not exists
    if "strategy_spec_cache" not in st.session_state:
        st.session_state.strategy_spec_cache = {}
    
    # Check if we have a cached spec for this exact strategy
    if cache_key in st.session_state.strategy_spec_cache:
        # Return cached spec (ensures consistent results)
        cached_spec = st.session_state.strategy_spec_cache[cache_key].copy()
        cached_spec["stock_name"] = symbol  # Update symbol in case it changed
        cached_spec["cash"] = cash  # Update cash in case it changed
        return cached_spec

    # Generate new spec using LLM
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
            spec = default_spec(symbol, cash)
        else:
            spec.setdefault("stock_name", symbol)
            spec.setdefault("cash", cash)

            if "params" not in spec:
                spec["params"] = default_spec(symbol, cash)["params"]

            fallback = default_spec(symbol, cash)
            spec.setdefault("entry", fallback["entry"])
            spec.setdefault("exit", fallback["exit"])
        
        # Cache the spec for future use
        st.session_state.strategy_spec_cache[cache_key] = spec.copy()
        return spec
    except Exception:
        spec = default_spec(symbol, cash)
        # Cache even the fallback spec
        st.session_state.strategy_spec_cache[cache_key] = spec.copy()
        return spec


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


def _show_results(results: dict, trades: pd.DataFrame, equity: pd.DataFrame):
    """Helper to render backtest results in a clean format."""
    trades_count = int(results.get("trades", 0))
    win_rate = results.get("win_rate", 0.0)
    sharpe = results.get("sharpe", float("nan"))
    max_dd = results.get("max_drawdown_pct", 0.0)
    buy_hold = results.get("buy_hold_return_pct", 0.0)
    ret = results.get("return_pct", 0.0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Return %", f"{ret:.2f}")
    m1.metric("Trades", trades_count)
    
    if trades_count > 0 and win_rate == win_rate:
        m2.metric("Win Rate", f"{win_rate:.2f}%")
    else:
        m2.metric("Win Rate", "N/A")
    
    if trades_count > 0 and sharpe == sharpe:
        m2.metric("Sharpe", f"{sharpe:.2f}")
    else:
        m2.metric("Sharpe", "N/A")
    
    m3.metric("Max DD %", f"{max_dd:.2f}")
    m3.metric("Buy & Hold %", f"{buy_hold:.2f}")

    if isinstance(equity, pd.DataFrame) and not equity.empty:
        st.line_chart(equity["Equity"] if "Equity" in equity.columns else equity.iloc[:, 0])

    if trades_count == 0:
        st.info(
            "This strategy did not place any trades over the selected period. "
            "Try a different symbol, timeframe, or describe a less restrictive strategy."
        )
    elif isinstance(trades, pd.DataFrame) and not trades.empty:
        st.dataframe(trades)
        csv = trades.to_csv(index=False).encode("utf-8")
        st.download_button("Download Trades CSV", data=csv, file_name="trades.csv", mime="text/csv")


def render_strategy_explanation_with_gemini(
    symbol: str,
    spec_or_code: Union[dict, str],
    is_json_spec: bool,
    results: dict,
    period: str,
    interval: str,
    model_name: str = "gemini-flash-latest",
):
    """Calls Gemini to produce an educational explanation + implementation guide."""
    try:
        context = f"""
SYMBOL: {symbol}
TIMEFRAME REQUESTED: period={period}, interval={interval}
BACKTEST METRICS (for context, do not repeat numbers verbatim unless helpful):
{_safe_json(results)}
""".strip()

        if is_json_spec:
            strategy_block = "JSON STRATEGY SPEC:\n" + _safe_json(spec_or_code)
        else:
            strategy_block = "PYTHON STRATEGY CODE:\n```python\n" + str(spec_or_code) + "\n```"

        prompt = f"""
You are an expert trading mentor. Explain clearly and concisely.

IMPORTANT: The backtesting engine only supports SMA_fast, SMA_slow, RSI, and crossover().

If the user's original description mentioned concepts like "opening range breakout",
time-of-day, or R-multiples, treat this strategy as an APPROXIMATION of that idea using
SMA/RSI. Explicitly mention in your explanation that it is an approximation of the
user's described strategy using these limited building blocks.

{context}

{strategy_block}

Write an explanation to help a *new trader* understand and implement this strategy in the real world.

Structure the answer with these sections and keep each section concise and practical:

1) What this strategy tries to capture
2) Exact rules in plain English
3) How to implement it on a broker/charting platform
4) Risk management
5) When it tends to work / fail
6) Ideas to iterate

Tone: friendly, practical, and non-promissory. Do not guarantee profits.
"""

        model = genai.GenerativeModel(model_name)
        with st.spinner("Generating a plain-English explanation…"):
            resp = model.generate_content(prompt)

        st.markdown("### 🧠 How this strategy works (and how to use it)")
        st.markdown(resp.text if resp and resp.text else "_No explanation generated._")

    except Exception as e:
        st.warning(f"Couldn't generate explanation: {e}")


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

        chosen_strategy = st.session_state.saved_strategies[chosen_name]
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
                "Backtest Lookback",
                ["6mo", "1y", "2y", "5y"],
                index=1,  # default to 1y
                key="saved_period",
            )

        st.info(
            f"ℹ️ Using daily data over approximately the last {period} for this stock."
        )

        if st.button("Run Backtest (Saved Strategy)", type="primary"):
            with st.spinner("Running backtest..."):
                try:
                    # Use the saved JSON spec directly (no LLM conversion needed)
                    spec = base_json.copy()

                    # Ensure required fields exist with fallbacks
                    if "params" not in spec or not spec["params"]:
                        spec["params"] = {
                            "fast": 10,
                            "slow": 30,
                            "rsi": 14,
                            "rsi_buy": 30,
                            "rsi_sell": 70,
                        }

                    if "entry" not in spec or not spec.get("entry"):
                        spec["entry"] = "crossover(SMA_fast, SMA_slow) and RSI < rsi_buy"

                    if "exit" not in spec or not spec.get("exit"):
                        spec["exit"] = "crossover(SMA_slow, SMA_fast) or RSI > rsi_sell"

                    # Update symbol and name for display purposes
                    spec["stock_name"] = symbol
                    spec["name"] = spec.get("name", chosen_name)

                    # Fetch data to know number of points used
                    data = fetch_ohlcv(symbol, period=period, interval="1d")

                    # Basic safety check for newer stocks (very little history)
                    if len(data) < 60:
                        st.warning(
                            f"Only {len(data)} daily candles available for {symbol} over {period}. "
                            "Please select a shorter lookback or choose a more established stock."
                        )
                        st.stop()  # stop this run cleanly

                    data_points = len(data)
                    data_start_date = data.index[0] if len(data) > 0 else None
                    data_end_date = data.index[-1] if len(data) > 0 else None

                    # Run backtest once, as-is
                    bt_result, trades_df, equity_df = run_backtest(
                        symbol=symbol,
                        mode="spec",
                        strategy_spec_or_code=spec,
                        cash=initial_capital,
                        period=period,
                        interval="1d",
                    )

                    total_trades = bt_result.get("trades", 0)
                    win_rate = bt_result.get("win_rate", 0.0)
                    min_trades = get_min_trades_for_period(period)

                    # Warn if sample size is low, but do NOT change the strategy
                    if total_trades < min_trades:
                        st.warning(
                            f"Strategy only generated {total_trades} trades "
                            f"(minimum recommended: {min_trades} for {period}). "
                            "Treat these results as low-confidence. "
                            "Consider loosening your entry rules or using a different symbol/timeframe."
                        )

                    # ---------- Standard results (same as custom tab) ----------
                    st.markdown("## 📊 Backtest Results (Saved Strategy)")

                    # Show data points metric above the shared results block
                    st.metric("Data Points Used", data_points)

                    # This draws: Return %, Trades, Win Rate, Sharpe, Max DD, Buy & Hold,
                    # equity curve chart, full trades table + CSV download
                    _show_results(bt_result, trades_df, equity_df)

                    render_strategy_explanation_with_gemini(
                        symbol=symbol,
                        spec_or_code=spec,
                        is_json_spec=True,
                        results=bt_result,
                        period=period,
                        interval="1d",
                    )

                    # ---------- Trade highlights (top/bottom 3) ----------
                    detailed_trades = build_trade_summary(trades_df, initial_capital)
                    st.markdown("## 🧾 Trade Highlights")
                    render_trade_lists(detailed_trades)

                    # ---------- AI explanation ----------
                    st.markdown("## 🧠 Backtest Summary (AI)")
                    if not GOOGLE_API_KEY:
                        st.warning("Set GOOGLE_API_KEY in your .env file to enable AI summaries.")
                    else:
                        time_analysis = analyze_time_based_performance(
                            trades_df, equity_df, data_start_date, data_end_date
                        )

                        time_context = ""
                        if time_analysis:
                            time_context = f"""

TIME-BASED PERFORMANCE ANALYSIS:
{time_analysis}
"""

                        low_trades_note = ""
                        if total_trades < min_trades:
                            low_trades_note = (
                                f"Note: This strategy only produced {total_trades} trades, which is below "
                                f"the recommended minimum of {min_trades} for {period}. "
                                "Emphasize that the statistical confidence is low.\n"
                            )

                        summary_prompt = f"""
Summarize this trading backtest for a beginner trader.

{low_trades_note}
Explain:
- What these results roughly mean
- Whether the strategy appears strong or weak compared to buy & hold
- Main risks or weaknesses to be aware of
- One or two ideas to improve the strategy

Strategy name: {spec.get('name', 'Unnamed Strategy')}
Symbol: {symbol}
Data Range: {data_start_date.strftime('%Y-%m-%d') if data_start_date else 'Unknown'} to {data_end_date.strftime('%Y-%m-%d') if data_end_date else 'Unknown'}
Data Points: {data_points}
Total Trades: {total_trades}
Win Rate: {win_rate:.2f}%
Strategy Return [%]: {bt_result.get('return_pct', 0.0):.2f}
Buy & Hold Return [%]: {bt_result.get('buy_hold_return_pct', 0.0):.2f}
Max Drawdown [%]: {bt_result.get('max_drawdown_pct', 0.0):.2f}

{time_context}
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
    
    st.caption(
        "Describe your strategy in plain English. For example:\n"
        "*'I want a swing-trading strategy on AAPL that uses a 20-day and 50-day moving average and RSI. "
        "Go long when the 20-day SMA crosses above the 50-day SMA AND RSI(14) is below 60 (not overbought yet). "
        "Exit when RSI rises above 70 or when the 20-day SMA crosses back below the 50-day SMA. "
        "Use a stop-loss just below the recent swing low and risk about 1% of my account per trade.'*"
    )

    user_text = st.text_area(
        "Describe your strategy:",
        height=200,
        key="plain_desc_custom",
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
            "Backtest Lookback",
            ["6mo", "1y", "2y", "5y"],
            index=1,  # default to 1y
            key="custom_period",
        )

    custom_interval = "1d"
    st.info(
        f"ℹ️ Using daily data over approximately the last {custom_period} for this stock."
    )

    if st.button("Run Backtest (Plain English)", type="primary", key="btn_plain_custom"):
        if not user_text.strip():
            st.warning("Please describe your strategy first.")
        else:
            plain_prompt = f"""
You are a trading-strategy generator for a restricted backtesting engine.

The user will describe the kind of strategy they want in natural language.

Your job is to translate THAT SPECIFIC IDEA into a JSON spec in the format below.

You MUST NOT invent a completely different, generic strategy.

Return ONLY JSON (no markdown, no backticks, no commentary).

- Do NOT include any explanations, comments, parentheses, or text outside the JSON object.

JSON FORMAT EXAMPLE (adapt values to the user idea):

{{
  "name": "Short human-readable name that clearly reflects the user's idea",
  "params": {{"fast": 10, "slow": 30, "rsi": 14, "rsi_buy": 55, "rsi_sell": 70}},
  "entry": "crossover(SMA_fast, SMA_slow) and RSI > rsi_buy",
  "exit": "RSI > rsi_sell"
}}

HARD CONSTRAINTS:

- Keys: name, params, entry, exit.

- params: only numeric values.

- Indicators: SMA_fast, SMA_slow, RSI.

- Functions: crossover(SMA_fast, SMA_slow).

- Allowed variable: close.

- Allowed logic: comparisons vs numbers/params, and/or/not.

CONTEXT:

SYMBOL: {custom_symbol}

TIMEFRAME: period={custom_period}, interval={custom_interval}

USER STRATEGY DESCRIPTION:

{user_text}
"""

            try:
                with st.spinner("Turning your description into a backtestable strategy..."):
                    model = genai.GenerativeModel("gemini-flash-latest")
                    resp = model.generate_content(plain_prompt)
                    spec_text_raw = (resp.text or "").strip()

                st.subheader("Generated Strategy (Raw JSON from your description)")
                st.code(spec_text_raw, language="json")

                try:
                    custom_spec = extract_json_block(spec_text_raw)
                    if custom_spec is None:
                        raise ValueError("Could not extract JSON from LLM response")
                    
                    st.markdown("#### Parsed JSON")
                    st.code(json.dumps(custom_spec, indent=2), language="json")

                    # Simple structural validation
                    required_keys = {"name", "params", "entry", "exit"}
                    if not required_keys.issubset(custom_spec.keys()):
                        st.warning(
                            "The generated JSON is missing required fields. "
                            "Refine your description or try again."
                        )
                    else:
                        # Run backtest
                        data2 = fetch_ohlcv(custom_symbol, period=custom_period, interval=custom_interval)

                        # Basic safety check for newer stocks (very little history)
                        if len(data2) < 60:
                            st.warning(
                                f"Only {len(data2)} daily candles available for {custom_symbol} over {custom_period}. "
                                "Please select a shorter lookback or choose a more established stock."
                            )
                            st.stop()

                        bt_res2, trades2_df, eq2_df = run_backtest(
                            symbol=custom_symbol,
                            mode="spec",
                            strategy_spec_or_code=custom_spec,
                            period=custom_period,
                            interval=custom_interval,
                            cash=custom_capital,
                            commission=0.001,
                        )

                        st.success("✅ Backtest complete")
                        _show_results(bt_res2, trades2_df, eq2_df)
                        
                        # Show strategy explanation
                        render_strategy_explanation_with_gemini(
                            symbol=custom_symbol,
                            spec_or_code=custom_spec,
                            is_json_spec=True,
                            results=bt_res2,
                            period=custom_period,
                            interval=custom_interval,
                        )

                except json.JSONDecodeError as e:
                    st.warning("Gemini's output wasn't valid JSON. Please try rephrasing your strategy description.")
                    st.error(f"JSON Error: {e}")
                except Exception as e:
                    st.error(f"Backtest error: {e}")

            except Exception as e:
                st.error(f"LLM Error while parsing your description: {e}")

render_ai_drawer(
    context_hint=(
        "Running backtests on saved or custom strategies and interpreting "
        "metrics such as win rate, Sharpe ratio, and max drawdown."
    ),
    page_title="Backtest Strategy",
    key_prefix="backtest_ai",
)