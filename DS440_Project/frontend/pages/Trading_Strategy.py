# ensure project root is importable
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # -> .../DS440_Project
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Union

# -------- interval options based on selected period --------
def get_interval_options(period: str):
    """
    Return valid Yahoo/yfinance interval options for a given period.

    Yahoo limits intraday history roughly as:
      - 1m: ~7d
      - 2m/5m/15m/30m: ~60d
      - 60m/1h: ~2y
      - 1d+: many years
    """
    if period in ["1d", "5d"]:
        # very short history: allow full intraday range
        return ["1m", "2m", "5m", "15m", "30m", "1h", "1d"]
    elif period in ["1mo"]:
        # one month: 1m is too long; 2m/5m/15m/30m/1h/1d are ok-ish
        return ["2m", "5m", "15m", "30m", "1h", "1d"]
    elif period in ["3mo", "6mo"]:
        # 3–6 months: stick to 30m/1h/1d
        return ["30m", "1h", "1d"]
    else:  # "1y", "2y", or anything longer
        return ["1h", "1d"]
# -----------------------------------------------------------

# set page config FIRST
st.set_page_config(page_title="Trading Strategy", layout="wide")

from backend.backtest_engine import run_backtest, STRATEGY_TEMPLATE

col_back, col_title = st.columns([1, 11])
with col_back:
    if st.button("← Home"):
        st.switch_page("app.py")
with col_title:
    st.title("📊 Stock Data Viewer")

# ------- Gemini explainer helper -------
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

    def _safe_json(obj):
        try:
            # default=str handles DataFrames / arrays by stringifying them
            return json.dumps(obj, indent=2, default=str)
        except Exception:
            if isinstance(obj, dict):
                return json.dumps({k: str(v) for k, v in obj.items()}, indent=2)
            return str(obj)

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

1) **What this strategy tries to capture** — the intuition in 2–4 sentences.
2) **Exact rules in plain English** — entries/exits, stops/takes, time-in-market, and parameter meaning.
3) **How to implement it on a broker/charting platform** — a short checklist (e.g., how to add SMA/RSI, alerts, and order types).
4) **Risk management** — position sizing, stop-loss placement, max risk per trade, and when to stand down.
5) **When it tends to work / fail** — market regimes, pitfalls, and false signals to watch for.
6) **Ideas to iterate** — 2–4 quick variations (e.g., tweak periods, add filters, change exits).

Tone: friendly, practical, and non-promissory. Avoid hype. Do not guarantee profits.
"""

        model = genai.GenerativeModel(model_name)
        with st.spinner("Generating a plain-English explanation…"):
            resp = model.generate_content(prompt)

        st.markdown("### 🧠 How this strategy works (and how to use it)")
        st.markdown(resp.text if resp and resp.text else "_No explanation generated._")
    except Exception as e:
        st.warning(f"Couldn’t generate explanation: {e}")
# ------- end helper -------

# app init
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.title("🤖 AI-Powered Trading Strategy")

# -------------------- Shared controls --------------------
symbol = st.text_input("Enter a Stock Symbol:", "AAPL", key="sym_top")

c1, c2, c3, c4 = st.columns(4)
with c1:
    # Expanded period options
    period = st.selectbox(
        "History",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
        index=2,  # default to "1mo" (or whatever you prefer)
        key="period_top",
    )
with c2:
    # Interval options depend on selected period
    interval_options = get_interval_options(period)
    interval = st.selectbox(
        "Interval",
        interval_options,
        index=0,  # first valid option
        key="int_top",
    )
with c3:
    cash = st.number_input("Starting Cash", value=10000, step=1000, key="cash_top")
with c4:
    commission = st.number_input("Commission (fraction)", value=0.001, step=0.001, format="%.3f", key="comm_top")

# -------------------- Helper to render results --------------------
def _show_results(results: dict, trades: pd.DataFrame, equity: pd.DataFrame):
    trades_count = int(results.get("trades", 0))
    win_rate = results.get("win_rate", 0.0)
    sharpe = results.get("sharpe", float("nan"))
    max_dd = results.get("max_drawdown_pct", 0.0)
    buy_hold = results.get("buy_hold_return_pct", 0.0)
    ret = results.get("return_pct", 0.0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Return %", f"{ret:.2f}")
    m1.metric("Trades", trades_count)

    # Win rate & Sharpe: show N/A if no trades
    if trades_count > 0 and win_rate == win_rate:  # win_rate == win_rate filters out NaN
        m2.metric("Win Rate", f"{win_rate:.2f}%")
    else:
        m2.metric("Win Rate", "N/A")

    if trades_count > 0 and sharpe == sharpe:
        m2.metric("Sharpe", f"{sharpe:.2f}")
    else:
        m2.metric("Sharpe", "N/A")

    m3.metric("Max DD %", f"{max_dd:.2f}")
    m3.metric("Buy & Hold %", f"{buy_hold:.2f}")

    # Equity curve
    if isinstance(equity, pd.DataFrame) and not equity.empty:
        st.line_chart(equity["Equity"] if "Equity" in equity.columns else equity.iloc[:, 0])

    # Extra message if no trades
    if trades_count == 0:
        st.info(
            "This strategy did not place any trades over the selected period. "
            "Try a different symbol, timeframe, or describe a less restrictive strategy."
        )
    elif isinstance(trades, pd.DataFrame) and not trades.empty:
        st.dataframe(trades)
        csv = trades.to_csv(index=False).encode("utf-8")
        st.download_button("Download Trades CSV", data=csv, file_name="trades.csv", mime="text/csv")

# -------------------- PART 4: Gemini → JSON → Backtest --------------------
if st.button("Generate Strategy", key="btn_gen"):
    json_prompt = f"""
You are a trading-strategy generator for backtesting.py.
Return ONLY JSON (no markdown, no backticks, no commentary).
The JSON must have these keys:
- name (string)
- params (object) may include: fast, slow, rsi, rsi_buy, rsi_sell
- entry (string) using allowed tokens: crossover(SMA_fast, SMA_slow), RSI, numeric thresholds, boolean operators
- exit (string) using allowed tokens: RSI and/or crossover(...)

Use this context:
SYMBOL: {symbol}

Example JSON to imitate exactly (adapt values to the summary):
{{
  "name": "SMA Cross with RSI Filter",
  "params": {{"fast": 10, "slow": 30, "rsi": 14, "rsi_buy": 55, "rsi_sell": 70}},
  "entry": "crossover(SMA_fast, SMA_slow) and RSI > rsi_buy",
  "exit": "RSI > rsi_sell"
}}
"""
    try:
        with st.spinner("Generating JSON strategy with Gemini..."):
            model = genai.GenerativeModel("gemini-flash-latest")  # or "gemini-1.5-flash"
            resp = model.generate_content(json_prompt)
            spec_text = (resp.text or "").strip()

        st.subheader("Generated Strategy (Raw JSON)")
        st.code(spec_text, language="json")

        # Some models wrap in ```json ... ```
        if spec_text.startswith("```"):
            spec_text = spec_text.strip("` \n")
            if spec_text.lower().startswith("json"):
                spec_text = spec_text[4:].strip()

        try:
            spec = json.loads(spec_text)
            results, trades, equity = run_backtest(
                symbol=symbol,
                mode="spec",
                strategy_spec_or_code=spec,
                period=period,
                interval=interval,
                cash=cash,
                commission=commission,
            )
            st.success("✅ Backtest complete")
            _show_results(results, trades, equity)
            # NEW: explanation under the chart & metrics
            render_strategy_explanation_with_gemini(
                symbol=symbol,
                spec_or_code=spec,  # this is your parsed JSON spec
                is_json_spec=True,
                results=results,
                period=period,
                interval=interval,
            )
        except json.JSONDecodeError:
            st.warning("Gemini output wasn't valid JSON. Fix it and paste below in the manual JSON box.")
        except Exception as e:
            st.error(f"Backtest error: {e}")

    except Exception as e:
        st.error(f"LLM Error: {e}")

# -------------------- Manual Backtest (Step 3 UI) --------------------
st.markdown("---")
st.subheader("Backtest a Strategy (Manual)")

# Plain English mode as default
mode = st.radio(
    "Strategy Input Mode",
    ["Plain English (AI-assisted)", "JSON Spec (safe)", "Python Code (advanced)"],
    key="mode_radio",
    index=0,
)

# ---------- 1) Plain English (AI-assisted via Gemini) ----------
if mode == "Plain English (AI-assisted)":
    st.caption(
        "Describe your strategy in plain English. For example:\n"
        "*'I want a swing-trading strategy on AAPL that uses a 20-day and 50-day moving average and RSI. "
        "Go long when the 20-day SMA crosses above the 50-day SMA AND RSI(14) is below 60 (not overbought yet). "
        "Exit when RSI rises above 70 or when the 20-day SMA crosses back below the 50-day SMA. "
        "Use a stop-loss just below the recent swing low and risk about 1% of my account per trade.'*"
    )
    description = st.text_area("Describe your strategy:", height=200, key="plain_desc")

    if st.button("Run Backtest (Plain English)", key="btn_plain"):
        if not description.strip():
            st.warning("Please describe your strategy first.")
        else:
            plain_prompt = f"""
            You are a trading-strategy generator for a restricted backtesting engine.
            The user will describe the kind of strategy they want in natural language.
            Your job is to translate THAT SPECIFIC IDEA into a JSON spec in the format below.
            You MUST NOT invent a completely different, generic strategy.

            Return ONLY JSON (no markdown, no backticks, no commentary).

            JSON FORMAT EXAMPLE (structure only, adapt values to the user idea):

            {{
              "name": "Short human-readable name that clearly reflects the user's idea",
              "params": {{
                "fast": 10,
                "slow": 30,
                "rsi": 14,
                "rsi_buy": 55,
                "rsi_sell": 70
              }},
              "entry": "crossover(SMA_fast, SMA_slow) and RSI > rsi_buy",
              "exit": "RSI > rsi_sell"
            }}

            HARD CONSTRAINTS (VERY IMPORTANT):
            - Use keys: "name", "params", "entry", "exit".
            - "name" MUST clearly reflect the user's idea (e.g. "Opening Range Breakout Approximation" if they said ORB).
            - "params" must contain only numeric values (ints or floats).
            - Allowed indicator names in expressions:
              - SMA_fast, SMA_slow   (simple moving averages defined by params.fast and params.slow)
              - RSI                   (RSI defined by params.rsi)
            - Allowed function calls in entry/exit:
              - crossover(SMA_fast, SMA_slow)
            - Allowed additional variable:
              - close (the current close price)
            - Allowed operations in entry/exit:
              - comparisons on RSI or close vs numeric thresholds or params (>, <, >=, <=).
              - boolean operators: and, or, not.
            - Do NOT reference any other indicators, functions, or variables.

            INTERPRETING THE USER'S IDEA:
            - You MUST implement the same *core* idea the user described, as faithfully as possible,
              using ONLY the allowed building blocks.
            - If the user mentions something like "opening range breakout" or "first 30 minutes",
              approximate that idea using SMA/RSI logic (for example, using moving-average
              crossovers and RSI thresholds to represent breakouts and momentum),
              but KEEP the name and explanation aligned with the ORB concept.

            CONTEXT:
            SYMBOL: {symbol}
            TIMEFRAME: period={period}, interval={interval}

            USER STRATEGY DESCRIPTION:
            {description}
            """

            try:
                with st.spinner("Turning your description into a backtestable strategy..."):
                    model = genai.GenerativeModel("gemini-flash-latest")
                    resp = model.generate_content(plain_prompt)
                    spec_text = (resp.text or "").strip()

                st.subheader("Generated Strategy (Raw JSON from your description)")
                st.code(spec_text, language="json")

                # Strip ```json fences if present
                if spec_text.startswith("```"):
                    spec_text = spec_text.strip("` \n")
                    if spec_text.lower().startswith("json"):
                        spec_text = spec_text[4:].strip()

                try:
                    spec = json.loads(spec_text)

                    # Simple structural validation
                    required_keys = {"name", "params", "entry", "exit"}
                    if not required_keys.issubset(spec.keys()):
                        st.warning(
                            "The generated JSON is missing some required fields. "
                            "Please refine your description or fix the JSON in JSON Spec mode."
                        )
                    else:
                        results, trades, equity = run_backtest(
                            symbol=symbol,
                            mode="spec",
                            strategy_spec_or_code=spec,
                            period=period,
                            interval=interval,
                            cash=cash,
                            commission=commission,
                        )
                        st.success("✅ Backtest complete (from your plain-English description)")
                        _show_results(results, trades, equity)

                        render_strategy_explanation_with_gemini(
                            symbol=symbol,
                            spec_or_code=spec,
                            is_json_spec=True,
                            results=results,
                            period=period,
                            interval=interval,
                        )

                except json.JSONDecodeError:
                    st.warning("Gemini's output wasn't valid JSON. You can fix it and paste into JSON Spec mode.")
                except Exception as e:
                    st.error(f"Backtest error: {e}")

            except Exception as e:
                st.error(f"LLM Error while parsing your description: {e}")

# ---------- 2) JSON Spec (safe) ----------
elif mode == "JSON Spec (safe)":
    st.caption("Provide a JSON spec. Example:")
    st.code("""{
  "name": "SMA Cross with RSI Filter",
  "params": {"fast":10,"slow":30,"rsi":14,"rsi_buy":55,"rsi_sell":70},
  "entry": "crossover(SMA_fast, SMA_slow) and RSI > rsi_buy",
  "exit": "RSI > rsi_sell"
}""", language="json")

    spec_text_manual = st.text_area("Strategy Spec (JSON)", height=220, key="json_box")

    if st.button("Run Backtest (Spec)", key="btn_spec_manual"):
        try:
            spec = json.loads(spec_text_manual)

            results, trades, equity = run_backtest(
                symbol=symbol,
                mode="spec",
                strategy_spec_or_code=spec,
                period=period,
                interval=interval,
                cash=cash,
                commission=commission,
            )
            st.success("Backtest complete")
            _show_results(results, trades, equity)

            render_strategy_explanation_with_gemini(
                symbol=symbol,
                spec_or_code=spec,
                is_json_spec=True,
                results=results,
                period=period,
                interval=interval,
            )

        except Exception as e:
            st.error(f"Backtest error: {e}")

# ---------- 3) Python Code (advanced) ----------
else:
    st.caption("Paste a Strategy subclass named `UserStrategy`. A starter template is below.")
    with st.expander("Template"):
        st.code(STRATEGY_TEMPLATE, language="python")

    code_text = st.text_area("Strategy Code (Python)", height=260, key="code_box")

    if st.button("Run Backtest (Code)", key="btn_code_manual"):
        try:
            results, trades, equity = run_backtest(
                symbol=symbol,
                mode="code",
                strategy_spec_or_code=code_text,
                period=period,
                interval=interval,
                cash=cash,
                commission=commission,
            )
            st.success("Backtest complete")
            _show_results(results, trades, equity)

            render_strategy_explanation_with_gemini(
                symbol=symbol,
                spec_or_code=code_text,
                is_json_spec=False,
                results=results,
                period=period,
                interval=interval,
            )

        except Exception as e:
            st.error(f"Backtest error: {e}")

st.markdown(
    "> ⚠️ **Note:** Backtests are simulations based on historical data and assumptions. "
    "They do not guarantee future performance and are for educational use only."
)