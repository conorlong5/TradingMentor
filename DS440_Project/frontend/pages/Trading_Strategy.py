# --- ensure project root is importable ---
import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]  # -> .../DS440_Project
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ----------------------------------------

import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Union

# set page config FIRST
st.set_page_config(page_title="Trading Strategy", layout="wide")

from backend.backtest_engine import run_backtest, STRATEGY_TEMPLATE

# ------- Gemini explainer helper -------
def render_strategy_explanation_with_gemini(
    *, model_name: str = "gemini-flash-latest",
    symbol: str,
    spec_or_code: Union[dict, str],
    is_json_spec: bool,
    results: dict,
    period: str,
    interval: str
):
    """Calls Gemini to produce an educational explanation + implementation guide."""
    try:
        # Build a tight, deterministic prompt
        header = "You are an expert trading mentor. Explain clearly and concisely."

        # Convert results into JSON-safe format
        def _safe_json(obj):
            try:
                return json.dumps(obj, indent=2, default=str)
            except Exception:
                # Fall back: make any complex object a string
                clean = {k: str(v) for k, v in (obj.items() if isinstance(obj, dict) else {})}
                return json.dumps(clean, indent=2)

        context = f"""
        SYMBOL: {symbol}
        TIMEFRAME REQUESTED: period={period}, interval={interval}

        BACKTEST METRICS (for context, do not repeat numbers verbatim unless helpful):
        {_safe_json(results)}
        """.strip()

        if is_json_spec:
            strategy_block = "JSON STRATEGY SPEC:\n" + json.dumps(spec_or_code, indent=2)
        else:
            strategy_block = "PYTHON STRATEGY CODE:\n```python\n" + spec_or_code + "\n```"

        ask = f"""
{header}

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
            resp = model.generate_content(ask)

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
summary = st.text_area("Paste your indicator summary here:")

c1, c2, c3, c4 = st.columns(4)
with c1:
    period = st.selectbox("History", ["3mo", "6mo", "1y", "2y"], index=1, key="period_top")
with c2:
    interval = st.selectbox("Interval", ["1d","1h","30m","15m"], index=0, key="int_top")
with c3:
    cash = st.number_input("Starting Cash", value=10000, step=1000, key="cash_top")
with c4:
    commission = st.number_input("Commission (fraction)", value=0.001, step=0.001, format="%.3f", key="comm_top")

# -------------------- Helper to render results --------------------
def _show_results(results: dict, trades: pd.DataFrame, equity: pd.DataFrame):
    m1, m2, m3 = st.columns(3)
    m1.metric("Return %", f"{results.get('return_pct', 0):.2f}")
    m2.metric("Win Rate", f"{results.get('win_rate', 0):.2f}")
    m3.metric("Max DD %", f"{results.get('max_drawdown_pct', 0):.2f}")
    m1.metric("# Trades", results.get("trades", 0))
    m2.metric("Sharpe", f"{results.get('sharpe', 0):.2f}")
    m3.metric("Buy & Hold %", f"{results.get('buy_hold_return_pct', 0):.2f}")

    if isinstance(equity, pd.DataFrame) and not equity.empty:
        st.line_chart(equity["Equity"] if "Equity" in equity.columns else equity.iloc[:, 0])

    if isinstance(trades, pd.DataFrame) and not trades.empty:
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
SUMMARY:
{summary}

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
                interval=interval
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

mode = st.radio("Strategy Input Mode", ["JSON Spec (safe)", "Python Code (advanced)"], key="mode_radio")

if mode == "JSON Spec (safe)":
    st.caption("Provide a JSON spec. Example:")
    st.code("""{
  "name": "SMA Cross with RSI Filter",
  "params": {"fast":10,"slow":30,"rsi":14,"rsi_buy":55,"rsi_sell":70},
  "entry": "crossover(SMA_fast, SMA_slow) and RSI > rsi_buy",
  "exit": "RSI > rsi_sell"
}""", language="json")

    spec_text_manual = st.text_area("Strategy Spec (JSON)", height=220, key="json_box")

    if st.button("Run Backtest (Spec)", key="btn_spec"):
        try:
            spec = json.loads(spec_text_manual)
            results, trades, equity = run_backtest(
                symbol=symbol, mode="spec", strategy_spec_or_code=spec,
                period=period, interval=interval, cash=cash, commission=commission
            )
            st.success("Backtest complete")
            _show_results(results, trades, equity)
            render_strategy_explanation_with_gemini(
                symbol=symbol,
                spec_or_code=spec,
                is_json_spec=True,
                results=results,
                period=period,
                interval=interval
            )
        except Exception as e:
            st.error(f"Backtest error: {e}")

else:
    st.caption("Paste a Strategy subclass named `UserStrategy`. A starter template is below.")
    with st.expander("Template"):
        st.code(STRATEGY_TEMPLATE, language="python")

    code_text = st.text_area("Strategy Code (Python)", height=260, key="code_box")

    if st.button("Run Backtest (Code)", key="btn_code"):
        try:
            results, trades, equity = run_backtest(
                symbol=symbol, mode="code", strategy_spec_or_code=code_text,
                period=period, interval=interval, cash=cash, commission=commission
            )
            st.success("Backtest complete")
            _show_results(results, trades, equity)
            render_strategy_explanation_with_gemini(
                symbol=symbol,
                spec_or_code=code_text,  # the code the user pasted
                is_json_spec=False,
                results=results,
                period=period,
                interval=interval
            )
        except Exception as e:
            st.error(f"Backtest error: {e}")