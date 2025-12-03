import streamlit as st
import json
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from components.ai_drawer import render_ai_drawer

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Strategy Generator", layout="wide")

col_back, _ = st.columns([1, 11])
with col_back:
    if st.button("← Home", use_container_width=True):
        st.switch_page("app.py")

st.markdown("""
<h1 style='text-align:left; color:#27EEF5; margin-bottom:10px;'>
    🧠 Stock Strategy Generator
</h1>
<p style='text-align:left; color:#b0b0b0; margin-top:-10px; margin-bottom:20px;'>
    Let the LLM design a trading strategy for you, then backtest it.
</p>
""", unsafe_allow_html=True)


st.divider()
strategy_spec = None

st.caption("Fill in the details below to generate a clear trading strategy.")
col1, col2 = st.columns(2)
with col1:
    stock_name = st.text_input("Stock Name (ticker):", "AAPL", key="stock_name")
    total_capital = st.text_input("Total Capital ($):", "10000", key="total_capital")
    potential_profit = st.text_input("Potential Profit Target (%):", "10", key="potential_profit")
with col2:
    risk_level = st.text_input("Total Risk Level:", "Medium", key="risk_level")
    term = st.selectbox("Time Horizon:", ["Short Term", "Long Term"], key="term")
    experience = st.selectbox("Experience Level:", ["Beginner", "Intermediate", "Expert"], key="experience")

st.caption("Describe any additional strategy details below (optional):")
description = st.text_area("Additional Strategy Details:", height=100, key="plain_desc")

if 'saved_strategies' not in st.session_state or not isinstance(st.session_state.saved_strategies, dict):
    st.session_state.saved_strategies = {}

strategy_save_name = st.text_input("Strategy Name:", "My Strategy", key="save_name_bottom")

if st.button("Generate Strategy", key="btn_gen"):
    if not stock_name.strip() or not total_capital.strip() or not potential_profit.strip() or not risk_level.strip():
        st.warning("Please fill in all the required fields.")
    else:
        try:
            capital_float = float(total_capital) if total_capital.replace('.', '', 1).isdigit() else 10000.0
        except:
            capital_float = 10000.0
        
        if experience == "Beginner":
            exp_prompt = (
                "Write the strategy in very simple, step-by-step language. Avoid jargon. "
                "Explain indicators and risk management in plain English."
            )
        elif experience == "Intermediate":
            exp_prompt = (
                "Write the strategy with moderate detail, using some technical terms. "
                "Assume the user understands moving averages, RSI, and basic risk management."
            )
        else:
            exp_prompt = (
                "Write the strategy with advanced detail, using technical language and robust risk management. "
                "Assume the user is an experienced trader."
            )

        model = genai.GenerativeModel("gemini-flash-latest")

        with st.spinner("Generating strategy (plain English)..."):
            plain_prompt = f"""
You are a trading-strategy generator for a restricted backtesting engine.

{exp_prompt}

Do NOT return any code or JSON in this response. Just return the strategy as a readable text description, including:
- Overall idea
- Entry rules
- Exit rules
- Risk management guidelines
- Position sizing and any tips for the trader

User details:
- Stock Name: {stock_name}
- Total Capital: {total_capital}
- Potential Profit Target (%): {potential_profit}
- Total Risk Level: {risk_level}
- Term: {term}
- Experience Level: {experience}
- Additional Details: {description}
"""

            resp = model.generate_content(plain_prompt)
            strategy_text = (resp.text or "_No strategy generated._").strip()

        with st.spinner("Encoding strategy into JSON spec for backtesting..."):
            json_prompt = f"""
You are a trading strategy encoder for a Python backtesting engine.

You are given the user's trading preferences and a natural-language trading strategy.

User details:
- Stock Name: {stock_name}
- Total Capital: {total_capital}
- Potential Profit Target (%): {potential_profit}
- Total Risk Level: {risk_level}
- Term: {term}
- Experience Level: {experience}
- Additional Details: {description}

Natural language strategy:
\"\"\"{strategy_text}\"\"\"

You MUST output ONLY a JSON object (no markdown, no backticks, no commentary) following exactly this schema:

{{
  "name": "short name for this strategy",
  "stock_name": "ticker symbol like 'AAPL'",
  "total_capital": {capital_float},
  "params": {{
    "fast": 10,
    "slow": 30,
    "rsi": 14,
    "rsi_buy": 55,
    "rsi_sell": 70
  }},
  "entry": "crossover(SMA_fast, SMA_slow) and RSI > rsi_buy",
  "exit": "RSI > rsi_sell or crossover(SMA_slow, SMA_fast)"
}}

Rules:
- stock_name must match the user's stock_name "{stock_name}".
- fast must be an integer < slow (e.g., 10 and 30).
- rsi must be an integer (e.g., 14).
- rsi_buy and rsi_sell must be integers between 0 and 100.
- entry and exit are Python expressions used in a backtesting engine.
- Use ONLY these names in the expressions: crossover, SMA_fast, SMA_slow, RSI, rsi_buy, rsi_sell, and, or, >, <, >=, <=.
- A safe default is:
  - entry: "crossover(SMA_fast, SMA_slow) and RSI > rsi_buy"
  - exit: "RSI > rsi_sell or crossover(SMA_slow, SMA_fast)"

Output VALID JSON only.
"""

            try:
                json_resp = model.generate_content(json_prompt)
                raw_json = json_resp.text.strip()
                
                if "```json" in raw_json:
                    raw_json = raw_json.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_json:
                    raw_json = raw_json.split("```")[1].split("```")[0].strip()
                
                strategy_json = json.loads(raw_json)
            except Exception:
                strategy_json = {
                    "name": f"LLM Strategy for {stock_name}",
                    "stock_name": stock_name,
                    "total_capital": capital_float,
                    "params": {
                        "fast": 10,
                        "slow": 30,
                        "rsi": 14,
                        "rsi_buy": 55,
                        "rsi_sell": 70,
                    },
                    "entry": "crossover(SMA_fast, SMA_slow) and RSI > rsi_buy",
                    "exit": "RSI > rsi_sell or crossover(SMA_slow, SMA_fast)",
                }

        st.session_state.generated_strategy = {
            "text": strategy_text,
            "json": strategy_json,
        }
        st.session_state.strategy_type = "llm_spec"
        
        st.success("✅ Strategy generated successfully!")
        st.info("💡 Go to Backtest page to view detailed results and performance metrics.")

        st.markdown("### 📋 Strategy Description")
        st.markdown(
            """
            <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.12); 
                       padding: 20px; border-radius: 12px; margin-top: 10px;'>
            """,
            unsafe_allow_html=True
        )
        st.write(strategy_text)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.session_state.saved_strategies[strategy_save_name] = st.session_state.generated_strategy

if st.button("Save Strategy with Different Name", key="btn_save_strategy_bottom"):
    if 'generated_strategy' in st.session_state and st.session_state.generated_strategy:
        st.session_state.saved_strategies[strategy_save_name] = st.session_state.generated_strategy
        st.success(f"Strategy saved as '{strategy_save_name}'!")
    else:
        st.warning("Please generate a strategy before saving.")

if st.session_state.saved_strategies:
    st.divider()
    st.markdown("### 📁 Saved Strategies")
    st.caption("Click on a strategy to view its description, or go to Backtest page to see full results.")
    for name, strat in st.session_state.saved_strategies.items():
        with st.expander(name):
            st.write("**Strategy Description:**")
            st.write(strat.get("text", ""))
            if st.button(f"📈 Backtest '{name}'", key=f"backtest_{name}"):
                st.session_state.generated_strategy = strat
                st.session_state.strategy_type = "llm_spec"
                st.switch_page("pages/Backtest_Strategy.py")


render_ai_drawer(
    context_hint=(
        "Generating and backtesting trading strategies using SMA, RSI, "
        "win rate, drawdown, and other performance metrics."
    ),
    page_title="Trading Strategy",
    key_prefix="strategy_ai",
)