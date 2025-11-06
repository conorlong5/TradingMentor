import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

if st.session_state.get("page_loaded_once", False) is False:
    st.session_state.page_loaded_once = True
    st.switch_page("app")

st.set_page_config(page_title="Trading Strategy", layout="wide")
st.title("🤖 AI-Powered Trading Strategy")

symbol = st.text_input("Enter a Stock Symbol:", "AAPL")
summary = st.text_area("Paste your indicator summary here:")

if st.button("Generate Strategy"):
    prompt = f"""
    You are a professional trading strategist.
    Use the following stock data summary to suggest one strategy.

    Stock: {symbol}
    Summary:
    {summary}

    Include: Strategy name, Description, Entry/Exit rules, Stop Loss, Take Profit, Risk, and Holding period.
    """
    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(prompt)
        st.markdown(response.text)
    except Exception as e:
        st.error(f"LLM Error: {e}")
