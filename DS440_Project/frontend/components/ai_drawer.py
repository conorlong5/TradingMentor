import os
from typing import Optional
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

if "AI_DRAWER_LOADED" not in st.session_state:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    st.session_state["AI_DRAWER_LOADED"] = True


def render_ai_drawer(
    *,
    context_hint: str,
    page_title: str,
    key_prefix: str = "ai_drawer",
    expanded: bool = False,
) -> None:
    """
    Right-side 'drawer' helper that answers conceptual questions.

    Call this ONCE near the bottom of each page.
    """
    st.markdown(
        """
        <style>
        /* Make the helper column a bit visually separated */
        .ai-helper-box {
            border-radius: 12px;
            padding: 12px 14px;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    main_col, helper_col, sec_col = st.columns([0.000000000000000000000000000000000000000000000000000000000000000000000000000001, 1000000000000, 0.000000000000000000000000000000000000000000000000000000000000000000000000000001])

    with helper_col:
        with st.expander("💬 Ask the Trading Mentor", expanded=expanded):
            st.markdown(
                f"""
                <div class="ai-helper-box">
                    <p style="font-size: 13px; opacity: 0.8; margin-bottom: 6px;">
                        You’re on the <b>{page_title}</b> page.
                        Ask about any term you see (e.g. RSI, SMA, win rate, drawdown),
                        or general questions about trading and indicators.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            question = st.text_area(
                "Your question",
                placeholder="Example: What is RSI? How is win rate calculated in backtesting?",
                key=f"{key_prefix}_question",
                height=100,
            )

            if st.button("Ask AI", key=f"{key_prefix}_ask"):
                if not question.strip():
                    st.warning("Type a question first.")
                    return

                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    st.error("GEMINI_API_KEY / GOOGLE_API_KEY is not set in your .env file.")
                    return

                try:
                    model = genai.GenerativeModel("gemini-flash-latest")
                    prompt = f"""
You are a patient trading mentor.

Context:
- The user is currently on the "{page_title}" page of a trading app.
- The page is about: {context_hint}

User question:
\"\"\"{question.strip()}\"\"\"

Answer in a clear, concise way,
using beginner-friendly language.
Avoid hype or promises of profit.
If you use jargon (like RSI, SMA, drawdown),
explain it in plain English.
"""
                    with st.spinner("Thinking..."):
                        resp = model.generate_content(prompt)

                    answer = (resp.text or "").strip() if resp else ""
                    if not answer:
                        st.warning("The AI didn't return any text. Try rephrasing your question.")
                    else:
                        st.markdown("#### 🧠 Answer")
                        st.write(answer)
                except Exception as e:
                    st.error(f"AI helper error: {e}")
