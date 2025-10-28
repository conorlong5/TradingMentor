import os
from supabase import create_client
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def login_or_signup():
    st.sidebar.header("🔑 Account Access")

    choice = st.sidebar.radio("Choose an option:", ["Login", "Sign Up"])

    email = st.sidebar.text_input("Email:")
    password = st.sidebar.text_input("Password:", type="password")

    if choice == "Sign Up":
        if st.sidebar.button("Create Account"):
            try:
                user = supabase.auth.sign_up({"email": email, "password": password})
                st.sidebar.success("✅ Account created! Please log in.")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    elif choice == "Login":
        if st.sidebar.button("Login"):
            try:
                user = supabase.auth.sign_in_with_password({"email": email, "password": password})
                if user:
                    st.session_state["user_email"] = email
                    st.sidebar.success("✅ Logged in!")
            except Exception as e:
                st.sidebar.error("Login failed.")
