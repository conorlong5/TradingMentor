import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_strategy(user_email, symbol, strategy_text, sentiment_score):
    """Save a user's trading strategy in Supabase"""
    supabase.table("strategies").insert({
        "user_email": user_email,
        "symbol": symbol,
        "strategy": strategy_text,
        "sentiment": sentiment_score
    }).execute()

def get_user_strategies(user_email):
    """Retrieve saved strategies for a given user"""
    response = supabase.table("strategies").select("*").eq("user_email", user_email).order("id", desc=True).execute()
    return response.data
