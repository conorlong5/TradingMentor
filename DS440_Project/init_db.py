import sqlite3

conn = sqlite3.connect("trading_mentor.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_symbol TEXT,
    date_requested TEXT,
    user_capital REAL,
    risk_level TEXT
)
""")

print("✅ Database initialized")
conn.commit()
conn.close()
