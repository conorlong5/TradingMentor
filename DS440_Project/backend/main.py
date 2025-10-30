from fastapi import FastAPI
import yfinance as yf

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Trading Mentor Backend Running"}

@app.get("/stock/{symbol}")
def get_stock(symbol: str):
    # Fetch stock data
    data = yf.download(symbol, period="1mo", interval="1d")

    # Handle case: no data returned
    if data.empty:
        return {"error": f"No data found for symbol {symbol}"}

    # Extract the 'Close' prices
    close_prices = data["Close"].to_list()

    return {
        "symbol": symbol,
        "dates": data.index.strftime("%Y-%m-%d").to_list(),
        "prices": close_prices
    }
