import yfinance as yf
import ta
import pandas as pd

def get_stock_data(symbol, period="3mo"):
    data = yf.download(symbol, period=period, interval="1d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data["RSI"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
    data["MACD"] = ta.trend.MACD(data["Close"]).macd()
    data["Signal"] = ta.trend.MACD(data["Close"]).macd_signal()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    return data
