from __future__ import annotations
import ast
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import yfinance as yf

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import ta


# =========================================================
# DATA FETCHER
# =========================================================
def fetch_ohlcv(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV with safe Yahoo limits."""
    data = yf.download(
        symbol,
        start=start,
        end=end,
        period=None if (start or end) else period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        raise ValueError(f"No data for {symbol} with interval={interval} and period={period}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    data.dropna(inplace=True)
    return data


# =========================================================
# BUILD INDICATORS
# =========================================================
def _compute_indicators(df: pd.DataFrame, params: Dict[str, Any]):
    fast = int(params.get("fast", 10))
    slow = int(params.get("slow", 30))
    rsi_len = int(params.get("rsi", 14))

    df["SMA_fast"] = df["Close"].rolling(fast).mean()
    df["SMA_slow"] = df["Close"].rolling(slow).mean()
    df["RSI"] = ta.momentum.rsi(df["Close"], window=rsi_len)
    return df


# =========================================================
# SAFE EVAL FOR ENTRY/EXIT RULES
# =========================================================
def _safe_eval(expr: str, ctx: Dict[str, Any]) -> bool:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import not allowed")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id == "crossover"):
                raise ValueError("Only crossover() allowed as callable")
    return bool(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, ctx))


# =========================================================
# SPEC STRATEGY BUILDER
# =========================================================
def build_strategy_from_spec(spec: Dict[str, Any]) -> type[Strategy]:
    params = spec.get("params", {})
    entry_expr = spec.get("entry", "")
    exit_expr = spec.get("exit", "")

    class SpecStrategy(Strategy):
        def init(self):
            self.sma_fast = self.I(SMA, self.data.Close, int(params.get("fast", 10)))
            self.sma_slow = self.I(SMA, self.data.Close, int(params.get("slow", 30)))

            close = pd.Series(self.data.Close)
            rsi_series = ta.momentum.rsi(close, window=int(params.get("rsi", 14))).fillna(50)
            self.rsi = self.I(lambda: rsi_series.values)

        def next(self):
            ctx = {
                "crossover": crossover,
                "SMA_fast": self.sma_fast,
                "SMA_slow": self.sma_slow,
                "RSI": self.rsi[-1],
                "close": self.data.Close[-1],
                **{k: v for k, v in params.items() if isinstance(v, (int, float))},
            }

            if self.position.size == 0:
                if entry_expr and _safe_eval(entry_expr, ctx):
                    self.buy()
            else:
                if exit_expr and _safe_eval(exit_expr, ctx):
                    self.position.close()

    return SpecStrategy


# =========================================================
# CODE EXEC STRATEGY BUILDER (OPTIONAL)
# =========================================================
def build_strategy_from_code(py_code: str) -> type[Strategy]:
    """Not used by your UI but left here safely."""
    local_env = {}
    exec(py_code, {"__builtins__": {}}, local_env)
    Strat = local_env.get("UserStrategy")
    if not Strat:
        raise ValueError("UserStrategy not found")
    return Strat


# =========================================================
# RUN BACKTEST
# =========================================================
def run_backtest(
    symbol: str,
    mode: str,
    strategy_spec_or_code: Dict[str, Any] | str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "6mo",
    interval: str = "1d",
    cash: float = 10000,
    commission: float = 0.001,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:

    data = fetch_ohlcv(symbol, start, end, period, interval)

    if mode == "spec":
        Strat = build_strategy_from_spec(strategy_spec_or_code)
    else:
        Strat = build_strategy_from_code(strategy_spec_or_code)

    bt = Backtest(data, Strat, cash=cash, commission=commission)

    stats = bt.run()
    trades = bt._results._trades if hasattr(bt._results, "_trades") else pd.DataFrame()
    equity = bt._results._equity_curve if hasattr(bt._results, "_equity_curve") else pd.DataFrame()

    result = {
        "trades": int(stats.get("# Trades", 0)),
        "win_rate": float(stats.get("Win Rate [%]", 0.0)),
        "return_pct": float(stats.get("Return [%]", 0.0)),
        "buy_hold_return_pct": float(stats.get("Buy & Hold Return [%]", 0.0)),
        "max_drawdown_pct": float(stats.get("Max. Drawdown [%]", 0.0)),
    }

    return result, trades, equity
