from __future__ import annotations
import ast
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import yfinance as yf

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import ta

def fetch_ohlcv(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = "6mo",
    interval: Optional[str] = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance with sensible caps for intraday intervals.
    Yahoo limits:
      - 1m  : ~7d
      - 2/5/15/30/90m : ~60d
      - 60m or 1h     : up to ~2y
      - 1d+           : years
    """
    def _days_for_period(p: str) -> int:
        if p.endswith("d"):
            return int(p[:-1])
        if p.endswith("mo"):
            return int(p[:-2]) * 30
        if p.endswith("y"):
            return int(p[:-1]) * 365
        if p == "max":
            return 999999
        return 999999

    intraday_caps = {
        "1m": "7d",
        "2m": "60d",
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "90m": "60d",
        "60m": "730d",  
        "1h": "730d",
    }

    if period and interval in intraday_caps:
        maxp = intraday_caps[interval]
        if _days_for_period(period) > _days_for_period(maxp):
            raise ValueError(
                f"Yahoo only provides about {maxp} of history for {interval} data. "
                f"Try period≤{maxp} or use a higher interval (e.g., 1d) for longer periods."
            )

    kwargs = {"interval": interval, "auto_adjust": True, "progress": False, "threads": False}
    
    if start or end:
        data = yf.download(symbol, start=start, end=end, **kwargs)
    else:
        data = yf.download(symbol, period=period or "6mo", **kwargs)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    data.dropna(inplace=True)

    if data.empty:
        raise ValueError(f"No data for {symbol} (period={period}, interval={interval}).")

    return data


def summarize_results(stats: pd.Series) -> Dict[str, Any]:
    def val(key, default=None):
        return (stats.get(key, default) if hasattr(stats, "get") else default)

    return {
        "trades": int(val('# Trades', 0)),
        "win_rate": float(val('Win Rate [%]', 0.0)),
        "expectancy": float(val('Expectancy', 0.0)),
        "avg_trade": float(val('Avg. Trade', 0.0)),
        "return_pct": float(val('Return [%]', 0.0)),
        "buy_hold_return_pct": float(val('Buy & Hold Return [%]', 0.0)),
        "max_drawdown_pct": float(val('Max. Drawdown [%]', 0.0)),
        "sharpe": float(val('Sharpe Ratio', 0.0)),
        "equity_curve": stats._equity_curve if hasattr(stats, "_equity_curve") else None,
    }


def _safe_eval(expr: str, context: Dict[str, Any]) -> bool:
    """
    Very small, safe evaluator for boolean expressions of indicators.
    Allowed names must be in `context`. No function calls besides `crossover`.
    """
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)) and not (
            isinstance(node, ast.Call) and getattr(node.func, "id", "") == "crossover"
        ):
            raise ValueError("Disallowed expression in rules")
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.FunctionDef, ast.ClassDef)):
            raise ValueError("Not allowed in rules")
    return bool(eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, context))

def build_strategy_from_spec(spec: Dict[str, Any]) -> type[Strategy]:
    params = spec.get("params", {})
    entry_expr = spec.get("entry", "")
    exit_expr = spec.get("exit", "")

    class SpecStrategy(Strategy):
        _params = params  

        def init(self):
            self.sma_fast = self.I(SMA, self.data.Close, int(params.get("fast", 10)))
            self.sma_slow = self.I(SMA, self.data.Close, int(params.get("slow", 30)))

            import numpy as np
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
                "position": self.position,  
            }
            ctx.update({k: v for k, v in params.items() if isinstance(v, (int, float))})

            if self.position.size == 0:
                if entry_expr and _safe_eval(entry_expr, ctx):
                    self.buy()
            else:
                if exit_expr and _safe_eval(exit_expr, ctx):
                    self.position.close()

    SpecStrategy.__name__ = spec.get("name", "SpecStrategy").replace(" ", "_")
    return SpecStrategy


def build_strategy_from_code(py_code: str) -> type[Strategy]:
    """Not used by your UI but left here safely."""
    local_env = {}
    exec(py_code, {"__builtins__": {}}, local_env)
    Strat = local_env.get("UserStrategy")
    if not Strat:
        raise ValueError("UserStrategy not found")
    return Strat


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
    elif mode == "code":
        Strat = build_strategy_from_code(strategy_spec_or_code)
    else:
        raise ValueError("mode must be 'spec' or 'code'")

    bt = Backtest(data, Strat, cash=cash, commission=commission)

    stats = bt.run()

    trades = bt._results._trades if hasattr(bt._results, "_trades") else pd.DataFrame()
    equity = bt._results._equity_curve if hasattr(bt._results, "_equity_curve") else pd.DataFrame()

    return summarize_results(stats), trades, equity
