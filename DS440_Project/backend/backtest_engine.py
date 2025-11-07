# backtest_engine.py
from __future__ import annotations
import ast
import io
import textwrap
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import yfinance as yf

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import ta   # technical indicators

# ---------- Data ----------
def fetch_ohlcv(symbol: str, start: Optional[str]=None, end: Optional[str]=None,
                period: Optional[str]="6mo", interval: Optional[str]="1d") -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance with sensible caps for intraday intervals.
    Yahoo limits:
      - 1m  : ~7d
      - 2/5/15/30/90m : ~60d
      - 60m or 1h     : up to ~2y
      - 1d+           : years
    """
    def _days_for_period(p: str) -> int:
        # very rough conversion; just for comparing caps
        if p.endswith("d"): return int(p[:-1])
        if p.endswith("mo"): return int(p[:-2]) * 30
        if p.endswith("y"): return int(p[:-1]) * 365
        return 999999

    intraday_caps = {
        "1m": "7d",
        "2m": "60d",
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "90m": "60d",
        "60m": "730d",  # ~2y
        "1h": "730d",
    }

    # If user asked for an intraday interval with too-long period, clamp or explain
    if period and interval in intraday_caps:
        maxp = intraday_caps[interval]
        if _days_for_period(period) > _days_for_period(maxp):
            # You can EITHER silently clamp...
            # period = maxp
            # ...OR raise a helpful error (I prefer this):
            raise ValueError(
                f"Yahoo only provides about {maxp} of history for {interval} data. "
                f"Try period≤{maxp} or use a higher interval (e.g., 1d) for 2y."
            )
    kwargs = {"interval": interval, "auto_adjust": True, "progress": False, "threads": False}

    # Prefer period for simplicity; start/end with intraday often fails if too old
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

# ---------- Result packing ----------
def summarize_results(stats: pd.Series) -> Dict[str, Any]:
    def val(key, default=None):
        return (stats.get(key, default) if hasattr(stats, "get") else default)
    return {
        "trades": int(val('# Trades', 0)),
        "win_rate": float(val('% Win Rate', 0.0)),
        "expectancy": float(val('Expectancy', 0.0)),
        "avg_trade": float(val('Avg. Trade', 0.0)),
        "return_pct": float(val('Return [%]', 0.0)),
        "buy_hold_return_pct": float(val('Buy & Hold Return [%]', 0.0)),
        "max_drawdown_pct": float(val('Max. Drawdown [%]', 0.0)),
        "sharpe": float(val('Sharpe Ratio', 0.0)),
        "equity_curve": stats._equity_curve if hasattr(stats, "_equity_curve") else None,
    }

# ---------- Structured SPEC mode ----------
# Minimal, readable DSL via JSON the LLM can reliably emit.
# Example:
# {
#  "name":"SMA Cross with RSI Filter",
#  "params":{"fast":10,"slow":30,"rsi":14,"rsi_buy":55,"rsi_sell":70},
#  "entry":"crossover(SMA_fast, SMA_slow) and RSI > rsi_buy",
#  "exit":"RSI > rsi_sell or stop=-0.05 or take=0.10"
# }

def _compute_indicators(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    fast = int(params.get("fast", 10))
    slow = int(params.get("slow", 30))
    rsi_len = int(params.get("rsi", 14))
    df["SMA_fast"] = df["Close"].rolling(fast).mean()
    df["SMA_slow"] = df["Close"].rolling(slow).mean()
    df["RSI"] = ta.momentum.rsi(df["Close"], window=rsi_len)
    return df

def build_strategy_from_spec(spec: Dict[str, Any]) -> type[Strategy]:
    params = spec.get("params", {})
    entry_expr = spec.get("entry", "")
    exit_expr  = spec.get("exit", "")

    class SpecStrategy(Strategy):
        _params = params  # for transparency

        def init(self):
            self.sma_fast = self.I(SMA, self.data.Close, int(params.get("fast", 10)))
            self.sma_slow = self.I(SMA, self.data.Close, int(params.get("slow", 30)))
            # RSI via TA: use underlying close array
            import numpy as np
            close = pd.Series(self.data.Close)
            rsi_series = ta.momentum.rsi(close, window=int(params.get("rsi", 14))).fillna(50)
            self.rsi = self.I(lambda: rsi_series.values)

        def next(self):
            # Build a tiny expression context
            ctx = {
                "crossover": crossover,
                "SMA_fast": self.sma_fast,
                "SMA_slow": self.sma_slow,
                "RSI": self.rsi[-1],
                # last prices:
                "close": self.data.Close[-1],
                "position": self.position,  # can inspect size, pl, etc.
            }

            ctx.update({k: v for k, v in params.items() if isinstance(v, (int, float))})

            # Evaluate entry
            if self.position.size == 0:
                # Entry rule
                if entry_expr and _safe_eval(entry_expr, ctx):
                    self.buy()
            else:
                # Exit rule
                if exit_expr and _safe_eval(exit_expr, ctx):
                    self.position.close()

    SpecStrategy.__name__ = spec.get("name", "SpecStrategy").replace(" ", "_")
    return SpecStrategy

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

# ---------- CODE mode (sandboxed exec) ----------
STRATEGY_TEMPLATE = '''
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import ta
import pandas as pd

class UserStrategy(Strategy):
    # Optional params dict; engine will set from UI if provided
    params = {}

    def init(self):
        # Example indicators (edit/extend freely)
        self.sma_fast = self.I(SMA, self.data.Close, 10)
        self.sma_slow = self.I(SMA, self.data.Close, 30)

    def next(self):
        if self.position.size == 0 and crossover(self.sma_fast, self.sma_slow):
            self.buy()
        elif self.position.size > 0 and crossover(self.sma_slow, self.sma_fast):
            self.position.close()
'''

ALLOWED_IMPORTS = {"backtesting", "ta", "pandas", "numpy", "math"}

def _static_code_check(py_code: str) -> None:
    tree = ast.parse(py_code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.names[0].name.split(".")[0]
            if module not in ALLOWED_IMPORTS:
                raise ValueError(f"Import of '{module}' is not allowed")
        if isinstance(node, (ast.Expr,)) and isinstance(getattr(node, "value", None), ast.Call):
            # ban exec/eval/open/compile/os/system etc by name
            fn = getattr(node.value.func, "id", "") or getattr(node.value.func, "attr", "")
            if fn in {"exec", "eval", "open", "compile", "__import__", "system", "popen"}:
                raise ValueError(f"Call to '{fn}' is not allowed")
    # Must define UserStrategy
    if not any(isinstance(n, ast.ClassDef) and n.name == "UserStrategy" for n in tree.body):
        raise ValueError("Strategy code must define a class named 'UserStrategy'")

def build_strategy_from_code(py_code: str) -> type[Strategy]:
    _static_code_check(py_code)
    local_env: Dict[str, Any] = {}
    safe_globals = {"__builtins__": {}}
    exec(py_code, safe_globals, local_env)
    cls = local_env.get("UserStrategy")
    if not cls or not issubclass(cls, Strategy):
        raise ValueError("UserStrategy not found or not a Strategy subclass")
    return cls

# ---------- Runner ----------
def run_backtest(symbol: str,
                 mode: str,
                 strategy_spec_or_code: Dict[str, Any] | str,
                 start: Optional[str]=None, end: Optional[str]=None,
                 period: str="6mo", interval: str="1d",
                 cash: float=10000, commission: float=0.001
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
    # Save useful tables
    trades = bt._results._trades if hasattr(bt._results, "_trades") else pd.DataFrame()
    equity = bt._results._equity_curve if hasattr(bt._results, "_equity_curve") else pd.DataFrame()
    return summarize_results(stats), trades, equity

def plot_equity_png(symbol: str, Strat: type[Strategy], data: pd.DataFrame, path: str) -> str:
    bt = Backtest(data, Strat)
    bt.run()
    fig = bt.plot(open_browser=False)
    import matplotlib.pyplot as plt
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path