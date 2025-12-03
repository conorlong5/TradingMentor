import backtrader as bt
import pandas as pd


class JSONStrategy(bt.Strategy):
    """
    Simple moving-average crossover strategy driven by JSON-like params.
    Expects params:
        - fast: int, fast SMA window
        - slow: int, slow SMA window
    """

    params = dict(
        fast=10,
        slow=30,
    )

    def __init__(self):
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.p.fast
        )
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.p.slow
        )

        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.pnl_list = []

    def next(self):
        if self.sma_fast[0] > self.sma_slow[0] and not self.position:
            self.buy()
        elif self.sma_fast[0] < self.sma_slow[0] and self.position:
            self.sell()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades += 1
            pnl = trade.pnlcomm
            self.pnl_list.append(pnl)

            if pnl > 0:
                self.wins += 1
            else:
                self.losses += 1


def run_backtrader_backtest(df: pd.DataFrame, strategy_params: dict):
    """
    Run a Backtrader backtest on a price dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Price data with at least: Open, High, Low, Close, Volume, and a datetime index.
    strategy_params : dict
        Dictionary that can include:
            - "fast": int (fast SMA window)
            - "slow": int (slow SMA window)
            - "cash": float (starting portfolio value)

    Returns
    -------
    dict
        {
            "final_value": float,
            "trades": int,
            "win_rate": float,
            "avg_return": float
        }
    """

    fast = int(strategy_params.get("fast", 10))
    slow = int(strategy_params.get("slow", 30))
    cash = float(strategy_params.get("cash", 10000))

    if not isinstance(df.index, (pd.DatetimeIndex,)):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    cerebro = bt.Cerebro()

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    cerebro.addstrategy(JSONStrategy, fast=fast, slow=slow)

    cerebro.broker.set_cash(cash)

    result = cerebro.run()
    strat_obj = result[0]

    final_value = cerebro.broker.getvalue()
    trades = strat_obj.trades

    if trades > 0:
        win_rate = strat_obj.wins / trades * 100.0
        avg_return = sum(strat_obj.pnl_list) / trades
    else:
        win_rate = 0.0
        avg_return = 0.0

    return {
        "final_value": final_value,
        "trades": trades,
        "win_rate": win_rate,
        "avg_return": avg_return,
    }
