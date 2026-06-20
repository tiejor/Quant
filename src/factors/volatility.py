"""波动率因子 — volatility 等."""
import numpy as np
import pandas as pd


def _volatility(close: pd.Series, window: int) -> float:
    """日收益率的滚动标准差."""
    if len(close) <= window:
        return np.nan
    pct = close.pct_change().dropna()
    if len(pct) < window:
        return np.nan
    return float(pct.iloc[-window:].std(ddof=1))


def volatility_20(data: pd.DataFrame) -> pd.Series:
    """20 日波动率因子."""
    results: dict[object, float] = {}
    for code, grp in data["close"].groupby("ts_code"):
        grp_sorted = grp.sort_index(level="trade_date")
        results[code] = _volatility(grp_sorted, 20)
    return pd.Series(results)


def volatility_60(data: pd.DataFrame) -> pd.Series:
    """60 日波动率因子."""
    results: dict[object, float] = {}
    for code, grp in data["close"].groupby("ts_code"):
        grp_sorted = grp.sort_index(level="trade_date")
        results[code] = _volatility(grp_sorted, 60)
    return pd.Series(results)
