"""动量/反转因子 — momentum, reversal 等."""
import numpy as np
import pandas as pd


def _momentum(close: pd.Series, window: int) -> float:
    """通用动量计算：close_now / close_{window}_ago - 1."""
    if len(close) <= window:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-window - 1] - 1)


def _factor_from_panel(data: pd.DataFrame, col: str, fn, window: int) -> pd.Series:
    """从 multi-index 面板数据提取因子值."""
    series = data[col]
    results: dict[object, float] = {}
    for code, grp in series.groupby("ts_code"):
        grp_sorted = grp.sort_index(level="trade_date")
        results[code] = fn(grp_sorted, window)
    return pd.Series(results)


def momentum_20(data: pd.DataFrame) -> pd.Series:
    """20 日动量因子：close / close_20d_ago - 1."""
    return _factor_from_panel(data, "close", _momentum, 20)


def momentum_60(data: pd.DataFrame) -> pd.Series:
    """60 日动量因子."""
    return _factor_from_panel(data, "close", _momentum, 60)


def _reversal(close: pd.Series, window: int) -> float:
    """反转因子：-pct_change."""
    if len(close) <= window:
        return np.nan
    return float(-(close.iloc[-1] / close.iloc[-window - 1] - 1))


def reversal_5(data: pd.DataFrame) -> pd.Series:
    """5 日反转因子."""
    return _factor_from_panel(data, "close", _reversal, 5)


def reversal_10(data: pd.DataFrame) -> pd.Series:
    """10 日反转因子."""
    return _factor_from_panel(data, "close", _reversal, 10)
