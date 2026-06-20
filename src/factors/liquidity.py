"""流动性因子 — 换手率、成交量等."""
import numpy as np
import pandas as pd


def turnover_factor(data: pd.DataFrame) -> pd.Series:
    """
    换手率因子（流动性溢价，低换手率效应）。
    数据列要求: 'turnover_rate'
    因子值 = -turnover_rate，换手率越低因子值越高。
    """
    turnover = data["turnover_rate"].copy()
    turnover = turnover.replace([np.inf, -np.inf], np.nan)
    result = -turnover
    result.name = None
    return result


def volume_ratio(data: pd.DataFrame) -> pd.Series:
    """成交量比因子：volume / volume_20d_mean.
    输入 multi-index 面板数据，需要 'vol' 列。
    """
    vol = data["vol"]
    results: dict[object, float] = {}
    for code, grp in vol.groupby("ts_code"):
        grp_sorted = grp.sort_index(level="trade_date")
        if len(grp_sorted) > 20:
            mean_20 = grp_sorted.iloc[-21:-1].mean()
            if mean_20 > 0:
                results[code] = float(grp_sorted.iloc[-1] / mean_20)
            else:
                results[code] = np.nan
        else:
            results[code] = np.nan
    return pd.Series(results)

