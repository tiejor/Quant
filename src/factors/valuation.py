"""估值因子 — pb, pe, ps 等."""
import numpy as np
import pandas as pd


def pb_factor(data: pd.DataFrame) -> pd.Series:
    """
    市净率倒数因子（低 PB 效应）。
    数据列要求: 'pb'
    因子值 = 1 / pb，PB 越低因子值越高。
    pb 为 0 或负值的股票返回 NaN。
    """
    pb = data["pb"].copy()
    pb = pb.replace([np.inf, -np.inf], np.nan)
    result = pd.Series(np.nan, index=pb.index)
    valid = pb > 0
    result.loc[valid] = 1.0 / pb.loc[valid]
    return result


def size_factor(data: pd.DataFrame) -> pd.Series:
    """
    小市值因子。
    数据列要求: 'circ_mv'（流通市值，单位：万元）
    因子值 = -ln(circ_mv)，市值越小因子值越高。
    circ_mv 为 0 或负值返回 NaN。
    """
    mv = data["circ_mv"].copy()
    mv = mv.replace([np.inf, -np.inf], np.nan)
    result = pd.Series(np.nan, index=mv.index)
    valid = mv > 0
    result.loc[valid] = -np.log(mv.loc[valid])
    return result


def pe_factor(data: pd.DataFrame) -> pd.Series:
    """
    PE 倒数因子（低 PE 效应）。
    数据列要求: 'pe_ttm'
    因子值 = 1 / pe_ttm，PE 越低因子值越高。
    pe_ttm 为 0 或负值的股票返回 NaN。
    """
    pe = data["pe_ttm"].copy()
    pe = pe.replace([np.inf, -np.inf], np.nan)
    result = pd.Series(np.nan, index=pe.index)
    valid = pe > 0
    result.loc[valid] = 1.0 / pe.loc[valid]
    return result
