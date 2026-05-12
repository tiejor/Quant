"""
因子模块。
一个因子 = 一个 Python 函数。
输入：每日截面的行情与基本面 DataFrame（多股票、单日）
输出：因子值 Series（index = ts_code）
用户自定义因子在此文件中添加新函数即可。
"""

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
    pb = pb[pb > 0]
    return 1.0 / pb


def size_factor(data: pd.DataFrame) -> pd.Series:
    """
    小市值因子。
    数据列要求: 'circ_mv'（流通市值，单位：万元）
    因子值 = -ln(circ_mv)，市值越小因子值越高。
    circ_mv 为 0 或负值返回 NaN。
    """
    mv = data["circ_mv"].copy()
    mv = mv.replace([np.inf, -np.inf], np.nan)
    mv = mv[mv > 0]
    return -np.log(mv)


def turnover_factor(data: pd.DataFrame) -> pd.Series:
    """
    换手率因子（流动性溢价，低换手率效应）。
    数据列要求: 'turnover_rate'
    因子值 = -turnover_rate，换手率越低因子值越高。
    """
    turnover = data["turnover_rate"].copy()
    turnover = turnover.replace([np.inf, -np.inf], np.nan)
    return -turnover


# 因子注册表：因子名 → 因子函数
FACTOR_REGISTRY = {
    "pb_factor": pb_factor,
    "size_factor": size_factor,
    "turnover_factor": turnover_factor,
}


def get_factor(name: str):
    """根据名称获取因子函数"""
    if name not in FACTOR_REGISTRY:
        raise ValueError(f"未知因子: {name}，可用: {list(FACTOR_REGISTRY.keys())}")
    return FACTOR_REGISTRY[name]
