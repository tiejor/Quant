"""因子模块 — 一个因子 = 一个 Python 函数."""
from src.factors.valuation import pb_factor, size_factor, pe_factor
from src.factors.liquidity import turnover_factor, volume_ratio
from src.factors.momentum import (
    momentum_20,
    momentum_60,
    reversal_5,
    reversal_10,
)
from src.factors.volatility import volatility_20, volatility_60

FACTOR_REGISTRY: dict[str, object] = {
    "pb_factor": pb_factor,
    "size_factor": size_factor,
    "pe_factor": pe_factor,
    "turnover_factor": turnover_factor,
    "momentum_20": momentum_20,
    "momentum_60": momentum_60,
    "reversal_5": reversal_5,
    "reversal_10": reversal_10,
    "volatility_20": volatility_20,
    "volatility_60": volatility_60,
    "volume_ratio": volume_ratio,
}


# 需要历史面板数据（multi-index）的因子
FACTOR_PANEL_KEYS = {
    "momentum_20", "momentum_60",
    "reversal_5", "reversal_10",
    "volatility_20", "volatility_60",
    "volume_ratio",
}


def get_factor(name: str):
    """根据名称获取因子函数（大小写不敏感）"""
    name_lower = name.lower()
    if name_lower not in FACTOR_REGISTRY:
        raise ValueError(f"未知因子: {name}，可用: {list(FACTOR_REGISTRY.keys())}")
    return FACTOR_REGISTRY[name_lower]
