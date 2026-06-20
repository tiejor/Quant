"""因子模块 — 向后兼容重新导出。新代码从 src.factors 包导入."""
from src.factors import (
    pb_factor, size_factor, pe_factor, turnover_factor,
    momentum_20, momentum_60,
    reversal_5, reversal_10,
    volatility_20, volatility_60, volume_ratio,
    FACTOR_REGISTRY, FACTOR_PANEL_KEYS, get_factor,
)
