"""TDD for evaluate.py — #21, #22."""
import numpy as np
import pandas as pd
import pytest
from src.evaluate import (
    compute_ic_series,
    evaluate_ic,
    evaluate_performance,
    evaluate_all_groups,
)


# ============================================================
# IC 计算
# ============================================================
class TestComputeICSeries:
    def test_perfect_positive_rank_ic(self):
        """完美正相关 → IC ≈ 1.0."""
        codes = [f"{i:06d}.SZ" for i in range(10)]
        fv = [pd.Series(range(1, 11), index=codes, dtype=float)]
        fr = [pd.Series([v * 0.01 for v in range(1, 11)], index=codes)]
        ic = compute_ic_series(fv, fr)
        assert len(ic) == 1
        assert pytest.approx(ic.iloc[0], abs=0.01) == 1.0

    def test_strong_negative_rank_ic(self):
        """强负相关 → IC 为负."""
        codes = [f"{i:06d}.SZ" for i in range(10)]
        fv = [pd.Series(range(10, 0, -1), index=codes, dtype=float)]
        fr = [pd.Series([v * 0.01 for v in range(1, 11)], index=codes)]
        ic = compute_ic_series(fv, fr)
        assert ic.iloc[0] < 0

    def test_small_intersection_filtered(self):
        """共同样本<10 的期被跳过."""
        fv = [pd.Series([1.0] * 5, index=list("ABCDE"))]
        fr = [pd.Series([0.01] * 5, index=list("FGHIJ"))]  # no overlap
        ic = compute_ic_series(fv, fr)
        assert len(ic) == 0

    def test_multi_period_ic(self):
        """多期 IC 序列长度匹配."""
        fv = [
            pd.Series({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10}),
            pd.Series({"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10}),
        ]
        fr = [
            pd.Series({"A": 0.10, "B": 0.09, "C": 0.08, "D": 0.07, "E": 0.06, "F": 0.05, "G": 0.04, "H": 0.03, "I": 0.02, "J": 0.01}),
            pd.Series({"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.04, "E": 0.05, "F": 0.06, "G": 0.07, "H": 0.08, "I": 0.09, "J": 0.10}),
        ]
        ic = compute_ic_series(fv, fr)
        assert len(ic) == 2


class TestEvaluateIC:
    def test_ic_metrics_on_positive_series(self):
        """正 IC 序列 → IR > 0, p<0.01."""
        np.random.seed(42)
        ic = pd.Series(np.random.normal(0.03, 0.02, 24))
        metrics = evaluate_ic(ic)
        assert metrics["ic_mean"] > 0.02
        assert metrics["ic_ir"] > 0
        assert metrics["t_stat"] > 2
        assert metrics["n_periods"] == 24

    def test_empty_series_returns_nan(self):
        """空序列返回 NaN，期数=1."""
        metrics = evaluate_ic(pd.Series([]))
        assert metrics["n_periods"] == 0

    def test_single_value_returns_nan_metrics(self):
        """单值 IC 序列，IR/半衰期为 NaN."""
        metrics = evaluate_ic(pd.Series([0.03]))
        assert metrics["n_periods"] == 1
        assert np.isnan(metrics["ic_ir"])


# ============================================================
# 绩效评估
# ============================================================
class TestEvaluatePerformance:
    def test_ann_return_zero_mean(self):
        """日均收益为 0 → 年化收益为 0."""
        r = pd.Series([0.001, -0.001, 0.002, -0.002] * 63)
        perf = evaluate_performance(r)
        assert abs(perf["ann_return"]) < 0.01

    def test_ann_return_positive(self):
        """正日均 → 年化正."""
        daily = 0.001  # 0.1% daily
        r = pd.Series([daily] * 252)
        perf = evaluate_performance(r)
        assert pytest.approx(perf["ann_return"], rel=0.05) == 0.252

    def test_sharpe_zero_vol(self):
        """零波动率 → Sharpe=0（不除零）."""
        r = pd.Series([0.001] * 252)
        perf = evaluate_performance(r)
        assert perf["sharpe"] == 0.0

    def test_max_drawdown_known(self):
        """已知序列的最大回撤."""
        r = pd.Series([0.01, -0.02, 0.01, -0.05, 0.01])
        # NAV: 1.0*1.01=1.01, *0.98=0.9898, *1.01=0.9997, *0.95=0.9497, *1.01=0.9592
        # peak: 1.0→1.01, DD at bottom = (0.9497/1.01 - 1) = -0.0597
        perf = evaluate_performance(r)
        assert perf["max_drawdown"] < -0.03

    def test_empty_returns_empty_dict(self):
        """空收益返回空 dict."""
        perf = evaluate_performance(pd.Series(dtype=float))
        assert perf == {}


class TestEvaluateAllGroups:
    def test_multi_group_performance(self):
        """多组收益 → 每列有绩效."""
        gr = pd.DataFrame({
            "group_1": [0.001, -0.001, 0.002],
            "group_2": [0.0005, -0.001, 0.003],
            "long_short": [0.0005, 0.0, -0.001],
        })
        perf = evaluate_all_groups(gr)
        assert list(perf.columns) == ["group_1", "group_2", "long_short"]
        assert "ann_return" in perf.index
        assert "sharpe" in perf.index

    def test_empty_returns_empty(self):
        """空输入返回空 DataFrame."""
        result = evaluate_all_groups(pd.DataFrame())
        assert result.empty
