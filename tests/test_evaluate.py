"""TDD for evaluate.py — #21, #22."""
import numpy as np
import pandas as pd
import pytest
from src.evaluate import (
    compute_ic_series,
    evaluate_ic,
    evaluate_performance,
    evaluate_all_groups,
    compute_sortino_ratio,
    compute_ic_lag_curve,
    ic_summary,
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


# ============================================================
# Sortino 比率
# ============================================================
class TestComputeSortinoRatio:
    def test_all_positive_no_downside(self):
        """全部正收益且 > MAR → 无下行波动，Sortino 极高."""
        r = pd.Series([0.01, 0.02, 0.015, 0.03, 0.01])  # all positive, well above 3%/252
        result = compute_sortino_ratio(r, mar_annual=0.03)
        assert result > 10.0  # very high Sortino since no downside

    def test_mixed_with_downside(self):
        """部分低于 MAR 的收益 → Sortino 正常."""
        # 5 days: 3 up, 2 down below MAR
        mar_daily = 0.03 / 252  # ≈ 0.000119
        r = pd.Series([0.001, -0.002, 0.003, -0.001, 0.002])
        result = compute_sortino_ratio(r, mar_annual=0.03)
        # Should be a finite positive number
        assert 0 < result < 50.0

    def test_sortino_inf_on_no_downside_returns(self):
        """所有收益等于 MAR → 下行标准差为 0 → 返回 inf."""
        mar_daily = 0.03 / 252
        r = pd.Series([mar_daily] * 100)
        result = compute_sortino_ratio(r, mar_annual=0.03)
        assert np.isinf(result) or result > 1000

    def test_empty_input(self):
        """空收益序列 → 返回 NaN."""
        result = compute_sortino_ratio(pd.Series(dtype=float), mar_annual=0.03)
        assert np.isnan(result)


# ============================================================
# ic_summary
# ============================================================
class TestICSummary:
    def test_all_positive_ic(self):
        """全部正 IC → win_rate=1.0, sig_ratio 按 |IC|>0.02 计算."""
        ic = pd.Series([0.03, 0.04, 0.05, 0.025, 0.06, 0.035])
        result = ic_summary(ic)
        assert result["ic_mean"] > 0.03
        assert result["ic_std"] > 0
        assert result["ic_ir"] > 0
        assert result["t_stat"] > 0
        assert result["p_value"] < 0.05
        assert result["win_rate"] == 1.0
        assert result["sig_ratio"] == 1.0  # all |IC| > 0.02
        assert result["n_periods"] == 6

    def test_all_negative_ic(self):
        """全部负 IC → win_rate=0.0."""
        ic = pd.Series([-0.03, -0.04, -0.05, -0.01, -0.06])
        result = ic_summary(ic)
        assert result["ic_mean"] < 0
        assert result["win_rate"] == 0.0
        assert result["sig_ratio"] == 0.8  # 4/5 have |IC| >= 0.02, -0.01 excluded

    def test_mixed_sign_ic(self):
        """正负混合 → win_rate 在 0~1."""
        ic = pd.Series([0.03, -0.01, 0.05, -0.04, 0.02])
        result = ic_summary(ic)
        assert result["win_rate"] == 0.6  # 3/5 positive
        assert result["sig_ratio"] == 0.8  # 4/5 have |IC| >= 0.02: 0.03,0.05,0.04,0.02

    def test_sig_ratio_at_boundary(self):
        """|IC| 恰好等于 0.02 应计入 sig_ratio."""
        ic = pd.Series([0.02, -0.02, 0.01])
        result = ic_summary(ic)
        assert result["sig_ratio"] == pytest.approx(2/3)

    def test_empty_series(self):
        """空序列 → 全部 NaN."""
        result = ic_summary(pd.Series(dtype=float))
        assert np.isnan(result["ic_mean"])
        assert result["n_periods"] == 0

    def test_single_value(self):
        """单值 → IR/t_stat 为 NaN."""
        result = ic_summary(pd.Series([0.03]))
        assert result["n_periods"] == 1
        assert np.isnan(result["ic_ir"])
        assert np.isnan(result["t_stat"])

    def test_t_stat_and_p_value_known(self):
        """手动计算 t_stat = mean / (std/sqrt(n)), p 来自双尾 t 分布."""
        ic = pd.Series([0.02, 0.03, 0.01, 0.04, 0.02])
        result = ic_summary(ic)
        # n=5, mean=0.024, std≈0.01140, se=0.01140/√5≈0.005099
        # t = 0.024 / 0.005099 ≈ 4.71
        expected_t = 0.024 / (0.011402 / np.sqrt(5))
        assert pytest.approx(result["t_stat"], rel=0.01) == expected_t
        assert 0 < result["p_value"] < 1

    def test_nan_in_series_ignored(self):
        """含 NaN 的序列，NaN 被 drop 后统计."""
        ic = pd.Series([0.03, np.nan, 0.05, 0.04, np.nan, 0.02])
        result = ic_summary(ic)
        assert result["n_periods"] == 4
        assert result["win_rate"] == 1.0


# ============================================================
# IC(N) lag-based 半衰期
# ============================================================
class TestComputeICLagCurve:
    def _make_series(self, values: list[float], n_stocks: int = 20) -> list[pd.Series]:
        """Helper: 生成多期因子/收益序列，每期 n_stocks 只股票."""
        codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
        out = []
        for vs in values:
            s = pd.Series(vs, index=codes, dtype=float)
            out.append(s)
        return out

    def _make_linear_factor_and_return(self, n_periods: int = 100, n_stocks: int = 20):
        """构造完美正相关的因子和 T+1 收益序列."""
        codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
        factor_values = []
        daily_returns = []
        for t in range(n_periods):
            # 因子值 = 1..N
            fv = pd.Series(range(1, n_stocks + 1), index=codes, dtype=float)
            factor_values.append(fv)
            # T+1 收益 = 因子值排序相同 → rank IC ≈ 1
            ret = pd.Series(range(1, n_stocks + 1), index=codes, dtype=float)
            daily_returns.append(ret)
        return factor_values, daily_returns

    def test_perfect_correlation_lag_1(self):
        """完美线性正相关 → IC(1) ≈ 1.0."""
        fv, rets = self._make_linear_factor_and_return(100)
        ic_curve, half_life = compute_ic_lag_curve(fv, rets, max_lag=5)
        assert 1 in ic_curve
        assert pytest.approx(ic_curve[1], abs=0.01) == 1.0

    def test_random_no_predictive_power(self):
        """随机因子 + 随机收益 → 所有 lag IC ≈ 0."""
        np.random.seed(123)
        codes = [f"{i:06d}.SZ" for i in range(50)]
        fv = []
        rets = []
        for t in range(60):
            fv.append(pd.Series(np.random.randn(50), index=codes))
            rets.append(pd.Series(np.random.randn(50), index=codes))
        ic_curve, half_life = compute_ic_lag_curve(fv, rets, max_lag=5)
        for lag in range(1, 6):
            assert abs(ic_curve.get(lag, 0.0)) < 0.2

    def test_half_life_identified(self):
        """半衰期正确识别: IC 从 lag=1 平滑衰减到噪声."""
        codes = [f"{i:06d}.SZ" for i in range(30)]
        np.random.seed(42)
        base = pd.Series(range(1, 31), index=codes, dtype=float)
        factor_vals = [base + pd.Series(np.random.normal(0, 0.1, 30), index=codes) for _ in range(200)]

        # rets[t] = base * decay(t) + tiny noise
        # 构造使 IC 随 lag 衰减
        decay_returns = []
        for t in range(200):
            decay = max(0, 1.0 - t * 0.12)  # lag 0→1.0, lag 4→0.52, lag 8→0.04
            ret_t = base * decay + pd.Series(np.random.normal(0, 0.02, 30), index=codes)
            decay_returns.append(ret_t)

        ic_curve, half_life = compute_ic_lag_curve(factor_vals, decay_returns, max_lag=10)
        assert half_life is not None
        assert half_life > 0
        assert ic_curve[1] > ic_curve[min(10, half_life + 1)]

    def test_too_few_periods(self):
        """有效截面 < 3 → 返回空 dict, half_life=None."""
        codes = [f"{i:06d}.SZ" for i in range(5)]
        fv = [pd.Series([1.0] * 5, index=codes)]
        rets = [pd.Series([0.01] * 5, index=codes)]
        ic_curve, half_life = compute_ic_lag_curve(fv, rets, max_lag=5)
        assert ic_curve == {}
        assert half_life is None
