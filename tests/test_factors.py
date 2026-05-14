"""TDD for factors.py — #9."""
import os
import numpy as np
import pandas as pd
import pytest
import duckdb
from conftest import requires_db, DB_PATH
from src.factors import pb_factor, size_factor, turnover_factor, get_factor, FACTOR_REGISTRY


# ============================================================
# pb_factor
# ============================================================
class TestPbFactor:
    def test_normal_pb_returns_inverse(self):
        """pb_factor returns 1/pb for normal positive values."""
        data = pd.DataFrame({"pb": [1.0, 2.0, 4.0]}, index=["A", "B", "C"])
        result = pb_factor(data)
        expected = pd.Series([1.0, 0.5, 0.25], index=["A", "B", "C"])
        pd.testing.assert_series_equal(result, expected, check_dtype=False)

    def test_pb_zero_returns_nan(self):
        """pb=0 returns NaN."""
        data = pd.DataFrame({"pb": [0.0, 1.0]}, index=["A", "B"])
        result = pb_factor(data)
        assert np.isnan(result["A"])
        assert result["B"] == 1.0

    def test_pb_negative_returns_nan(self):
        """pb < 0 returns NaN."""
        data = pd.DataFrame({"pb": [-1.0, 2.0]}, index=["A", "B"])
        result = pb_factor(data)
        assert np.isnan(result["A"])
        assert result["B"] == 0.5

    def test_pb_nan_returns_nan(self):
        """pb=NaN returns NaN."""
        data = pd.DataFrame({"pb": [np.nan, 2.0]}, index=["A", "B"])
        result = pb_factor(data)
        assert np.isnan(result["A"])
        assert result["B"] == 0.5

    def test_pb_inf_returns_nan(self):
        """pb=inf returns NaN."""
        data = pd.DataFrame({"pb": [np.inf, -np.inf, 2.0]}, index=["A", "B", "C"])
        result = pb_factor(data)
        assert np.isnan(result["A"])
        assert np.isnan(result["B"])
        assert result["C"] == 0.5


# ============================================================
# size_factor
# ============================================================
class TestSizeFactor:
    def test_normal_size_returns_neg_log(self):
        data = pd.DataFrame({"circ_mv": [100, 10000, 1000000]}, index=["A", "B", "C"])
        result = size_factor(data)
        expected = pd.Series(-np.log([100, 10000, 1000000]), index=["A", "B", "C"])
        pd.testing.assert_series_equal(result, expected, check_dtype=False)

    def test_mv_zero_returns_nan(self):
        data = pd.DataFrame({"circ_mv": [0, 1000]}, index=["A", "B"])
        result = size_factor(data)
        assert np.isnan(result["A"])
        assert np.isfinite(result["B"])

    def test_mv_negative_returns_nan(self):
        data = pd.DataFrame({"circ_mv": [-500, 1000]}, index=["A", "B"])
        result = size_factor(data)
        assert np.isnan(result["A"])
        assert np.isfinite(result["B"])

    def test_mv_nan_returns_nan(self):
        data = pd.DataFrame({"circ_mv": [np.nan, 1000]}, index=["A", "B"])
        result = size_factor(data)
        assert np.isnan(result["A"])
        assert np.isfinite(result["B"])

    def test_mv_inf_returns_nan(self):
        data = pd.DataFrame({"circ_mv": [np.inf, -np.inf, 1000]}, index=["A", "B", "C"])
        result = size_factor(data)
        assert np.isnan(result["A"])
        assert np.isnan(result["B"])
        assert np.isfinite(result["C"])


# ============================================================
# turnover_factor
# ============================================================
class TestTurnoverFactor:
    def test_normal_turnover_returns_negative(self):
        data = pd.DataFrame({"turnover_rate": [1.5, 3.0, 0.5]}, index=["A", "B", "C"])
        result = turnover_factor(data)
        expected = pd.Series([-1.5, -3.0, -0.5], index=["A", "B", "C"])
        pd.testing.assert_series_equal(result, expected, check_dtype=False)

    def test_turnover_zero_ok(self):
        data = pd.DataFrame({"turnover_rate": [0.0, 3.0]}, index=["A", "B"])
        result = turnover_factor(data)
        assert result["A"] == 0.0
        assert result["B"] == -3.0

    def test_turnover_nan_returns_nan(self):
        data = pd.DataFrame({"turnover_rate": [np.nan, 3.0]}, index=["A", "B"])
        result = turnover_factor(data)
        assert np.isnan(result["A"])
        assert result["B"] == -3.0

    def test_turnover_inf_returns_nan(self):
        data = pd.DataFrame({"turnover_rate": [np.inf, -np.inf, 3.0]}, index=["A", "B", "C"])
        result = turnover_factor(data)
        assert np.isnan(result["A"])
        assert np.isnan(result["B"])
        assert result["C"] == -3.0


# ============================================================
# FACTOR_REGISTRY & get_factor
# ============================================================
class TestRegistry:
    def test_registry_contains_all_factors(self):
        assert set(FACTOR_REGISTRY.keys()) == {"pb_factor", "size_factor", "turnover_factor"}

    def test_get_factor_returns_callable(self):
        assert get_factor("pb_factor") is pb_factor
        assert get_factor("size_factor") is size_factor
        assert get_factor("turnover_factor") is turnover_factor

    def test_get_factor_unknown_raises(self):
        with pytest.raises(ValueError, match="未知因子"):
            get_factor("nonexistent")

    def test_get_factor_case_insensitive(self):
        """FACTOR_REGISTRY keys are lowercase; get_factor normalises case."""
        assert get_factor("PB_FACTOR") is pb_factor
        assert get_factor("Size_Factor") is size_factor
        assert get_factor("TURNOVER_FACTOR") is turnover_factor


# ============================================================
# 真实数据快照
# ============================================================
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "data")


@requires_db
class TestSnapshot:
    """用真实数据跑全部因子，保存快照，后续对比防 regression."""

    def _get_available_dates(self) -> list[str]:
        db = duckdb.connect(DB_PATH, read_only=True)
        df = db.execute("SELECT DISTINCT trade_date FROM daily_basic ORDER BY trade_date").df()
        db.close()
        return df["trade_date"].tolist()

    def _get_factor_data(self, date: str) -> pd.DataFrame:
        db = duckdb.connect(DB_PATH, read_only=True)
        df = db.execute(
            "SELECT ts_code, pb, total_mv, circ_mv, turnover_rate "
            "FROM daily_basic WHERE trade_date = ?",
            [date],
        ).df()
        db.close()
        return df.set_index("ts_code") if not df.empty else pd.DataFrame()

    def test_all_factors_run_on_real_data(self):
        """三个因子在所有可用截面上不抛异常，产出非空."""
        dates = self._get_available_dates()
        assert len(dates) >= 1, "daily_basic 表为空，先运行 pipeline"
        for d in dates:
            data = self._get_factor_data(d)
            r_pb = pb_factor(data)
            r_sz = size_factor(data)
            r_to = turnover_factor(data)
            assert not r_pb.empty
            assert not r_sz.empty
            assert not r_to.empty
            assert len(r_pb) == len(r_sz) == len(r_to)

    def test_snapshot_pb_factor(self):
        """pb_factor 快照：最早可用截面，对比保存的 csv."""
        dates = self._get_available_dates()
        assert len(dates) >= 1, "daily_basic 表为空，先运行 pipeline"
        snap_date = dates[0]

        data = self._get_factor_data(snap_date)
        result = pb_factor(data).dropna().sort_index()
        assert len(result) > 0

        snap_path = os.path.join(SNAPSHOT_DIR, f"pb_factor_{snap_date}.csv")
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        if not os.path.exists(snap_path):
            result.to_csv(snap_path, float_format="%.12f")
            pytest.skip(f"快照已保存到 {snap_path}，下次运行将做对比")

        expected = pd.read_csv(snap_path, index_col=0).squeeze(axis=1)
        expected.name = None
        common = result.index.intersection(expected.index)
        pd.testing.assert_series_equal(
            result.loc[common], expected.loc[common], check_dtype=False, rtol=1e-10
        )
