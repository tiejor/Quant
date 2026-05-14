"""TDD for backtest.py helper functions — #19."""
import numpy as np
import pandas as pd
import pytest
import duckdb
from conftest import requires_db
from src.backtest import mad_standardize, assign_groups, run_backtest


# ============================================================
# db_path 注入
# ============================================================
class TestRunBacktestWithTmpDb:
    def _seed_tmp_db(self, tmp_db):
        """在 tmp_db 中构造 2 个月 × 5 只股票的完整数据."""
        from src.pipeline import init_tables
        init_tables(db_path=tmp_db)
        db = duckdb.connect(tmp_db)

        # 交易日历：2024年1-2月，简化到每周2天
        trading_days = [
            (20240102, 1), (20240103, 1),
            (20240108, 1), (20240110, 1),
            (20240115, 1), (20240117, 1),
            (20240122, 1), (20240124, 1),
            (20240129, 1), (20240131, 1),
            (20240201, 1), (20240205, 1),
            (20240207, 1), (20240212, 1),
            (20240214, 1), (20240219, 1),
            (20240221, 1), (20240226, 1),
            (20240228, 1), (20240229, 1),
        ]
        for d, is_open in trading_days:
            db.execute("INSERT INTO trade_cal VALUES (?, ?, 'SSE')", (str(d), is_open))

        # 股票基础信息
        stocks = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH", "600519.SH"]
        for c in stocks:
            db.execute("INSERT INTO stock_basic VALUES (?, ?, ?, ?, ?)",
                       (c, c, "金融", "20200101", None))

        # 日线行情：每只股票每天有涨跌
        np.random.seed(42)
        for d, is_open in trading_days:
            for c in stocks:
                pct = np.random.normal(0, 2)  # ~0%均值的日收益率
                db.execute("""
                    INSERT INTO daily (ts_code, trade_date, open, high, low, close, pre_close, pct_chg, vol, amount)
                    VALUES (?, ?, 10, 11, 9, 10, 10, ?, 10000, 100000)
                """, (c, str(d), pct))

        # 日基本面
        for d, is_open in trading_days:
            for c in stocks:
                db.execute("""
                    INSERT INTO daily_basic (ts_code, trade_date, pb, total_mv, circ_mv, turnover_rate)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (c, str(d), np.random.uniform(0.5, 5), 1e6, 5e5, np.random.uniform(0, 10)))

        db.close()
        return stocks, trading_days

    def test_run_backtest_with_tmp_db(self, tmp_db):
        """run_backtest 可接受 db_path 指向临时库."""
        self._seed_tmp_db(tmp_db)

        result = run_backtest(
            start_date="20240101",
            end_date="20240228",
            factor_name="pb_factor",
            n_groups=2,
            standardize=False,
            db_path=tmp_db,
            silent=True,
        )

        assert "group_returns" in result
        assert not result["group_returns"].empty
        assert "group_nav" in result
        assert "factor_values" in result
        assert "forward_returns" in result


# ============================================================
# IC 数据产出
# ============================================================
@requires_db
class TestBacktestICOutput:
    def test_outputs_factor_values_and_forward_returns(self):
        """回测产出 factor_values 和 forward_returns，长度匹配月数."""
        result = run_backtest(
            start_date="20240101",
            end_date="20240630",
            factor_name="pb_factor",
            n_groups=10,
            silent=True,
        )

        assert "factor_values" in result
        assert "forward_returns" in result

        fv = result["factor_values"]
        fr = result["forward_returns"]
        assert len(fv) > 0
        assert len(fv) == len(fr)

        # 每项格式：(date_str, Series)
        for date_str, values in fv:
            assert isinstance(date_str, str)
            assert isinstance(values, pd.Series)
            assert len(values) > 0


@requires_db
class TestRunBacktestSmoke:
    def test_run_backtest_produces_valid_output(self):
        """回测产出 group_returns 和 group_nav，含各组 + 多空列."""
        result = run_backtest(
            start_date="20240101",
            end_date="20240630",
            factor_name="pb_factor",
            n_groups=10,
            silent=True,
        )

        gr = result["group_returns"]
        nav = result["group_nav"]

        assert not gr.empty
        assert not nav.empty
        assert len(gr) == len(nav)
        assert "group_1" in gr.columns
        assert f"group_10" in gr.columns
        assert "long_short" in gr.columns
        assert gr.index.is_monotonic_increasing

    def test_run_pathways_produces_multiple_tracks(self):
        """多轨道回测产出 n 条独立轨道."""
        from src.backtest import run_pathways
        results = run_pathways(
            start_date="20240101",
            end_date="20240630",
            factor_name="pb_factor",
            n_pathways=3,
            n_groups=10,
        )

        assert len(results) == 3
        for i, r in enumerate(results):
            assert r["pathway"] == i
            assert "group_returns" in r
            assert "group_nav" in r
            assert not r["group_returns"].empty


# ============================================================
# 纯函数测试
# ============================================================
class TestMadStandardize:
    def test_normal_standardization_zero_median(self):
        """MAD 标准化后中位数应为 0."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        result = mad_standardize(s)
        assert abs(result.median()) < 1e-10

    def test_all_same_values_unchanged(self):
        """MAD=0 时返回原序列，不除零."""
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = mad_standardize(s)
        pd.testing.assert_series_equal(result, s)

    def test_nan_preserved(self):
        """NaN 保留在输出中，非 NaN 值正常标准化."""
        s = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
        result = mad_standardize(s)
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[3])
        non_nan = result.dropna()
        assert abs(non_nan.median()) < 1e-10
        assert len(non_nan) == 3


# ============================================================
# assign_groups
# ============================================================
class TestAssignGroups:
    def test_normal_grouping_20_stocks_10_groups(self):
        """20 只股票分 10 组，每组约 2 只，最高因子值进第 1 组."""
        np.random.seed(42)
        codes = [f"{i:06d}.SZ" for i in range(20)]
        values = pd.Series(np.random.randn(20), index=codes).sort_values(ascending=False)
        groups = assign_groups(values, n_groups=10)
        assert len(groups) == 20
        assert groups.nunique() == 10
        # 最高因子值在组 1
        assert groups.iloc[0] == 1

    def test_nan_excluded_from_groups(self):
        """NaN 值被排除，不在任何组中."""
        s = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0], index=list("ABCDE"))
        groups = assign_groups(s, n_groups=3)
        assert "B" not in groups.index
        assert "D" not in groups.index
        assert len(groups) == 3

    def test_qcut_fallback_to_rank(self):
        """构造重复值使 qcut 失败，验证 rank 回退路径."""
        s = pd.Series([1.0] * 10 + [2.0] * 10, index=[f"{i:06d}.SZ" for i in range(20)])
        groups = assign_groups(s, n_groups=10)
        assert len(groups) == 20
        assert groups.nunique() == 10

    def test_empty_input_returns_empty(self):
        """空输入返回空 int Series."""
        result = assign_groups(pd.Series(dtype=float), n_groups=10)
        assert len(result) == 0
        assert result.dtype == int
