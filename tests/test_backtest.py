"""TDD for backtest.py helper functions — #19."""
import numpy as np
import pandas as pd
import pytest
import duckdb
from conftest import requires_db
from src.backtest import mad_standardize, assign_groups, run_backtest, compute_group_returns


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

    def test_run_backtest_daily_freq(self, tmp_db):
        """日频回测产出更多记录（每个交易日一次调仓）."""
        stocks, trading_days = self._seed_tmp_db(tmp_db)

        result = run_backtest(
            start_date="20240101",
            end_date="20240228",
            factor_name="pb_factor",
            n_groups=2,
            standardize=False,
            db_path=tmp_db,
            freq="daily",
            silent=True,
        )

        gr = result["group_returns"]
        assert not gr.empty
        assert "long_short" in gr.columns
        # 日频应有更多交易日记录
        assert len(gr) > 0
        # factor_values 数量应接近交易日数
        assert len(result["factor_values"]) > 0

    def test_run_backtest_weekly_freq(self, tmp_db):
        """周频回测：每周一次调仓."""
        self._seed_tmp_db(tmp_db)

        result = run_backtest(
            start_date="20240101",
            end_date="20240228",
            factor_name="pb_factor",
            n_groups=2,
            standardize=False,
            db_path=tmp_db,
            freq="weekly",
            silent=True,
        )

        gr = result["group_returns"]
        assert not gr.empty
        assert "long_short" in gr.columns


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


# ============================================================
# compute_group_returns — 纯函数
# ============================================================
class TestComputeGroupReturns:
    def _make_trading_days(self, n_days: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
        """生成连续交易日序列，跳过周末."""
        dates = []
        current = pd.Timestamp(start)
        while len(dates) < n_days:
            if current.dayofweek < 5:  # Mon-Fri
                dates.append(current)
            current += pd.Timedelta(days=1)
        return pd.DatetimeIndex(dates)

    def _make_stocks(self, n: int) -> list[str]:
        return [f"{i:06d}.SZ" for i in range(n)]

    def test_all_equal_returns(self):
        """所有股票每天同涨 1% → 各组收益相等，多空 ≈ 0."""
        stocks = self._make_stocks(20)
        trading_days = self._make_trading_days(60)
        np.random.seed(1)

        # 因子值：随机但固定
        base_fv = pd.Series(np.random.randn(20), index=stocks)
        factor_values = [(td, base_fv + pd.Series(np.random.randn(20)*0.01, index=stocks))
                         for td in trading_days]

        # 等收益：所有股票每 1%
        daily_returns = {td: pd.Series(0.01, index=stocks) for td in trading_days}

        gr = compute_group_returns(factor_values, daily_returns, trading_days,
                                   freq="daily", n_groups=10)
        assert not gr.empty
        # 各组日收益应接近（因为因子值差异很小）
        group_cols = [c for c in gr.columns if c.startswith("group_")]
        for day in gr.index[:10]:
            row_vals = gr.loc[day, group_cols].values
            assert row_vals.max() - row_vals.min() < 1e-6  # essentially equal
        # 多空 ≈ 0
        assert abs(gr["long_short"].mean()) < 1e-4

    def test_top_group_outperforms_bottom(self):
        """高因子值股票收益高，低因子值股票收益低 → G1 > G10."""
        stocks = self._make_stocks(20)
        trading_days = self._make_trading_days(30)
        # 因子值递减：stock_0 最高, stock_19 最低
        fv_base = pd.Series(range(20, 0, -1), index=stocks, dtype=float)
        factor_values = [(td, fv_base) for td in trading_days]

        # 收益递减：stock_0 每天 1%, stock_19 每天 -1%
        rets = {td: pd.Series([0.01 - 0.002 * i for i in range(20)], index=stocks)
                for td in trading_days}

        gr = compute_group_returns(factor_values, daily_returns=rets,
                                   trading_days=trading_days, freq="daily", n_groups=10)
        # G1（最高因子）收益均值 > G10（最低因子）
        g1_mean = gr["group_1"].mean()
        g10_mean = gr["group_10"].mean()
        assert g1_mean > g10_mean
        assert gr["long_short"].mean() > 0

    def test_weekly_freq(self):
        """周频：每周一次调仓."""
        stocks = self._make_stocks(15)
        trading_days = self._make_trading_days(30)
        factor_values = [(td, pd.Series(range(15, 0, -1), index=stocks, dtype=float))
                         for td in trading_days]
        daily_returns = {td: pd.Series(0.01, index=stocks) for td in trading_days}
        gr = compute_group_returns(factor_values, daily_returns, trading_days,
                                   freq="weekly", n_groups=5)
        assert not gr.empty
        assert len(gr) > 0

    def test_monthly_freq(self):
        """月频：每月一次调仓."""
        stocks = self._make_stocks(15)
        trading_days = self._make_trading_days(60)
        factor_values = [(td, pd.Series(range(15, 0, -1), index=stocks, dtype=float))
                         for td in trading_days]
        daily_returns = {td: pd.Series(0.01, index=stocks) for td in trading_days}
        gr = compute_group_returns(factor_values, daily_returns, trading_days,
                                   freq="monthly", n_groups=5)
        assert not gr.empty
        assert len(gr) > 0

    def test_too_few_stocks_skips(self):
        """股票数 < n_groups × 2 时跳过当期不崩溃."""
        stocks = self._make_stocks(5)  # only 5 stocks, need 10 for n_groups=5
        trading_days = self._make_trading_days(20)
        factor_values = [(td, pd.Series(range(5, 0, -1), index=stocks, dtype=float))
                         for td in trading_days]
        daily_returns = {td: pd.Series(0.01, index=stocks) for td in trading_days}
        gr = compute_group_returns(factor_values, daily_returns, trading_days,
                                   freq="daily", n_groups=5)
        assert gr.empty

    def test_offset_delays_rebalance(self):
        """offset=1 的调仓日比 offset=0 晚一个交易日."""
        stocks = self._make_stocks(20)
        trading_days = self._make_trading_days(60)
        fv_base = pd.Series(range(20, 0, -1), index=stocks, dtype=float)
        factor_values = [(td, fv_base) for td in trading_days]
        daily_returns = {td: pd.Series(0.01, index=stocks) for td in trading_days}

        gr0 = compute_group_returns(factor_values, daily_returns, trading_days,
                                    freq="daily", n_groups=10, offset=0)
        gr1 = compute_group_returns(factor_values, daily_returns, trading_days,
                                    freq="daily", n_groups=10, offset=1)

        assert not gr0.empty
        assert not gr1.empty
        # offset=1 的第一条记录应该比 offset=0 晚（更晚开始建仓）
        assert gr1.index[0] > gr0.index[0]

    def test_market_cap_weighting_known(self):
        """市值加权：大市值股票权重更高，影响组收益."""
        stocks = ["A", "B", "C"]
        trading_days = self._make_trading_days(10)
        # 因子值相同 → 三只股票进同一组
        fv = pd.Series([1.0, 1.0, 1.0], index=stocks)
        factor_values = [(trading_days[0], fv)]
        # 收益：A=10%, B=0%, C=0%, 等权收益 = 3.33%
        daily_returns = {td: pd.Series({"A": 0.10, "B": 0.0, "C": 0.0}) for td in trading_days}
        # A 市值 900, B 市值 100, C 市值 0 (被排除) → A 权重=0.9
        market_caps = {trading_days[0]: pd.Series({"A": 900, "B": 100, "C": 0})}

        gr = compute_group_returns(factor_values, daily_returns, trading_days,
                                   freq="daily", n_groups=1, weighting="market_cap",
                                   market_caps=market_caps)
        assert not gr.empty
        # 市值加权：A=0.9, B=0.1 → 加权收益 = 0.9*0.10 + 0.1*0.0 = 0.09
        g1_val = gr["group_1"].iloc[0]
        assert pytest.approx(g1_val, abs=0.001) == 0.09

    def test_market_cap_weighting_falls_back_to_equal(self):
        """无 market_caps 数据时市值加权回退为等权."""
        stocks = self._make_stocks(20)
        trading_days = self._make_trading_days(30)
        factor_values = [(td, pd.Series(range(20, 0, -1), index=stocks, dtype=float))
                         for td in trading_days]
        daily_returns = {td: pd.Series(0.01, index=stocks) for td in trading_days}

        gr_mcap = compute_group_returns(factor_values, daily_returns, trading_days,
                                        freq="daily", n_groups=10, weighting="market_cap")
        gr_equal = compute_group_returns(factor_values, daily_returns, trading_days,
                                         freq="daily", n_groups=10, weighting="equal")
        pd.testing.assert_frame_equal(gr_mcap, gr_equal)

    def test_default_weighting_is_equal(self):
        """默认 weighting='equal'，向后兼容."""
        stocks = self._make_stocks(20)
        trading_days = self._make_trading_days(30)
        factor_values = [(td, pd.Series(range(20, 0, -1), index=stocks, dtype=float))
                         for td in trading_days]
        daily_returns = {td: pd.Series(0.01, index=stocks) for td in trading_days}

        gr = compute_group_returns(factor_values, daily_returns, trading_days,
                                   freq="daily", n_groups=10)
        assert not gr.empty
        # 不传 weighting 参数应该等于传 'equal'
        gr_explicit = compute_group_returns(factor_values, daily_returns, trading_days,
                                            freq="daily", n_groups=10, weighting="equal")
        pd.testing.assert_frame_equal(gr, gr_explicit)
