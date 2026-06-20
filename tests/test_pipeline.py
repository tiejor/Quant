"""TDD for pipeline.py — #11."""
import duckdb
import pytest
from conftest import requires_tushare
from src.pipeline import (
    _is_rate_limit_error,
    _get_trading_days_in_range,
    init_tables,
    get_max_date,
    _fetch_with_retry,
    pull_stock_basic,
    pull_trade_cal,
    pull_daily,
    pull_daily_basic,
    _find_missing_dates,
    fill_missing_dates,
)


class TestIsRateLimitError:
    def test_frequency_keyword(self):
        assert _is_rate_limit_error(Exception("frequency limit exceeded"))
        assert _is_rate_limit_error(Exception("接口访问频率限制"))

    def test_limit_keyword(self):
        assert _is_rate_limit_error(Exception("rate limit hit"))
        assert _is_rate_limit_error(Exception("请求限流"))

    def test_429_keyword(self):
        assert _is_rate_limit_error(Exception("HTTP 429 too many requests"))

    def test_too_many_keyword(self):
        assert _is_rate_limit_error(Exception("too many connections"))

    def test_non_rate_limit_error(self):
        assert not _is_rate_limit_error(Exception("connection timeout"))
        assert not _is_rate_limit_error(Exception("invalid parameter"))
        assert not _is_rate_limit_error(ValueError("something went wrong"))


class TestFetchWithRetry:
    def test_success_first_try_returns_result(self):
        result = _fetch_with_retry(lambda: 42, "test")
        assert result == 42

    def test_rate_limit_retries_then_succeeds(self):
        """频率限制后重试成功."""
        counter = {"calls": 0}

        def flaky():
            counter["calls"] += 1
            if counter["calls"] < 3:
                raise Exception("frequency limit exceeded")
            return "ok"

        result = _fetch_with_retry(flaky, "test", max_retries=5, sleep_sec=0)
        assert result == "ok"
        assert counter["calls"] == 3

    def test_rate_limit_exhausted_returns_none(self):
        """超过最大重试次数返回 None."""
        def always_limit():
            raise Exception("API frequency limit")

        result = _fetch_with_retry(always_limit, "test", max_retries=2, sleep_sec=0)
        assert result is None

    def test_non_rate_limit_error_raises_immediately(self):
        """非频率限制错误直接抛出，不重试."""
        counter = {"calls": 0}

        def bad_arg():
            counter["calls"] += 1
            raise ValueError("invalid argument")

        with pytest.raises(ValueError, match="invalid argument"):
            _fetch_with_retry(bad_arg, "test", max_retries=5, sleep_sec=0)
        assert counter["calls"] == 1  # only called once, no retry


class TestInitTables:
    def test_all_tables_created(self, tmp_db):
        """5 张表创建成功."""
        init_tables(db_path=tmp_db)
        db = duckdb.connect(tmp_db)
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        db.close()
        names = [t[0] for t in tables]
        assert "stock_basic" in names
        assert "trade_cal" in names
        assert "daily" in names
        assert "daily_basic" in names
        assert "stock_st" in names


class TestGetMaxDate:
    def test_empty_table_returns_none(self, tmp_db):
        init_tables(db_path=tmp_db)
        result = get_max_date("daily", db_path=tmp_db)
        assert result is None

    def test_with_data_returns_max(self, tmp_db):
        init_tables(db_path=tmp_db)
        db = duckdb.connect(tmp_db)
        db.execute("INSERT INTO daily (ts_code, trade_date) VALUES ('A', '20230101'), ('A', '20230102')")
        db.close()
        result = get_max_date("daily", db_path=tmp_db)
        assert result == "20230102"


class TestGetTradingDaysInRange:
    def test_returns_open_days_only(self, tmp_db):
        """只返回 is_open=1 的交易日."""
        init_tables(db_path=tmp_db)
        db = duckdb.connect(tmp_db)
        db.execute("""
            INSERT INTO trade_cal (cal_date, is_open)
            VALUES ('20230101', 0), ('20230102', 0), ('20230103', 1),
                   ('20230104', 1), ('20230105', 1), ('20230106', 0)
        """)
        db.close()
        result = _get_trading_days_in_range("20230101", "20230106", db_path=tmp_db)
        assert result == ["20230103", "20230104", "20230105"]

    def test_empty_range_returns_empty(self, tmp_db):
        """无交易日区间返回空列表."""
        init_tables(db_path=tmp_db)
        db = duckdb.connect(tmp_db)
        db.execute("INSERT INTO trade_cal (cal_date, is_open) VALUES ('20230201', 0)")
        db.close()
        result = _get_trading_days_in_range("20230101", "20230131", db_path=tmp_db)
        assert result == []


@requires_tushare
class TestPullDailySmoke:
    def test_writes_month_to_db(self, tmp_db):
        """拉取 1 个月日线数据写入 daily 表."""
        init_tables(db_path=tmp_db)
        db = duckdb.connect(tmp_db)
        db.execute("""
            INSERT INTO trade_cal (cal_date, is_open)
            VALUES ('20250102', 1), ('20250103', 1), ('20250106', 1),
                   ('20250107', 1), ('20250108', 1), ('20250109', 1),
                   ('20250110', 1), ('20250113', 1)
        """)
        db.close()

        pull_daily("20250101", "20250131", db_path=tmp_db)

        db = duckdb.connect(tmp_db, read_only=True)
        count = db.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
        db.close()
        assert count > 0


    def test_daily_basic_writes_month_to_db(self, tmp_db):
        """拉取 1 个月基本面数据写入 daily_basic 表."""
        init_tables(db_path=tmp_db)
        db = duckdb.connect(tmp_db)
        db.execute("""
            INSERT INTO trade_cal (cal_date, is_open)
            VALUES ('20250102', 1), ('20250103', 1), ('20250106', 1),
                   ('20250107', 1), ('20250108', 1), ('20250109', 1),
                   ('20250110', 1), ('20250113', 1)
        """)
        db.close()

        pull_daily_basic("20250101", "20250131", db_path=tmp_db)

        db = duckdb.connect(tmp_db, read_only=True)
        count = db.execute("SELECT COUNT(*) FROM daily_basic").fetchone()[0]
        db.close()
        assert count > 0


@requires_tushare
class TestSmoke:
    def test_pull_stock_basic_to_temp_db(self, tmp_db):
        """真实网络拉取 stock_basic 写入临时 DB."""
        init_tables(db_path=tmp_db)
        pull_stock_basic(db_path=tmp_db)

        db = duckdb.connect(tmp_db, read_only=True)
        count = db.execute("SELECT COUNT(*) FROM stock_basic").fetchone()[0]
        db.close()
        assert count > 100  # A 股至少几百只股票

    def test_pull_trade_cal_to_temp_db(self, tmp_db):
        """真实网络拉取 trade_cal（1 个月）写入临时 DB."""
        init_tables(db_path=tmp_db)
        pull_trade_cal("20230101", "20230131", db_path=tmp_db)

        db = duckdb.connect(tmp_db, read_only=True)
        count = db.execute("SELECT COUNT(*) FROM trade_cal").fetchone()[0]
        db.close()
        assert count >= 10  # 1 个月至少 10 个交易日


# ============================================================
# fill_missing_dates
# ============================================================
class TestFindMissingDates:
    def test_all_present_returns_empty(self, tmp_db):
        """所有交易日都有数据 → 返回空列表."""
        init_tables(db_path=tmp_db)
        db = duckdb.connect(tmp_db)
        db.execute("INSERT INTO trade_cal VALUES ('20240102', 1, 'SSE'), ('20240103', 1, 'SSE')")
        db.execute("INSERT INTO daily (ts_code, trade_date, open, high, low, close, pre_close, pct_chg, vol, amount) VALUES ('000001.SZ', '20240102', 10,11,9,10,10,1.0,10000,100000)")
        db.execute("INSERT INTO daily (ts_code, trade_date, open, high, low, close, pre_close, pct_chg, vol, amount) VALUES ('000001.SZ', '20240103', 10,11,9,10,10,1.0,10000,100000)")
        db.close()

        missing = _find_missing_dates("daily", "20240101", "20240105", db_path=tmp_db)
        assert missing == []

    def test_partial_missing(self, tmp_db):
        """部分交易日缺失 → 返回缺失日期列表."""
        init_tables(db_path=tmp_db)
        db = duckdb.connect(tmp_db)
        db.execute("INSERT INTO trade_cal VALUES ('20240102', 1, 'SSE'), ('20240103', 1, 'SSE'), ('20240104', 1, 'SSE')")
        db.execute("INSERT INTO daily (ts_code, trade_date, open, high, low, close, pre_close, pct_chg, vol, amount) VALUES ('000001.SZ', '20240102', 10,11,9,10,10,1.0,10000,100000)")
        # 20240103 and 20240104 missing from daily
        db.close()

        missing = _find_missing_dates("daily", "20240101", "20240105", db_path=tmp_db)
        assert set(missing) == {"20240103", "20240104"}

    def test_non_trading_days_excluded(self, tmp_db):
        """非交易日不参与比较."""
        init_tables(db_path=tmp_db)
        db = duckdb.connect(tmp_db)
        # Only 20240102 is a trading day
        db.execute("INSERT INTO trade_cal VALUES ('20240102', 1, 'SSE'), ('20240105', 0, 'SSE')")
        db.close()

        missing = _find_missing_dates("daily", "20240101", "20240105", db_path=tmp_db)
        assert missing == ["20240102"]  # only trading day, no data → missing

    def test_empty_trade_cal(self, tmp_db):
        """无交易日 → 返回空列表."""
        init_tables(db_path=tmp_db)
        missing = _find_missing_dates("daily", "20240101", "20240105", db_path=tmp_db)
        assert missing == []


class TestFillMissingDates:
    def test_no_missing_data_no_fetch(self, tmp_db):
        """数据完整 → 无需下载，返回空列表."""
        init_tables(db_path=tmp_db)
        db = duckdb.connect(tmp_db)
        db.execute("INSERT INTO trade_cal VALUES ('20240102', 1, 'SSE'), ('20240103', 1, 'SSE')")
        db.execute("INSERT INTO daily (ts_code, trade_date, open, high, low, close, pre_close, pct_chg, vol, amount) VALUES ('000001.SZ', '20240102', 10,11,9,10,10,1.0,10000,100000)")
        db.execute("INSERT INTO daily (ts_code, trade_date, open, high, low, close, pre_close, pct_chg, vol, amount) VALUES ('000001.SZ', '20240103', 10,11,9,10,10,1.0,10000,100000)")
        db.execute("INSERT INTO daily_basic (ts_code, trade_date, pb, total_mv, circ_mv, turnover_rate) VALUES ('000001.SZ', '20240102', 2.0, 1e6, 5e5, 5.0)")
        db.execute("INSERT INTO daily_basic (ts_code, trade_date, pb, total_mv, circ_mv, turnover_rate) VALUES ('000001.SZ', '20240103', 2.0, 1e6, 5e5, 5.0)")
        db.close()

        result = fill_missing_dates("20240101", "20240105", db_path=tmp_db)
        assert result["daily"] == []
        assert result["daily_basic"] == []

    def test_returns_dict_with_both_tables(self, tmp_db):
        """返回字典包含 daily 和 daily_basic 两个 key."""
        init_tables(db_path=tmp_db)
        db = duckdb.connect(tmp_db)
        db.execute("INSERT INTO trade_cal VALUES ('20240102', 1, 'SSE')")
        db.execute("INSERT INTO daily (ts_code, trade_date, open, high, low, close, pre_close, pct_chg, vol, amount) VALUES ('000001.SZ', '20240102', 10,11,9,10,10,1.0,10000,100000)")
        db.execute("INSERT INTO daily_basic (ts_code, trade_date, pb, total_mv, circ_mv, turnover_rate) VALUES ('000001.SZ', '20240102', 2.0, 1e6, 5e5, 5.0)")
        db.close()

        result = fill_missing_dates("20240101", "20240105", db_path=tmp_db)
        assert "daily" in result
        assert "daily_basic" in result
        assert isinstance(result["daily"], list)
        assert isinstance(result["daily_basic"], list)
