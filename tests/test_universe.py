"""TDD for universe.py — #10."""
import os
import tempfile
import duckdb
import pandas as pd
import pytest
from conftest import requires_db
from src.universe import filter_universe


def _make_temp_db() -> str:
    """Create a temp DuckDB path. Returns path; caller owns cleanup."""
    fd, path = tempfile.mkstemp(suffix=".duckdb", dir="tests/data")
    os.close(fd)
    os.remove(path)
    return path


def _trading_day_before(days: int) -> str:
    """Return a date string N days before a trade_date."""
    from datetime import datetime, timedelta
    dt = datetime(2023, 6, 30) - timedelta(days=days)
    return dt.strftime("%Y%m%d")


class TestFilterUniverse:
    def test_normal_stock_passes(self):
        """非 ST、上市满 365 天的股票通过过滤."""
        db_path = _make_temp_db()
        try:
            db = duckdb.connect(db_path)
            db.execute("CREATE TABLE stock_basic (ts_code TEXT PRIMARY KEY, name TEXT, list_date TEXT)")
            db.execute("CREATE TABLE stock_st (ts_code TEXT, trade_date TEXT, PRIMARY KEY(ts_code, trade_date))")
            db.execute("INSERT INTO stock_basic VALUES ('000001.SZ', '测试股', '20200101')")
            db.close()

            result = filter_universe("20230630", db_path=db_path)
            assert result == ["000001.SZ"]
        finally:
            os.remove(db_path)

    def test_listed_364_days_excluded(self):
        """上市 364 天的股票被排除."""
        db_path = _make_temp_db()
        try:
            db = duckdb.connect(db_path)
            db.execute("CREATE TABLE stock_basic (ts_code TEXT PRIMARY KEY, name TEXT, list_date TEXT)")
            db.execute("CREATE TABLE stock_st (ts_code TEXT, trade_date TEXT, PRIMARY KEY(ts_code, trade_date))")
            db.execute("INSERT INTO stock_basic VALUES ('000001.SZ', '新股', '20220701')")  # 2022-07-01 → 2023-06-30 = 364 days
            db.close()

            result = filter_universe("20230630", db_path=db_path)
            assert result == []
        finally:
            os.remove(db_path)

    def test_listed_365_days_included(self):
        """上市正好 365 天的股票通过."""
        db_path = _make_temp_db()
        try:
            db = duckdb.connect(db_path)
            db.execute("CREATE TABLE stock_basic (ts_code TEXT PRIMARY KEY, name TEXT, list_date TEXT)")
            db.execute("CREATE TABLE stock_st (ts_code TEXT, trade_date TEXT, PRIMARY KEY(ts_code, trade_date))")
            db.execute("INSERT INTO stock_basic VALUES ('000001.SZ', '刚满一年', '20220630')")  # 2022-06-30 → 2023-06-30 = 365 days
            db.close()

            result = filter_universe("20230630", db_path=db_path)
            assert result == ["000001.SZ"]
        finally:
            os.remove(db_path)

    def test_listed_366_days_included(self):
        """上市 366 天的股票通过."""
        db_path = _make_temp_db()
        try:
            db = duckdb.connect(db_path)
            db.execute("CREATE TABLE stock_basic (ts_code TEXT PRIMARY KEY, name TEXT, list_date TEXT)")
            db.execute("CREATE TABLE stock_st (ts_code TEXT, trade_date TEXT, PRIMARY KEY(ts_code, trade_date))")
            db.execute("INSERT INTO stock_basic VALUES ('000001.SZ', '老股', '20220629')")  # 2022-06-29 → 2023-06-30 = 366 days
            db.close()

            result = filter_universe("20230630", db_path=db_path)
            assert result == ["000001.SZ"]
        finally:
            os.remove(db_path)

    def test_st_stock_excluded(self):
        """当日为 ST 的股票被排除."""
        db_path = _make_temp_db()
        try:
            db = duckdb.connect(db_path)
            db.execute("CREATE TABLE stock_basic (ts_code TEXT PRIMARY KEY, name TEXT, list_date TEXT)")
            db.execute("CREATE TABLE stock_st (ts_code TEXT, trade_date TEXT, PRIMARY KEY(ts_code, trade_date))")
            db.execute("INSERT INTO stock_basic VALUES ('000001.SZ', 'ST股', '20200101')")
            db.execute("INSERT INTO stock_st VALUES ('000001.SZ', '20230630')")
            db.close()

            result = filter_universe("20230630", db_path=db_path)
            assert result == []
        finally:
            os.remove(db_path)

    def test_st_on_other_date_not_excluded(self):
        """ST 在其他日期才生效，当前调仓日不过滤."""
        db_path = _make_temp_db()
        try:
            db = duckdb.connect(db_path)
            db.execute("CREATE TABLE stock_basic (ts_code TEXT PRIMARY KEY, name TEXT, list_date TEXT)")
            db.execute("CREATE TABLE stock_st (ts_code TEXT, trade_date TEXT, PRIMARY KEY(ts_code, trade_date))")
            db.execute("INSERT INTO stock_basic VALUES ('000001.SZ', '曾ST', '20200101')")
            db.execute("INSERT INTO stock_st VALUES ('000001.SZ', '20230501')")  # ST on different date
            db.close()

            result = filter_universe("20230630", db_path=db_path)
            assert result == ["000001.SZ"]
        finally:
            os.remove(db_path)

    def test_all_filtered_returns_empty(self):
        """全部被过滤时返回空列表."""
        db_path = _make_temp_db()
        try:
            db = duckdb.connect(db_path)
            db.execute("CREATE TABLE stock_basic (ts_code TEXT PRIMARY KEY, name TEXT, list_date TEXT)")
            db.execute("CREATE TABLE stock_st (ts_code TEXT, trade_date TEXT, PRIMARY KEY(ts_code, trade_date))")
            db.execute("INSERT INTO stock_basic VALUES ('000001.SZ', '新股', '20230701')")  # not listed yet
            db.close()

            result = filter_universe("20230630", db_path=db_path)
            assert result == []
        finally:
            os.remove(db_path)


@requires_db
class TestSmoke:
    def test_real_db_returns_nonempty(self):
        """真实库冒烟：不抛异常且返回非空列表."""
        result = filter_universe("20231228")
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], str)
        assert all(r.endswith((".SZ", ".SH")) for r in result[:100])
