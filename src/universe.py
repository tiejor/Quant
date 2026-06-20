"""
股票池模块。
根据调仓日过滤有效股票：排除 ST、排除上市不足 365 天的新股。
"""

import pandas as pd
import duckdb

DB_PATH = "data/quant.duckdb"


def _get_db(db_path: str | None = None):
    """获取 DuckDB 只读连接"""
    return duckdb.connect(db_path or DB_PATH, read_only=True)


def filter_universe(trade_date: str, db_path: str | None = None, index_code: str | None = None) -> list[str]:
    """
    返回指定调仓日有资格进入分组的股票代码列表。
    过滤规则：
      1. 排除当日为 ST 的股票
      2. 排除上市不足 365 天的新股（list_date 早于 trade_date 至少 365 天）
    停牌不做剔除。
    index_code: 预留参数，指定指数成分股过滤（None = 全市场）。
    """
    db = _get_db(db_path)

    # 当日 ST 列表
    st_codes = set(
        db.execute(
            "SELECT DISTINCT ts_code FROM stock_st WHERE trade_date = ?",
            [trade_date]
        ).df()["ts_code"].tolist()
    )

    # 全量基础信息（含上市日期）
    all_stocks = db.execute(
        "SELECT ts_code, list_date FROM stock_basic"
    ).df()
    db.close()

    # 过滤上市不足 365 天
    trade_dt = pd.Timestamp(trade_date)
    all_stocks["list_dt"] = pd.to_datetime(all_stocks["list_date"], format="%Y%m%d")
    all_stocks = all_stocks[all_stocks["list_dt"].notna()]
    all_stocks["days_listed"] = (trade_dt - all_stocks["list_dt"]).dt.days
    valid = all_stocks[all_stocks["days_listed"] >= 365]

    # 排除 ST
    result = [c for c in valid["ts_code"].tolist() if c not in st_codes]

    if len(result) == 0:
        print(f"  [universe] {trade_date} 有效股票数为 0，请检查数据")
    return result
