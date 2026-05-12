"""
数据管线：Tushare → DuckDB
负责从 Tushare 拉取 A 股数据并存储到本地 DuckDB。
大数据表用 joblib 并行下载，带频率限制检测与重试。
"""

import os
import time
import duckdb
import pandas as pd
import tushare as ts
from joblib import Parallel, delayed

# ============================================================
# 配置
# ============================================================
DB_PATH = "data/quant.duckdb"
MAX_RETRIES = 4      # 频率限制后最大重试次数
RETRY_SLEEP = 15     # 命中频率限制后的休眠秒数

# Tushare 懒加载（实际调用时才检查 token）
_pro = None

def _get_pro():
    """获取 Tushare pro 接口（懒加载 token）"""
    global _pro
    if _pro is None:
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            raise RuntimeError("未找到 TUSHARE_TOKEN 环境变量，请先设置: export TUSHARE_TOKEN=your_token")
        _pro = ts.pro_api(token)
    return _pro


# ============================================================
# DuckDB 连接
# ============================================================
def get_db():
    """获取 DuckDB 连接"""
    return duckdb.connect(DB_PATH)


# ============================================================
# 建表
# ============================================================
def init_tables():
    """初始化所有数据表（不存在则创建）"""
    db = get_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS stock_basic (
            ts_code     TEXT PRIMARY KEY,
            name        TEXT,
            industry    TEXT,
            list_date   TEXT,
            delist_date TEXT
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS trade_cal (
            cal_date TEXT PRIMARY KEY,
            is_open  INTEGER,
            exchange TEXT
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS daily (
            ts_code    TEXT,
            trade_date TEXT,
            open       DOUBLE,
            high       DOUBLE,
            low        DOUBLE,
            close      DOUBLE,
            pre_close  DOUBLE,
            pct_chg    DOUBLE,
            vol        DOUBLE,
            amount     DOUBLE,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS daily_basic (
            ts_code       TEXT,
            trade_date    TEXT,
            pb            DOUBLE,
            total_mv      DOUBLE,
            circ_mv       DOUBLE,
            turnover_rate DOUBLE,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS stock_st (
            ts_code    TEXT,
            trade_date TEXT,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)
    db.close()


# ============================================================
# 重试包装器
# ============================================================
def _is_rate_limit_error(ex: Exception) -> bool:
    """判断是否为频率限制类错误"""
    msg = str(ex).lower()
    keywords = ["frequency", "limit", "频率", "限流", "429", "too many"]
    return any(k in msg for k in keywords)


def _fetch_with_retry(fetch_fn, label: str, max_retries=MAX_RETRIES, sleep_sec=RETRY_SLEEP):
    """
    带重试的数据拉取包装器。
    频率限制错误 → sleep 后重试，最多 max_retries 次。
    其他错误 → 直接抛出。
    全失败 → 返回 None。
    """
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_fn()
        except Exception as e:
            if _is_rate_limit_error(e):
                if attempt < max_retries:
                    print(f"  [{label}] 频率限制，休眠 {sleep_sec}s 后重试 ({attempt}/{max_retries})...")
                    time.sleep(sleep_sec)
                else:
                    print(f"  [{label}] 重试 {max_retries} 次仍失败，跳过此任务")
                    return None
            else:
                print(f"  [{label}] 非频率类错误: {e}")
                raise
    return None


# ============================================================
# 串行拉取：小数据表
# ============================================================
def pull_stock_basic():
    """拉取 A 股基础信息（全量，一次调用）"""
    print("[stock_basic] 拉取中...")

    def _do():
        df = _get_pro().stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,name,industry,list_date,delist_date"
        )
        if df is None or len(df) == 0:
            raise RuntimeError("返回空数据")
        return df

    df = _fetch_with_retry(_do, "stock_basic")
    if df is None:
        return

    db = get_db()
    db.execute("DELETE FROM stock_basic")  # 基础信息全量覆盖
    db.execute("INSERT INTO stock_basic SELECT * FROM df")
    db.close()
    print(f"[stock_basic] 完成，{len(df)} 条")


def pull_trade_cal(start_date, end_date):
    """拉取交易日历"""
    print(f"[trade_cal] {start_date}~{end_date}...")

    def _do():
        df = _get_pro().trade_cal(
            exchange="SSE",
            start_date=start_date,
            end_date=end_date,
            fields="cal_date,is_open"
        )
        if df is None or len(df) == 0:
            return None
        df["exchange"] = "SSE"
        return df

    df = _fetch_with_retry(_do, "trade_cal")
    if df is None:
        return

    db = get_db()
    db.execute("INSERT OR IGNORE INTO trade_cal SELECT * FROM df")
    db.close()
    print(f"[trade_cal] 完成，{len(df)} 条")


def pull_stock_st(start_date, end_date):
    """拉取 ST 股票列表（按年切片，串行）"""
    print(f"[stock_st] {start_date}~{end_date}...")
    start_yr = int(start_date[:4])
    end_yr = int(end_date[:4])
    all_dfs = []

    # ST 表较小，串行即可
    for y in range(start_yr, end_yr + 1):
        s = f"{y}0101"
        e = f"{y}1231"
        label = f"stock_st/{y}"

        def _do(s=s, e=e):
            df = _get_pro().stock_st(trade_date=s, end_date=e)
            if df is not None and len(df) > 0:
                return df[["ts_code", "trade_date"]].drop_duplicates()
            return None

        df = _fetch_with_retry(_do, label)
        if df is not None:
            all_dfs.append(df)
        time.sleep(0.3)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        db = get_db()
        db.execute("INSERT OR IGNORE INTO stock_st SELECT * FROM result")
        db.close()
        print(f"[stock_st] 完成，{len(result)} 条")
    else:
        print("[stock_st] 无数据")


# ============================================================
# 并行拉取：大数据表
# ============================================================
def _fetch_daily_year(year: int) -> pd.DataFrame | None:
    """拉取单年的日线行情（供 joblib 并行调用）"""
    s = f"{year}0101"
    e = f"{year}1231"
    label = f"daily/{year}"

    def _do():
        df = _get_pro().daily(
            start_date=s,
            end_date=e,
            fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount"
        )
        return df

    result = _fetch_with_retry(_do, label)
    if result is not None and len(result) > 0:
        print(f"  [daily/{year}] {len(result)} 条")
    return result


def _fetch_daily_basic_year(year: int) -> pd.DataFrame | None:
    """拉取单年的每日基本面指标（供 joblib 并行调用）"""
    s = f"{year}0101"
    e = f"{year}1231"
    label = f"daily_basic/{year}"

    def _do():
        df = _get_pro().daily_basic(
            start_date=s,
            end_date=e,
            fields="ts_code,trade_date,pb,total_mv,circ_mv,turnover_rate"
        )
        return df

    result = _fetch_with_retry(_do, label)
    if result is not None and len(result) > 0:
        print(f"  [daily_basic/{year}] {len(result)} 条")
    return result


def pull_daily(start_date, end_date):
    """拉取日线行情：按年拆任务，joblib 并行下载"""
    start_yr = int(start_date[:4])
    end_yr = int(end_date[:4])
    years = list(range(start_yr, end_yr + 1))
    print(f"[daily] 并行拉取 {len(years)} 年: {start_yr}~{end_yr} (n_jobs=-2)...")

    results = Parallel(n_jobs=-2, prefer="threads")(
        delayed(_fetch_daily_year)(y) for y in years
    )

    # 合并写入 DuckDB
    valid = [r for r in results if r is not None and len(r) > 0]
    if valid:
        merged = pd.concat(valid, ignore_index=True)
        db = get_db()
        db.execute("INSERT OR IGNORE INTO daily SELECT * FROM merged")
        db.close()
        print(f"[daily] 完成，共 {len(merged)} 条")
    else:
        print("[daily] 无数据")


def pull_daily_basic(start_date, end_date):
    """拉取每日基本面指标：按年拆任务，joblib 并行下载"""
    start_yr = int(start_date[:4])
    end_yr = int(end_date[:4])
    years = list(range(start_yr, end_yr + 1))
    print(f"[daily_basic] 并行拉取 {len(years)} 年: {start_yr}~{end_yr} (n_jobs=-2)...")

    results = Parallel(n_jobs=-2, prefer="threads")(
        delayed(_fetch_daily_basic_year)(y) for y in years
    )

    valid = [r for r in results if r is not None and len(r) > 0]
    if valid:
        merged = pd.concat(valid, ignore_index=True)
        db = get_db()
        db.execute("INSERT OR IGNORE INTO daily_basic SELECT * FROM merged")
        db.close()
        print(f"[daily_basic] 完成，共 {len(merged)} 条")
    else:
        print("[daily_basic] 无数据")


# ============================================================
# 增量逻辑
# ============================================================
def get_max_date(table):
    """获取表中最大交易日期，无数据返回 None"""
    db = get_db()
    result = db.execute(f"SELECT MAX(trade_date) FROM {table}").fetchone()
    db.close()
    return result[0] if result else None


def incremental_update(end_date=None):
    """增量更新：从各表最大日期 +1 天开始拉取"""
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y%m%d")

    # 小表直接串行增量
    last = get_max_date("trade_cal")
    if last:
        start = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        if start < end_date:
            pull_trade_cal(start, end_date)

    # 大表按年并行增量
    for label, pull_fn in [("daily", pull_daily), ("daily_basic", pull_daily_basic)]:
        last = get_max_date(label)
        if last is None:
            print(f"[{label}] 无存量数据，跳过增量")
            continue
        start = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        if start >= end_date:
            print(f"[{label}] 已是最新 ({last})")
            continue
        print(f"[{label}] 增量: {start} ~ {end_date}")
        pull_fn(start, end_date)

    # ST 小表串行增量
    last = get_max_date("stock_st")
    if last:
        start = (pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        if start < end_date:
            pull_stock_st(start, end_date)

    # 基础信息全量覆盖
    pull_stock_basic()
    print("增量更新完成")


# ============================================================
# 全量拉取
# ============================================================
def full_load(start_date, end_date):
    """全量拉取所有数据"""
    print(f"全量拉取: {start_date} ~ {end_date}")
    init_tables()
    pull_stock_basic()
    pull_trade_cal(start_date, end_date)
    pull_daily(start_date, end_date)
    pull_daily_basic(start_date, end_date)
    pull_stock_st(start_date, end_date)
    print("全量拉取完成")


# ============================================================
# 主入口
# ============================================================
def run(start_date=None, end_date=None):
    """
    主入口：自动判断全量还是增量。
    - 若 daily 表为空 → 全量拉取（start_date 必传）
    - 若 daily 表有数据 → 增量更新
    """
    if end_date is None:
        end_date = pd.Timestamp.now().strftime("%Y%m%d")

    last = get_max_date("daily")
    if last is None:
        if start_date is None:
            raise ValueError("首次运行需指定 start_date，例如: run('20200101')")
        full_load(start_date, end_date)
    else:
        print(f"daily 表最新日期: {last}，走增量更新")
        incremental_update(end_date)


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        run(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:
        run()
    else:
        print("用法: python src/pipeline.py [start_date] [end_date]")
        print("  无参数 → 增量更新")
        print("  start_date end_date → 全量拉取")
