"""
截面回测引擎 — 月频调仓，分组等权，日度追踪，多空合成。
"""

import pandas as pd
import numpy as np
import duckdb
from src.universe import filter_universe
from src.factors import get_factor

DB_PATH = "data/quant.duckdb"


def _get_db():
    """获取 DuckDB 只读连接"""
    return duckdb.connect(DB_PATH, read_only=True)


# ============================================================
# 交易日工具
# ============================================================
def _get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """获取回测区间内的交易日序列"""
    db = _get_db()
    df = db.execute(
        "SELECT cal_date FROM trade_cal WHERE is_open=1 AND cal_date BETWEEN ? AND ? ORDER BY cal_date",
        [start_date, end_date]
    ).df()
    db.close()
    return pd.DatetimeIndex(df["cal_date"])


def _last_trading_day_of_month(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    从交易日序列中挑出每月的最后一个交易日，作为因子计算日。
    """
    df = pd.DataFrame({"date": trading_days})
    df["year_month"] = df["date"].dt.to_period("M")
    return pd.DatetimeIndex(df.groupby("year_month")["date"].max())


def _next_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """返回 date 之后的下一个交易日，无则返回 None"""
    candidates = trading_days[trading_days > date]
    if len(candidates) == 0:
        return None
    return candidates[0]


# ============================================================
# 因子预处理
# ============================================================
def mad_standardize(series: pd.Series) -> pd.Series:
    """
    截面 MAD（中位数绝对离差）标准化。
    MAD = median(|x - median(x)|)
    标准化值 = (x - median(x)) / MAD
    """
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0:
        return series  # 所有值相同或 MAD 为 0，不标准化
    return (series - med) / mad


# ============================================================
# 分组
# ============================================================
def assign_groups(factor_values: pd.Series, n_groups: int) -> pd.Series:
    """
    按因子值分位数将股票分成 n_groups 组。
    返回 Series，index=ts_code，值=组号（1 最高因子值，n_groups 最低）。
    """
    # 去除 NaN
    valid = factor_values.dropna()
    if len(valid) == 0:
        return pd.Series(dtype=int)

    # qcut：按分位数等分
    labels = list(range(1, n_groups + 1))  # group 1 = highest factor
    try:
        groups = pd.qcut(valid, q=n_groups, labels=labels)
    except ValueError:
        # qcut 失败（如分位数重复），用 rank 分组
        groups = pd.Series(
            pd.cut(valid.rank(method="first"), bins=n_groups, labels=labels).values,
            index=valid.index
        )
    return groups.sort_values(ascending=False)  # 高因子值在前


# ============================================================
# 数据加载
# ============================================================
def _load_daily_returns(ts_codes: list[str], date: str) -> pd.Series:
    """
    加载指定日期的股票日收益率。
    返回 Series，index=ts_code，值=日收益率（小数）
    """
    if len(ts_codes) == 0:
        return pd.Series(dtype=float)
    db = _get_db()
    placeholders = ",".join(["?"] * len(ts_codes))
    df = db.execute(
        f"SELECT ts_code, pct_chg FROM daily WHERE trade_date=? AND ts_code IN ({placeholders})",
        [date] + ts_codes
    ).df()
    db.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["return"] = df["pct_chg"] / 100.0  # 百分比 → 小数
    return df.set_index("ts_code")["return"]


def _load_factor_data(date: str, ts_codes: list[str]) -> pd.DataFrame:
    """
    加载指定日期的截面数据，用于计算因子。
    返回 DataFrame，index=ts_code，含 pb, total_mv, circ_mv, turnover_rate 列。
    """
    if len(ts_codes) == 0:
        return pd.DataFrame()
    db = _get_db()
    placeholders = ",".join(["?"] * len(ts_codes))
    df = db.execute(
        f"SELECT ts_code, pb, total_mv, circ_mv, turnover_rate FROM daily_basic "
        f"WHERE trade_date=? AND ts_code IN ({placeholders})",
        [date] + ts_codes
    ).df()
    db.close()
    return df.set_index("ts_code") if not df.empty else pd.DataFrame()


# ============================================================
# 主回测函数
# ============================================================
def run_backtest(
    start_date: str,
    end_date: str,
    factor_name: str,
    n_groups: int = 10,
    standardize: bool = True,
) -> dict:
    """
    月频截面回测。

    参数：
        start_date:  回测起始日 YYYYMMDD
        end_date:    回测结束日 YYYYMMDD
        factor_name: 因子名称（factors.py FACTOR_REGISTRY 中的 key）
        n_groups:    分组数（默认 10）
        standardize: 是否做截面 MAD 标准化（默认 True）

    返回字典：
        group_returns:  DataFrame，每列为一组 + long_short 的日度收益率
        group_nav:      DataFrame，各组 + 多空累计净值
    """
    factor_fn = get_factor(factor_name)

    # 获取交易日序列
    all_days = _get_trading_days(start_date, end_date)
    if len(all_days) == 0:
        raise ValueError(f"区间 {start_date}~{end_date} 无交易日")

    # 每月最后一个交易日 = 因子计算日
    factor_dates = _last_trading_day_of_month(all_days)

    print(f"[backtest] 回测区间: {all_days[0].strftime('%Y%m%d')} ~ {all_days[-1].strftime('%Y%m%d')}")
    print(f"[backtest] 交易天数: {len(all_days)}，月数: {len(factor_dates)}")
    print(f"[backtest] 因子: {factor_name}，分组数: {n_groups}，标准化: {standardize}")

    # 存储各组 + 多空每日收益率
    daily_records = []  # [{trade_date, group_1, ..., group_N, long_short}]

    # 逐月回测
    for i, fc_date in enumerate(factor_dates):
        fc_str = fc_date.strftime("%Y%m%d")

        # 调仓日 = 因子计算日之后的下一个交易日
        rebalance_date = _next_trading_day(fc_date, all_days)
        if rebalance_date is None:
            continue

        # 持仓期结束日 = 下一个调仓日 - 1（即下个因子计算日之后的调仓日前一天）
        # 简化：持仓到当前月的最后一个交易日前
        # 实际 = 本月调仓日 → 下月调仓日前一天
        if i + 1 < len(factor_dates):
            next_fc = factor_dates[i + 1]
            next_rebalance = _next_trading_day(next_fc, all_days)
            if next_rebalance is None:
                hold_end = all_days[-1]
            else:
                # 找到 next_rebalance 在 all_days 中的位置，往前一天
                idx = all_days.get_loc(next_rebalance)
                hold_end = all_days[idx - 1] if idx > 0 else next_rebalance
        else:
            hold_end = all_days[-1]

        # 股票池（调仓日更新）
        universe = filter_universe(fc_str)  # 因子计算日用 stock_st 状态
        if len(universe) < n_groups * 2:
            print(f"  [{fc_str}] 股票池太小 ({len(universe)} 只)，跳过此月")
            continue

        # 加载因子计算日的截面数据
        fac_data = _load_factor_data(fc_str, universe)
        if fac_data.empty:
            print(f"  [{fc_str}] 无因子数据，跳过")
            continue

        # 计算因子值
        factor_values = factor_fn(fac_data)
        factor_values = factor_values.dropna()
        if len(factor_values) < n_groups * 2:
            print(f"  [{fc_str}] 有效因子值太少 ({len(factor_values)})，跳过")
            continue

        # MAD 标准化
        if standardize:
            factor_values = mad_standardize(factor_values)

        # 分组
        groups = assign_groups(factor_values, n_groups)

        # 持仓期内逐日追踪
        hold_days = all_days[(all_days >= rebalance_date) & (all_days <= hold_end)]
        for day in hold_days:
            day_str = day.strftime("%Y%m%d")
            # 加载该日所有分组内股票的收益率
            rets = _load_daily_returns(list(groups.index), day_str)
            if rets.empty:
                continue

            record = {"trade_date": day_str}
            for grp in range(1, n_groups + 1):
                grp_stocks = groups[groups == grp].index.tolist()
                grp_rets = rets[rets.index.isin(grp_stocks)]
                if len(grp_rets) > 0:
                    record[f"group_{grp}"] = grp_rets.mean()  # 等权平均
                else:
                    record[f"group_{grp}"] = 0.0

            # 多空：G1 - GN
            record["long_short"] = record.get(f"group_1", 0.0) - record.get(f"group_{n_groups}", 0.0)
            daily_records.append(record)

    # 汇总结果
    gr = pd.DataFrame(daily_records).set_index("trade_date")
    gr.index = pd.to_datetime(gr.index)
    gr = gr.sort_index()

    if gr.empty:
        raise RuntimeError("回测未产生任何收益数据，请检查数据覆盖范围")

    # 累计净值
    nav = (1 + gr).cumprod()

    print(f"[backtest] 回测完成，{len(gr)} 个交易日，{len(factor_dates)} 次调仓")
    return {
        "group_returns": gr,
        "group_nav": nav,
    }
