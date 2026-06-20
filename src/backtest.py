"""
截面回测引擎 — 月频调仓，分组等权，日度追踪，多空合成。
"""

import pandas as pd
import numpy as np
import duckdb
from src.universe import filter_universe
from src.factors import get_factor

DB_PATH = "data/quant.duckdb"


def _get_db(db_path: str | None = None):
    """获取 DuckDB 只读连接"""
    return duckdb.connect(db_path or DB_PATH, read_only=True)


# ============================================================
# 交易日工具
# ============================================================
def _get_trading_days(start_date: str, end_date: str, db_path: str | None = None) -> pd.DatetimeIndex:
    """获取回测区间内的交易日序列"""
    db = _get_db(db_path)
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


def _nth_trading_day_after(date: pd.Timestamp, trading_days: pd.DatetimeIndex, n: int = 0) -> pd.Timestamp | None:
    """返回 date 之后第 n 个交易日（n=0 为第一个下一个交易日），无则返回 None"""
    candidates = trading_days[trading_days > date]
    if len(candidates) <= n:
        return None
    return candidates[n]


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

    # qcut 按分位数等分，默认将最小值放入第 1 个 bin。
    # 将因子值取反后，高因子值 → 更小的 bin → 更小的 group 编号。
    neg = -valid
    labels = list(range(1, n_groups + 1))  # group 1 = highest factor
    try:
        groups = pd.qcut(neg, q=n_groups, labels=labels)
    except ValueError:
        groups = pd.Series(
            pd.cut(neg.rank(method="first"), bins=n_groups, labels=labels).values,
            index=valid.index
        )
    return groups.sort_values(ascending=True)


# ============================================================
# 分组收益追踪 — 纯函数，不依赖 DuckDB
# ============================================================
def compute_group_returns(
    factor_values: list[tuple[pd.Timestamp, pd.Series]],
    daily_returns: dict[pd.Timestamp, pd.Series],
    trading_days: pd.DatetimeIndex,
    freq: str = "monthly",
    n_groups: int = 10,
    offset: int = 0,
    weighting: str = "equal",
    market_caps: dict[pd.Timestamp, pd.Series] | None = None,
) -> pd.DataFrame:
    """
    给定因子值和日收益，追踪分组每日收益。

    参数：
        factor_values:  [(日期, 因子值Series), ...] 按时间升序
        daily_returns:  {日期: 日收益Series}
        trading_days:   完整交易日序列
        freq:           "daily" | "weekly" | "monthly"
        n_groups:       分组数
        weighting:      "equal" | "market_cap"
        market_caps:    {日期: circ_mv Series}，仅 market_cap 模式使用

    返回：
        DataFrame，index=trade_date，columns=group_1..group_N + long_short
    """
    if not factor_values or len(trading_days) == 0:
        return pd.DataFrame()

    fv_by_date: dict[pd.Timestamp, pd.Series] = {}
    for d, fv in factor_values:
        fv_by_date[pd.Timestamp(d)] = fv

    if freq == "daily":
        factor_dates = trading_days
    elif freq == "weekly":
        factor_dates = _last_trading_day_of_week(trading_days)
    elif freq == "monthly":
        factor_dates = _last_trading_day_of_month(trading_days)
    else:
        raise ValueError(f"未知频率: {freq}")

    daily_records: list[dict[str, object]] = []

    for i, fc_date in enumerate(factor_dates):
        fv = fv_by_date.get(fc_date)
        if fv is None or len(fv.dropna()) < n_groups * 2:
            continue

        rebalance_date = _nth_trading_day_after(fc_date, trading_days, offset)
        if rebalance_date is None:
            continue

        if i + 1 < len(factor_dates):
            next_rebalance = _nth_trading_day_after(factor_dates[i + 1], trading_days, offset)
            if next_rebalance is None:
                hold_end = trading_days[-1]
            else:
                try:
                    idx = trading_days.tolist().index(next_rebalance)
                    hold_end = trading_days[idx - 1] if idx > 0 else next_rebalance
                except ValueError:
                    hold_end = trading_days[-1]
        else:
            hold_end = trading_days[-1]

        groups = assign_groups(fv.dropna(), n_groups)
        if groups.empty:
            continue

        mc_series = None
        if weighting == "market_cap" and market_caps is not None:
            mc_series = market_caps.get(fc_date)

        hold_days = trading_days[(trading_days >= rebalance_date) & (trading_days <= hold_end)]
        for day in hold_days:
            rets = daily_returns.get(day)
            if rets is None or rets.empty:
                continue

            record: dict[str, object] = {"trade_date": day}
            for grp in range(1, n_groups + 1):
                grp_stocks = groups[groups == grp].index.tolist()
                grp_rets = rets[rets.index.isin(grp_stocks)]
                if len(grp_rets) == 0:
                    record[f"group_{grp}"] = 0.0
                    continue

                if weighting == "market_cap" and mc_series is not None:
                    w = mc_series[mc_series.index.isin(grp_rets.index)]
                    w = w[w > 0]
                    common = grp_rets.index.intersection(w.index)
                    if len(common) > 0:
                        w_norm = w.loc[common] / w.loc[common].sum()
                        record[f"group_{grp}"] = float(
                            (grp_rets.loc[common] * w_norm).sum()
                        )
                    else:
                        record[f"group_{grp}"] = float(grp_rets.mean())
                else:
                    record[f"group_{grp}"] = float(grp_rets.mean())

            record["long_short"] = float(record.get("group_1", 0.0)) - float(record.get(f"group_{n_groups}", 0.0))
            daily_records.append(record)

    if not daily_records:
        return pd.DataFrame()

    gr = pd.DataFrame(daily_records).set_index("trade_date")
    gr.index = pd.to_datetime(gr.index)
    return gr.sort_index()


def _last_trading_day_of_week(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """从交易日序列中挑出每周最后一个交易日."""
    df = pd.DataFrame({"date": trading_days})
    df["iso"] = df["date"].dt.isocalendar().year.astype(str) + "-" + df["date"].dt.isocalendar().week.astype(str)
    return pd.DatetimeIndex(df.groupby("iso")["date"].max())


# ============================================================
# 数据加载
# ============================================================
def _load_daily_returns(ts_codes: list[str], date: str, db_path: str | None = None) -> pd.Series:
    """
    加载指定日期的股票日收益率。
    返回 Series，index=ts_code，值=日收益率（小数）
    """
    if len(ts_codes) == 0:
        return pd.Series(dtype=float)
    db = _get_db(db_path)
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


def _load_factor_data(date: str, ts_codes: list[str], db_path: str | None = None) -> pd.DataFrame:
    """
    加载指定日期的截面数据，用于计算因子。
    返回 DataFrame，index=ts_code，含 pb, total_mv, circ_mv, turnover_rate 列。
    """
    if len(ts_codes) == 0:
        return pd.DataFrame()
    db = _get_db(db_path)
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
    offset: int = 0,
    silent: bool = False,
    db_path: str | None = None,
    freq: str = "monthly",
) -> dict:
    """
    截面回测（支持日/周/月三种频率）。

    参数：
        start_date:  回测起始日 YYYYMMDD
        end_date:    回测结束日 YYYYMMDD
        factor_name: 因子名称（factors.py FACTOR_REGISTRY 中的 key）
        n_groups:    分组数（默认 10）
        standardize: 是否做截面 MAD 标准化（默认 True）
        offset:      调仓日偏移交易日数（多轨道用）
        freq:        调仓频率 "daily" | "weekly" | "monthly"
        db_path:     数据库路径（默认使用 data/quant.duckdb）

    返回字典：
        group_returns:  DataFrame，每列为一组 + long_short 的日度收益率
        group_nav:      DataFrame，各组 + 多空累计净值
        factor_values:  [(date_str, Series), ...]
        forward_returns: [Series, ...]
    """
    factor_fn = get_factor(factor_name)

    # 获取交易日序列
    all_days = _get_trading_days(start_date, end_date, db_path=db_path)
    if len(all_days) == 0:
        raise ValueError(f"区间 {start_date}~{end_date} 无交易日")

    # 因子计算日（按频率）
    if freq == "daily":
        factor_dates = all_days
    elif freq == "weekly":
        factor_dates = _last_trading_day_of_week(all_days)
    elif freq == "monthly":
        factor_dates = _last_trading_day_of_month(all_days)
    else:
        raise ValueError(f"未知频率: {freq}")

    freq_label = {"daily": "日频", "weekly": "周频", "monthly": "月频"}.get(freq, freq)

    if not silent:
        print(f"[backtest] 回测区间: {all_days[0].strftime('%Y%m%d')} ~ {all_days[-1].strftime('%Y%m%d')}")
        print(f"[backtest] 交易天数: {len(all_days)}，{freq_label}，调仓次数: {len(factor_dates)}，偏移: {offset}")
        print(f"[backtest] 因子: {factor_name}，分组数: {n_groups}，标准化: {standardize}")

    # 批量预加载区间内所有 daily_basic 和 daily 数据
    db = _get_db(db_path)
    all_basic = db.execute(
        "SELECT ts_code, trade_date, pb, total_mv, circ_mv, turnover_rate "
        "FROM daily_basic WHERE trade_date BETWEEN ? AND ?",
        [start_date, end_date]
    ).df()
    all_daily = db.execute(
        "SELECT ts_code, trade_date, pct_chg FROM daily WHERE trade_date BETWEEN ? AND ?",
        [start_date, end_date]
    ).df()
    db.close()

    all_daily["return"] = all_daily["pct_chg"] / 100.0

    basic_by_date: dict[str, pd.DataFrame] = {}
    for d, grp in all_basic.groupby("trade_date"):
        basic_by_date[str(d)] = grp.set_index("ts_code")

    daily_returns: dict[pd.Timestamp, pd.Series] = {}
    for d, grp in all_daily.groupby("trade_date"):
        daily_returns[pd.Timestamp(str(d))] = grp.set_index("ts_code")["return"]

    # 因子值计算（遍历因子计算日）
    factor_values_list: list[tuple[str, pd.Series]] = []  # [(date_str, Series)]
    factor_values_for_cgr: list[tuple[pd.Timestamp, pd.Series]] = []

    for fc_date in factor_dates:
        fc_str = fc_date.strftime("%Y%m%d")

        fac_data = basic_by_date.get(fc_str)
        if fac_data is None or fac_data.empty:
            continue

        universe = filter_universe(fc_str, db_path=db_path)
        if len(universe) < n_groups * 2:
            continue

        fac_data = fac_data[fac_data.index.isin(universe)]
        if fac_data.empty:
            continue

        fv = factor_fn(fac_data).dropna()
        if len(fv) < n_groups * 2:
            continue
        if standardize:
            fv = mad_standardize(fv)

        factor_values_list.append((fc_str, fv.copy()))
        factor_values_for_cgr.append((fc_date, fv.copy()))

    # 分组收益追踪（调用纯函数）
    gr = compute_group_returns(
        factor_values=factor_values_for_cgr,
        daily_returns=daily_returns,
        trading_days=all_days,
        freq=freq,
        n_groups=n_groups,
        offset=offset,
    )

    if gr.empty:
        raise RuntimeError("回测未产生任何收益数据，请检查数据覆盖范围")

    # 前向收益：每期持仓期内各股累计收益（IC 评估用）
    forward_returns_list: list[pd.Series] = []
    for i, (fc_date, fv) in enumerate(factor_values_for_cgr):
        rebalance_date = _nth_trading_day_after(fc_date, all_days, offset)
        if rebalance_date is None:
            forward_returns_list.append(pd.Series(dtype=float))
            continue

        if i + 1 < len(factor_dates):
            next_rebalance = _nth_trading_day_after(factor_dates[i + 1], all_days, offset)
            if next_rebalance is None:
                hold_end = all_days[-1]
            else:
                idx = all_days.get_loc(next_rebalance)
                hold_end = all_days[idx - 1] if idx > 0 else next_rebalance
        else:
            hold_end = all_days[-1]

        hold_days = all_days[(all_days >= rebalance_date) & (all_days <= hold_end)]
        stocks = fv.index.tolist()
        stock_cum: dict[str, list[float]] = {c: [] for c in stocks}
        for day in hold_days:
            rets = daily_returns.get(day)
            if rets is None:
                continue
            for c in stocks:
                if c in rets.index:
                    stock_cum[c].append(rets[c])

        cum = pd.Series({
            c: (np.prod([1 + r for r in rets]) - 1) if rets else np.nan
            for c, rets in stock_cum.items()
        })
        cum.name = fc_date.strftime("%Y%m%d")
        forward_returns_list.append(cum)

    nav = (1 + gr).cumprod()

    if not silent:
        print(f"[backtest] 回测完成，{len(gr)} 个交易记录（含 {len(factor_dates)} 次调仓）")
    return {
        "group_returns": gr,
        "group_nav": nav,
        "factor_values": factor_values_list,
        "forward_returns": forward_returns_list,
    }


# ============================================================
# 多轨道偏移
# ============================================================
def run_pathways(
    start_date: str,
    end_date: str,
    factor_name: str,
    n_pathways: int = 15,
    n_groups: int = 10,
    standardize: bool = True,
    freq: str = "monthly",
) -> list[dict]:
    """
    多轨道回测：对调仓日偏移 0, 1, ..., n_pathways-1 个交易日，每条轨道独立回测。
    仅适用于月频，用于检验因子买点对调仓日选择的敏感度。

    参数：
        start_date, end_date, factor_name, n_groups, standardize: 同 run_backtest
        n_pathways: 轨道数（含基准轨道 k=0，默认 15）
        freq:        调仓频率（默认 "monthly"，仅月频推荐使用多轨道）

    返回：
        [{pathway, group_returns, group_nav}, ...] 按偏移量排序
    """
    results = []
    for k in range(n_pathways):
        print(f"\n[pathway] 轨道 k={k} / {n_pathways - 1}")
        try:
            res = run_backtest(
                start_date=start_date,
                end_date=end_date,
                factor_name=factor_name,
                n_groups=n_groups,
                standardize=standardize,
                offset=k,
                silent=(k > 0),
                freq=freq,
            )
            res["pathway"] = k
            results.append(res)
        except Exception as e:
            print(f"  [pathway k={k}] 失败: {e}")

    if not results:
        raise RuntimeError("所有轨道均回测失败")

    print(f"\n[pathway] 完成 {len(results)}/{n_pathways} 条轨道")
    return results

