"""
因子评估模块。
包含 IC 评估（不依赖回测）和绩效评估（依赖回测结果），以及可视化。
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_DIR = "output"

# ============================================================
# IC 评估（不依赖回测）
# ============================================================
def compute_ic_series(
    factor_values: list[pd.Series],
    forward_returns: list[pd.Series],
) -> pd.Series:
    """
    计算 Rank IC 序列。
    参数：
        factor_values:  每期因子值 Series 列表（index=ts_code）
        forward_returns: 每期未来收益 Series 列表（index=ts_code，小数形式）
    返回：
        每期 Spearman 秩相关系数 Series
    """
    ic_list = []
    dates = []
    # 最后一期无未来收益，对齐到 n-1
    n = min(len(factor_values), len(forward_returns))
    for i in range(n):
        fv = factor_values[i]
        fr = forward_returns[i]
        # 对齐 index
        common = fv.index.intersection(fr.index)
        if len(common) < 10:
            continue
        ic = fv.loc[common].corr(fr.loc[common], method="spearman")
        ic_list.append(ic)
        dates.append(fv.name if fv.name else i)
    return pd.Series(ic_list, index=dates, name="RankIC")


def evaluate_ic(ic_series: pd.Series) -> dict:
    """
    核心 IC 评估指标。
    返回字典：
        ic_mean:    Rank IC 均值
        ic_ir:      Information Ratio
        t_stat:     IC 序列 t 统计量
        p_value:    t 检验 p 值
        half_life:  IC 半衰期（期数）
    """
    ic = ic_series.dropna()
    n = len(ic)
    if n < 2:
        return {"ic_mean": np.nan, "ic_ir": np.nan, "t_stat": np.nan,
                "p_value": np.nan, "half_life": np.nan, "n_periods": n}

    ic_mean = ic.mean()
    ic_std = ic.std(ddof=1)
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
    t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else np.nan

    # 双尾 t 检验 p 值
    from scipy.stats import t as t_dist
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1)) if not np.isnan(t_stat) else np.nan

    # 半衰期：逐期滞后 IC，找到衰减到初始一半的期数
    half_life = None
    ic0 = ic.iloc[0] if n > 0 else None
    if ic0 is not None and abs(ic0) > 1e-8:
        for lag in range(1, n):
            lag_ic = ic.autocorr(lag=lag)
            if lag_ic is not None and abs(lag_ic) < abs(ic0) / 2:
                half_life = lag
                break
        if half_life is None:
            half_life = n  # 未衰减到一半，返回最大期数

    return {
        "ic_mean": ic_mean,
        "ic_ir": ic_ir,
        "t_stat": t_stat,
        "p_value": p_value,
        "half_life": half_life,
        "n_periods": n,
    }


# ============================================================
# 绩效评估（依赖回测结果）
# ============================================================
def evaluate_performance(daily_returns: pd.Series) -> dict:
    """
    计算单条收益序列的绩效指标。
    输入日度收益率（小数），返回各绩效指标。
    """
    r = daily_returns.dropna()
    n = len(r)
    if n == 0:
        return {}

    ann_return = 252 * r.mean()
    ann_vol = np.sqrt(252) * r.std(ddof=1)
    sharpe = ann_return / ann_vol if ann_vol > 1e-8 else 0.0

    # 最大回撤
    nav = (1 + r).cumprod()
    peak = nav.cummax()
    drawdown = (nav / peak) - 1
    max_dd = drawdown.min()

    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "n_days": n,
    }


def evaluate_all_groups(group_returns: pd.DataFrame) -> pd.DataFrame:
    """
    对回测产出的各组 + 多空收益，逐列计算绩效指标。
    返回 DataFrame，行=指标名，列=组名。
    """
    if group_returns.empty:
        return pd.DataFrame()
    metrics = {}
    for col in group_returns.columns:
        perf = evaluate_performance(group_returns[col])
        if perf:
            metrics[col] = perf
    return pd.DataFrame(metrics)


# ============================================================
# 可视化（3 张图）
# ============================================================
def _setup_chinese_font():
    """配置中文字体渲染"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def plot_group_nav(group_nav: pd.DataFrame, factor_name: str, filename: str = None):
    """分组累计净值曲线图"""
    _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in group_nav.columns:
        if col == "long_short":
            ax.plot(group_nav.index, group_nav[col], linewidth=2, label="多空", color="black")
        else:
            ax.plot(group_nav.index, group_nav[col], linewidth=0.6, alpha=0.7, label=col)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title(f"{factor_name} — 分组累计净值", fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("累计净值")
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    if filename is None:
        filename = f"{OUTPUT_DIR}/{factor_name}_nav.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[evaluate] 分组净值图 → {filename}")


def plot_ic_timeline(ic_series: pd.Series, factor_name: str, filename: str = None):
    """IC 时间序列图"""
    _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(ic_series)), ic_series.values, color="steelblue", alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(y=ic_series.mean(), color="red", linestyle="-", linewidth=1,
               label=f"均值={ic_series.mean():.4f}")
    ax.set_title(f"{factor_name} — Rank IC 时间序列", fontsize=14)
    ax.set_xlabel("期数")
    ax.set_ylabel("Rank IC")
    ax.legend()
    fig.tight_layout()
    if filename is None:
        filename = f"{OUTPUT_DIR}/{factor_name}_ic.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[evaluate] IC 时序图 → {filename}")


def plot_ic_decay(ic_series: pd.Series, factor_name: str, max_lag: int = 12, filename: str = None):
    """IC 衰减（半衰期）图"""
    _setup_chinese_font()
    lags = range(1, min(max_lag + 1, len(ic_series)))
    autocorrs = [ic_series.autocorr(lag=lag) for lag in lags]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags, autocorrs, marker="o", color="darkorange", linewidth=1.5)
    ax.axhline(y=ic_series.iloc[0], color="gray", linestyle="--", linewidth=0.5, label="初始 IC")
    ax.axhline(y=ic_series.iloc[0] / 2, color="red", linestyle="--", linewidth=0.5, label="半衰线")
    ax.set_title(f"{factor_name} — IC 衰减", fontsize=14)
    ax.set_xlabel("滞后期数")
    ax.set_ylabel("自相关系数")
    ax.legend()
    fig.tight_layout()
    if filename is None:
        filename = f"{OUTPUT_DIR}/{factor_name}_decay.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[evaluate] IC 衰减图 → {filename}")
