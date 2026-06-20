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
    ic_half_life 字段已废弃，推荐使用 compute_ic_lag_curve 的 lag-based 半衰期。
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

    from scipy.stats import t as t_dist
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1)) if not np.isnan(t_stat) else np.nan

    return {
        "ic_mean": ic_mean,
        "ic_ir": ic_ir,
        "t_stat": t_stat,
        "p_value": p_value,
        "half_life": np.nan,
        "n_periods": n,
    }


def ic_summary(ic_series: pd.Series) -> dict:
    """IC 统计概要（不含半衰期）。
    返回 ic_mean, ic_std, ic_ir, t_stat, p_value, win_rate, sig_ratio, n_periods。
    """
    ic = ic_series.dropna()
    n = len(ic)
    if n < 2:
        return {
            "ic_mean": np.nan if n == 0 else ic.iloc[0],
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "win_rate": np.nan if n == 0 else float(ic.iloc[0] > 0),
            "sig_ratio": np.nan if n == 0 else float(abs(ic.iloc[0]) >= 0.02),
            "n_periods": n,
        }

    ic_mean = ic.mean()
    ic_std = ic.std(ddof=1)
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
    t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else np.nan

    from scipy.stats import t as t_dist
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1)) if not np.isnan(t_stat) else np.nan

    win_rate = float((ic > 0).mean())
    sig_ratio = float((ic.abs() >= 0.02).mean())

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "t_stat": t_stat,
        "p_value": p_value,
        "win_rate": win_rate,
        "sig_ratio": sig_ratio,
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


def compute_ic_lag_curve(
    factor_values: list[pd.Series],
    daily_returns: list[pd.Series],
    max_lag: int = 30,
) -> tuple[dict[int, float], int | None]:
    """
    IC(N) lag-based 半衰期计算。
    IC(N) = (1/M) * Σ Pearson(rank(f_T), rank(r_{T+N}))
    对所有截面 T 计算 rank IC，取均值。

    参数：
        factor_values:  每期因子值列表，按时间升序
        daily_returns:  每日收益列表，与 factor_values 同频且时间对齐
        max_lag:        最大滞后天数

    返回：
        ic_curve: {lag: mean_IC} 字典
        half_life: IC 从 lag=1 峰值衰减到一半的第一个 lag，或 None
    """
    n = min(len(factor_values), len(daily_returns))
    if n < 3:
        return {}, None

    ic_curve: dict[int, float] = {}

    for lag in range(1, max_lag + 1):
        ics = []
        for t in range(n - lag):
            fv = factor_values[t]
            ret = daily_returns[t + lag]
            common = fv.index.intersection(ret.index)
            if len(common) < 10:
                continue
            ic = fv.loc[common].corr(ret.loc[common], method="spearman")
            if not pd.isna(ic):
                ics.append(ic)
        if ics:
            ic_curve[lag] = float(np.mean(ics))
        elif ic_curve:
            break  # 后续 lag 也不会更多数据
        # else: keep trying — 可能是 min_common < 10 导致的空

    if not ic_curve or 1 not in ic_curve:
        return ic_curve, None

    ic1 = ic_curve[1]
    if abs(ic1) < 1e-10:
        return ic_curve, None

    half_life = None
    for lag in sorted(ic_curve.keys()):
        if abs(ic_curve[lag]) < abs(ic1) / 2:
            half_life = lag
            break

    return ic_curve, half_life


def compute_sortino_ratio(daily_returns: pd.Series, mar_annual: float = 0.03) -> float:
    """
    Sortino 比率：年化收益率 / (年化下行波动率)。
    MAR 为年化最低可接受收益率，默认 0.03（3%）。
    输入日度收益率（小数），无下行收益时返回 +inf。
    """
    r = daily_returns.dropna()
    if len(r) == 0:
        return np.nan

    ann_return = 252 * r.mean()
    mar_daily = mar_annual / 252
    downside = r[r < mar_daily]

    if len(downside) == 0:
        return np.inf

    ann_downside_vol = np.sqrt(252) * downside.std(ddof=1)
    return ann_return / ann_downside_vol if ann_downside_vol > 1e-8 else np.nan


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


def plot_group_nav(group_nav: pd.DataFrame, factor_name: str, freq: str = "", filename: str = None):
    """分组累计净值曲线图"""
    _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in group_nav.columns:
        if col == "long_short":
            ax.plot(group_nav.index, group_nav[col], linewidth=2, label="多空", color="black")
        else:
            ax.plot(group_nav.index, group_nav[col], linewidth=0.6, alpha=0.7, label=col)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.5)
    title_freq = f" — {freq}" if freq else ""
    ax.set_title(f"{factor_name}{title_freq} 分组累计净值", fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("累计净值")
    ax.legend(loc="upper left", ncol=2, fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    if filename is None:
        freq_suffix = f"_{freq}" if freq else ""
        filename = f"{OUTPUT_DIR}/{factor_name}{freq_suffix}_nav.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[evaluate] 分组净值图 → {filename}")


def plot_ic_timeline(ic_series: pd.Series, factor_name: str, freq: str = "", filename: str = None):
    """IC 时间序列图"""
    _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(ic_series)), ic_series.values, color="steelblue", alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(y=ic_series.mean(), color="red", linestyle="-", linewidth=1,
               label=f"均值={ic_series.mean():.4f}")
    title_freq = f" — {freq}" if freq else ""
    ax.set_title(f"{factor_name}{title_freq} Rank IC 时间序列", fontsize=14)
    ax.set_xlabel("期数")
    ax.set_ylabel("Rank IC")
    ax.legend()
    fig.tight_layout()
    if filename is None:
        freq_suffix = f"_{freq}" if freq else ""
        filename = f"{OUTPUT_DIR}/{factor_name}{freq_suffix}_ic.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[evaluate] IC 时序图 → {filename}")


def plot_ic_decay(ic_curve: dict[int, float], half_life: int | None,
                  factor_name: str, freq: str = "", filename: str = None):
    """IC(N) 衰减图 — 基于 lag-based 半衰期（compute_ic_lag_curve 输出）"""
    if not ic_curve:
        return
    _setup_chinese_font()
    lags = sorted(ic_curve.keys())
    vals = [ic_curve[lag] for lag in lags]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lags, vals, marker="o", color="steelblue", linewidth=1.5, label="mean IC(N)")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    if 1 in ic_curve:
        ic1 = ic_curve[1]
        ax.axhline(y=ic1, color="green", linestyle="--", linewidth=0.5,
                   alpha=0.5, label=f"IC(1)={ic1:.4f}")
        ax.axhline(y=ic1 / 2, color="red", linestyle="--", linewidth=0.5,
                   alpha=0.5, label=f"半衰线={ic1/2:.4f}")

    if half_life is not None and half_life in ic_curve:
        ax.axvline(x=half_life, color="red", linestyle=":", linewidth=0.8)
        ax.annotate(f"半衰期 N={half_life}", xy=(half_life, ic_curve[half_life]),
                    xytext=(half_life + 1, ic_curve[half_life] * 0.7),
                    arrowprops=dict(arrowstyle="->", color="red"), fontsize=10)

    title_freq = f" — {freq}" if freq else ""
    ax.set_title(f"{factor_name}{title_freq} IC(N) 衰减", fontsize=14)
    ax.set_xlabel("Lag N（期）")
    ax.set_ylabel("mean IC(N)")
    ax.legend()
    fig.tight_layout()
    if filename is None:
        freq_suffix = f"_{freq}" if freq else ""
        filename = f"{OUTPUT_DIR}/{factor_name}{freq_suffix}_decay.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[evaluate] IC 衰减图 → {filename}")


def plot_ic_timeline_with_cumulative(ic_series: pd.Series, factor_name: str, freq: str = "", filename: str = None):
    """月频 IC 时序柱状图 + 累计折线双面板"""
    _setup_chinese_font()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.bar(range(len(ic_series)), ic_series.values, color="steelblue", alpha=0.8, label="Rank IC")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("Rank IC")

    cum = ic_series.cumsum()
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(cum)), cum.values, color="darkorange", linewidth=2, label="IC 累计")
    ax1_twin.set_ylabel("IC 累计")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    title_freq = f" — {freq}" if freq else ""
    ax1.set_title(f"{factor_name}{title_freq} IC 时序 + 累计", fontsize=14)

    ax2.bar(range(len(ic_series)), cum.values, color="darkorange", alpha=0.8)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("期数")
    ax2.set_ylabel("IC 累计")
    ax2.set_title("IC 累计时序", fontsize=12)

    fig.tight_layout()
    if filename is None:
        freq_suffix = f"_{freq}" if freq else ""
        filename = f"{OUTPUT_DIR}/{factor_name}{freq_suffix}_ic_cumulative.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[evaluate] IC 时序+累计 → {filename}")


def plot_ic_distribution(ic_series: pd.Series, factor_name: str, freq: str = "", filename: str = None):
    """IC 分布图：直方图 + KDE 密度曲线"""
    if len(ic_series.dropna()) < 3:
        return
    _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 5))
    ic = ic_series.dropna()
    ax.hist(ic, bins=20, density=True, alpha=0.6, color="steelblue", edgecolor="white")
    ic.plot.kde(ax=ax, color="darkorange", linewidth=2)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(x=ic.mean(), color="red", linestyle="-", linewidth=1,
               label=f"均值={ic.mean():.4f}")
    title_freq = f" — {freq}" if freq else ""
    ax.set_title(f"{factor_name}{title_freq} Rank IC 分布", fontsize=14)
    ax.set_xlabel("Rank IC")
    ax.set_ylabel("密度")
    ax.legend()
    fig.tight_layout()
    if filename is None:
        freq_suffix = f"_{freq}" if freq else ""
        filename = f"{OUTPUT_DIR}/{factor_name}{freq_suffix}_ic_dist.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[evaluate] IC 分布图 → {filename}")


def plot_pathway_nav_overlay(pathway_results: list[dict], factor_name: str, freq: str = "", filename: str = None):
    """多轨道多空净值叠加图"""
    if not pathway_results:
        return
    _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 6))
    for r in pathway_results:
        gr = r.get("group_returns", pd.DataFrame())
        if gr.empty or "long_short" not in gr.columns:
            continue
        nav = (1 + gr["long_short"]).cumprod()
        k = r.get("pathway", 0)
        ax.plot(nav.index, nav, linewidth=0.6, alpha=0.7, label=f"k={k}")
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.5)
    title_freq = f" — {freq}" if freq else ""
    ax.set_title(f"{factor_name}{title_freq} 多轨道多空净值叠加", fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("累计净值")
    ax.legend(loc="upper left", ncol=3, fontsize=7)
    fig.tight_layout()
    if filename is None:
        freq_suffix = f"_{freq}" if freq else ""
        filename = f"{OUTPUT_DIR}/{factor_name}{freq_suffix}_pathway_nav.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"[evaluate] 多轨道净值叠加图 → {filename}")
