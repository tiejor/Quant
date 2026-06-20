"""
入口脚本 — 串联数据管线、回测、评估全流程。
用法: python src/main.py [--factor pb_factor] [--start 20200101] [--end 20241231]
"""

import os
import sys
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import duckdb

from src.pipeline import run as pipeline_run
from src.backtest import run_pathways, run_backtest
from src.evaluate import (
    evaluate_performance,
    evaluate_all_groups,
    plot_group_nav,
    plot_ic_timeline,
    plot_ic_timeline_with_cumulative,
    plot_ic_distribution,
    plot_ic_decay,
    plot_pathway_nav_overlay,
    compute_ic_series,
    evaluate_ic,
    compute_ic_lag_curve,
)

DB_PATH = "data/quant.duckdb"
OUTPUT_DIR = "output"


def ensure_dirs():
    """确保输出目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="ClaudeQuant 因子回测系统")
    parser.add_argument("--factor", default="pb_factor", help="因子名称（见 factors.py）")
    parser.add_argument("--start", default="20200101", help="回测起始日 YYYYMMDD")
    parser.add_argument("--end", default=None, help="回测结束日 YYYYMMDD（默认今天）")
    parser.add_argument("--groups", type=int, default=10, help="分组数（默认 10）")
    parser.add_argument("--pathways", type=int, default=15, help="轨道数（默认 15，仅月频生效）")
    parser.add_argument("--no-standardize", action="store_true", help="不做截面 MAD 标准化")
    parser.add_argument("--skip-pipeline", action="store_true", help="跳过数据拉取")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y%m%d")

    ensure_dirs()

    # 1. 数据管线（增量更新）
    if not args.skip_pipeline:
        print("=" * 60)
        print("步骤 1/3: 数据管线（增量更新）")
        print("=" * 60)
        try:
            pipeline_run()
        except RuntimeError as e:
            print(f"管线失败: {e}")
            print("若数据已是最新，可用 --skip-pipeline 跳过")
            sys.exit(1)

    # 2. 回测 + 评估（三频率）
    FREQUENCIES = [
        ("daily", "日频", False),
        ("weekly", "周频", False),
        ("monthly", "月频", True),
    ]
    max_lag_map = {"daily": 20, "weekly": 12, "monthly": 12}

    pd.set_option("display.float_format", "{:.4f}".format)
    all_results: dict[str, dict] = {}

    for freq, freq_label, use_pathways in FREQUENCIES:
        print("\n" + "=" * 60)
        print(f"步骤 2/3: 回测 — {freq_label} 因子={args.factor}, 区间={args.start}~{args.end}")
        print("=" * 60)
        t0 = time.time()

        try:
            if use_pathways:
                pathways = run_pathways(
                    start_date=args.start,
                    end_date=args.end,
                    factor_name=args.factor,
                    n_pathways=args.pathways,
                    n_groups=args.groups,
                    standardize=not args.no_standardize,
                    freq=freq,
                )
                base = pathways[0]
            else:
                base = run_backtest(
                    start_date=args.start,
                    end_date=args.end,
                    factor_name=args.factor,
                    n_groups=args.groups,
                    standardize=not args.no_standardize,
                    offset=0,
                    freq=freq,
                )
                pathways = None
        except Exception as e:
            print(f"回测失败 ({freq_label}): {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - t0
        print(f"回测耗时: {elapsed:.1f}s")

        gr = base["group_returns"]
        nav = base["group_nav"]
        fv = base.get("factor_values", [])
        fwd = base.get("forward_returns", [])

        # 绩效指标
        perf_df = evaluate_all_groups(gr)
        print(f"\n{freq_label} 绩效指标:")
        print(perf_df.to_string())

        # 分组净值图
        plot_group_nav(nav, args.factor, freq=freq)

        # IC 评估
        ic_metrics = None
        if fv and fwd:
            fv_series = [s for _, s in fv]
            ic = compute_ic_series(fv_series, fwd)
            ic_metrics = evaluate_ic(ic)
            print(f"\n{freq_label} IC 评估:")
            print(f"  IC 均值: {ic_metrics['ic_mean']:.4f}")
            print(f"  IR:      {ic_metrics['ic_ir']:.4f}")
            print(f"  t 值:    {ic_metrics['t_stat']:.2f}  (p={ic_metrics['p_value']:.4f})")
            print(f"  期数:    {ic_metrics['n_periods']}")

            if freq == "monthly":
                plot_ic_timeline_with_cumulative(ic, args.factor, freq=freq)
            else:
                plot_ic_timeline(ic, args.factor, freq=freq)
            plot_ic_distribution(ic, args.factor, freq=freq)

            # IC 衰减曲线（lag-based）
            max_lag = max_lag_map.get(freq, 12)
            ic_curve, half_life = compute_ic_lag_curve(fv_series, fwd, max_lag=max_lag)
            if ic_curve:
                print(f"  IC 半衰期 (lag-based): {half_life} 期")
            plot_ic_decay(ic_curve, half_life, args.factor, freq=freq)

        # 多轨道汇总（仅月频）
        if pathways and len(pathways) > 1:
            print(f"\n多轨道汇总（{len(pathways)} 条轨道）:")
            all_perf = {}
            for pw in pathways:
                perf = evaluate_performance(pw["group_returns"]["long_short"])
                for metric, val in perf.items():
                    if metric not in all_perf:
                        all_perf[metric] = []
                    all_perf[metric].append(val)
            for metric, vals in all_perf.items():
                vals_arr = pd.Series(vals).dropna()
                if len(vals_arr) > 0:
                    print(f"  {metric}: {vals_arr.mean():.4f} ± {vals_arr.std():.4f}")

            plot_pathway_nav_overlay(pathways, args.factor, freq=freq)

        # 收集跨频率汇总数据
        ls = gr["long_short"] if "long_short" in gr.columns else pd.Series(dtype=float)
        all_results[freq_label] = {
            "long_short": ls,
            "ic": ic_metrics,
        }

    # 跨频率汇总
    if all_results:
        print("\n" + "=" * 60)
        print("频率对比汇总")
        print("=" * 60)
        rows = []
        for freq_label, data in all_results.items():
            row = {"频率": freq_label}
            ls = data["long_short"]
            if not ls.empty:
                perf = evaluate_performance(ls)
                row["多空年化收益"] = perf.get("ann_return", np.nan)
                row["多空夏普"] = perf.get("sharpe", np.nan)
                row["多空最大回撤"] = perf.get("max_drawdown", np.nan)
            ic_d = data.get("ic")
            if ic_d:
                row["IC均值"] = ic_d.get("ic_mean", np.nan)
                row["IC_IR"] = ic_d.get("ic_ir", np.nan)
            rows.append(row)
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n图表已保存到 {OUTPUT_DIR}/ 目录")


if __name__ == "__main__":
    main()
