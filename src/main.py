"""
入口脚本 — 串联数据管线、回测、评估全流程。
用法: python src/main.py [--factor pb_factor] [--start 20200101] [--end 20241231]
"""

import os
import sys
import time
import argparse
from datetime import datetime

import pandas as pd
import duckdb

from src.pipeline import run as pipeline_run
from src.backtest import run_pathways
from src.evaluate import (
    evaluate_performance,
    evaluate_all_groups,
    plot_group_nav,
    plot_ic_timeline,
    plot_ic_decay,
    compute_ic_series,
    evaluate_ic,
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
    parser.add_argument("--pathways", type=int, default=5, help="轨道数（默认 5）")
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

    # 2. 回测（多轨道）
    print("\n" + "=" * 60)
    print(f"步骤 2/3: 多轨道回测 — 因子={args.factor}, 区间={args.start}~{args.end}")
    print("=" * 60)
    t0 = time.time()

    try:
        pathways = run_pathways(
            start_date=args.start,
            end_date=args.end,
            factor_name=args.factor,
            n_pathways=args.pathways,
            n_groups=args.groups,
            standardize=not args.no_standardize,
        )
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"回测耗时: {elapsed:.1f}s")

    # 3. 评估 + 出图
    print("\n" + "=" * 60)
    print("步骤 3/3: 因子评估")
    print("=" * 60)

    base = pathways[0]  # 基准轨道（k=0）
    gr = base["group_returns"]
    nav = base["group_nav"]

    # 绩效指标
    perf_df = evaluate_all_groups(gr)
    print("\n基准轨道绩效指标:")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(perf_df.to_string())

    # 出图
    plot_group_nav(nav, args.factor)

    # IC 评估
    fv = base.get("factor_values", [])
    fwd = base.get("forward_returns", [])
    if fv and fwd:
        fv_series = [s for _, s in fv]
        ic = compute_ic_series(fv_series, fwd)
        ic_metrics = evaluate_ic(ic)
        print(f"\n基准轨道 IC 评估:")
        print(f"  IC 均值: {ic_metrics['ic_mean']:.4f}")
        print(f"  IR:      {ic_metrics['ic_ir']:.4f}")
        print(f"  t 值:    {ic_metrics['t_stat']:.2f}  (p={ic_metrics['p_value']:.4f})")
        print(f"  半衰期:  {ic_metrics['half_life']} 期")
        print(f"  期数:    {ic_metrics['n_periods']}")

        plot_ic_timeline(ic, args.factor)
        plot_ic_decay(ic, args.factor)

    # 多轨道汇总
    print(f"\n多轨道汇总（{len(pathways)} 条轨道）:")
    all_perf = {}
    for pw in pathways:
        k = pw["pathway"]
        perf = evaluate_performance(pw["group_returns"]["long_short"])
        for metric, val in perf.items():
            if metric not in all_perf:
                all_perf[metric] = []
            all_perf[metric].append(val)

    for metric, vals in all_perf.items():
        vals_arr = pd.Series(vals).dropna()
        if len(vals_arr) > 0:
            print(f"  {metric}: {vals_arr.mean():.4f} ± {vals_arr.std():.4f}")

    print(f"\n图表已保存到 {OUTPUT_DIR}/ 目录")


if __name__ == "__main__":
    main()
