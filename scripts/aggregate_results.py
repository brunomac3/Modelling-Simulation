#!/usr/bin/env python3
"""
Aggregate per-run summary CSVs into mean/std tables.
"""

from __future__ import annotations

import glob
import argparse
import os
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation summaries.")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["dqn", "ppo", "sac", "td3"],
        help="Agent names to include.",
    )
    parser.add_argument(
        "--pattern",
        default="run*_summary_*.csv",
        help="Glob pattern for summary files inside each agent summary dir.",
    )
    parser.add_argument(
        "--out-mean",
        default="results_mean.csv",
        help="Output CSV for mean table.",
    )
    parser.add_argument(
        "--out-std",
        default="results_std.csv",
        help="Output CSV for std table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []

    for agent in args.agents:
        summary_dir = f"{agent}_agent/summary"
        for path in glob.glob(os.path.join(summary_dir, args.pattern)):
            df = pd.read_csv(path, index_col=0)
            row = {"agent": agent, "summary_path": path}
            for metric, value in df["Value"].items():
                row[metric] = value
            rows.append(row)

    if not rows:
        raise SystemExit("No summary CSVs found. Run evals first.")

    all_df = pd.DataFrame(rows)
    metrics = [c for c in all_df.columns if c not in ("agent", "summary_path")]
    mean_df = all_df.groupby("agent")[metrics].mean()
    std_df = all_df.groupby("agent")[metrics].std()

    print("\n=== Mean ===")
    print(mean_df)
    print("\n=== Std Dev ===")
    print(std_df)

    mean_df.to_csv(args.out_mean)
    std_df.to_csv(args.out_std)
    print(f"\nSaved: {args.out_mean}, {args.out_std}")


if __name__ == "__main__":
    main()
