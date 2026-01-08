#!/usr/bin/env python3
"""
Plot mean/std results from results_mean.csv and results_std.csv.
"""

from __future__ import annotations

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot evaluation results.")
    parser.add_argument(
        "--mean",
        default="results_mean.csv",
        help="CSV with mean values.",
    )
    parser.add_argument(
        "--std",
        default="results_std.csv",
        help="CSV with std values.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["avg_speed_ms", "collision", "success", "lane_changes", "avg_jerk"],
        help="Metrics to plot.",
    )
    parser.add_argument(
        "--out",
        default="results_plot.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mean_df = pd.read_csv(args.mean, index_col=0)
    std_df = pd.read_csv(args.std, index_col=0)

    metrics = [m for m in args.metrics if m in mean_df.columns]
    if not metrics:
        raise SystemExit("No matching metrics found in mean CSV.")

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    agents = mean_df.index.tolist()
    for ax, metric in zip(axes, metrics):
        means = mean_df[metric]
        stds = std_df[metric] if metric in std_df.columns else None
        ax.bar(agents, means, yerr=stds, capsize=4, color="#4E79A7", alpha=0.85)
        ax.set_title(metric.replace("_", " "), fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
