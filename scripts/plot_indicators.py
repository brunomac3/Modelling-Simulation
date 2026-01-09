#!/usr/bin/env python3
"""
Compute and plot SI/EI/CI/RCI/GPS with mean/std across runs.
"""

from __future__ import annotations

import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt


SPEED_LIMIT_MS = 32.0
SAFE_TTC_THRESHOLD = 2.0
MAX_EXPECTED_JERK = 10.0
MAX_EXPECTED_LANE_CHANGES = 20.0

WEIGHTS = {
    "safety": {"collision": 0.4, "ttc": 0.4, "ttc_violations": 0.2},
    "efficiency": {"speed": 0.5, "success": 0.5},
    "comfort": {"jerk": 0.6, "lane_changes": 0.4},
    "global": {"safety": 0.40, "efficiency": 0.30, "comfort": 0.15, "compliance": 0.15},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot indicator results.")
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
        "--out",
        default="results_indicators_plot.png",
        help="Output image path.",
    )
    return parser.parse_args()


def compute_indicators(row: pd.Series) -> dict[str, float]:
    collision_rate = float(row["collision"])
    avg_ttc = float(row["avg_ttc"])
    ttc_violation_rate = float(row["ttc_violation_rate"])
    avg_speed = float(row["avg_speed_ms"])
    success_rate = float(row["success"])
    avg_jerk = float(row["avg_jerk"])
    lane_changes = float(row["lane_changes"])

    ttc_norm = min(avg_ttc / SAFE_TTC_THRESHOLD, 1.0) if avg_ttc > 0 else 0.0
    speed_ratio = min(avg_speed / SPEED_LIMIT_MS, 1.0)

    jerk_norm = min(avg_jerk / MAX_EXPECTED_JERK, 1.0)
    lane_norm = min(lane_changes / MAX_EXPECTED_LANE_CHANGES, 1.0)

    si = (
        WEIGHTS["safety"]["collision"] * (1 - collision_rate)
        + WEIGHTS["safety"]["ttc"] * ttc_norm
        + WEIGHTS["safety"]["ttc_violations"] * (1 - ttc_violation_rate)
    )
    ei = (
        WEIGHTS["efficiency"]["speed"] * speed_ratio
        + WEIGHTS["efficiency"]["success"] * success_rate
    )
    ci = 1 - (
        WEIGHTS["comfort"]["jerk"] * jerk_norm
        + WEIGHTS["comfort"]["lane_changes"] * lane_norm
    )
    ci = max(ci, 0.0)
    rci = speed_ratio
    gps = (
        WEIGHTS["global"]["safety"] * si
        + WEIGHTS["global"]["efficiency"] * ei
        + WEIGHTS["global"]["comfort"] * ci
        + WEIGHTS["global"]["compliance"] * rci
    )
    return {
        "SI": si,
        "EI": ei,
        "CI": ci,
        "RCI": rci,
        "GPS": gps,
    }


def main() -> None:
    args = parse_args()
    rows = []

    for agent in args.agents:
        summary_dir = f"{agent}_agent/summary"
        for path in glob.glob(os.path.join(summary_dir, args.pattern)):
            df = pd.read_csv(path, index_col=0)
            row = df["Value"]
            indicators = compute_indicators(row)
            indicators["agent"] = agent
            rows.append(indicators)

    if not rows:
        raise SystemExit("No summary CSVs found. Run evals first.")

    df_all = pd.DataFrame(rows)
    mean_df = df_all.groupby("agent")[["SI", "EI", "CI", "RCI", "GPS"]].mean()
    std_df = df_all.groupby("agent")[["SI", "EI", "CI", "RCI", "GPS"]].std()

    mean_df.to_csv("results_indicators_mean.csv")
    std_df.to_csv("results_indicators_std.csv")

    metrics = ["SI", "EI", "CI", "RCI", "GPS"]
    agents = mean_df.index.tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(agents))
    width = 0.15

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
    for i, metric in enumerate(metrics):
        means = mean_df[metric].values
        stds = std_df[metric].values
        ax.bar(
            [xi + i * width for xi in x],
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=metric,
            color=colors[i],
            alpha=0.85,
            edgecolor="black",
            linewidth=1.0 if metric != "GPS" else 2.0,
        )

    ax.set_xticks([xi + 2 * width for xi in x])
    ax.set_xticklabels(agents)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Indicators (Mean ± Std across runs)")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1.5)
    ax.axhline(y=0.75, color="green", linestyle="--", alpha=0.4, linewidth=1.5)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", ncol=5)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved indicator plot to {args.out}")


if __name__ == "__main__":
    main()
