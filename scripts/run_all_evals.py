#!/usr/bin/env python3
"""
Run evaluations for multiple agents and aggregate mean/std results.
"""

from __future__ import annotations

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evals for all agents.")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["dqn", "ppo", "sac", "td3"],
        help="Agent names to evaluate.",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per run.")
    parser.add_argument("--runs", type=int, default=3, help="Runs per agent.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering (slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for agent in args.agents:
        cmd = [
            "python",
            "scripts/eval.py",
            "--agent",
            agent,
            "--episodes",
            str(args.episodes),
            "--runs",
            str(args.runs),
        ]
        if args.render:
            cmd.append("--render")
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    subprocess.run(["python", "scripts/aggregate_results.py"], check=True)
    subprocess.run(["python", "scripts/plot_results.py"], check=True)


if __name__ == "__main__":
    main()
