#!/usr/bin/env python3
"""
Unified evaluation script for DQN/PPO/SAC/TD3.

Uses consistent environment configs and saves per-episode stats + summary CSV.
Loads VecNormalize for PPO to ensure correct observation normalization.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
import sys

import numpy as np
import pandas as pd
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env_config import get_env_config, get_continuous_env_config
from reward_wrappers import LaneCenteringOvertakeReward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained agent.")
    parser.add_argument(
        "--agent",
        choices=["dqn", "ppo", "sac", "td3"],
        required=True,
        help="Agent type to evaluate.",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable human rendering (slower).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for evaluation.",
    )
    return parser.parse_args()


def build_env(agent_type: str, render: bool):
    render_mode = "human" if render else None
    env = gym.make("highway-v0", render_mode=render_mode)
    if agent_type in ("sac", "td3"):
        config = get_continuous_env_config()
    else:
        config = get_env_config()
    env.unwrapped.config.update(config)
    if agent_type in ("sac", "td3"):
        env = LaneCenteringOvertakeReward(env)
    return env, config


def load_model(agent_type: str, agent_dir: str):
    if agent_type == "ppo":
        return PPO.load(f"{agent_dir}/model")
    if agent_type == "dqn":
        return DQN.load(f"{agent_dir}/model")
    if agent_type == "sac":
        return SAC.load(f"{agent_dir}/model")
    if agent_type == "td3":
        return TD3.load(f"{agent_dir}/model")
    raise ValueError(f"Unsupported agent: {agent_type}")


def main() -> None:
    args = parse_args()

    agent_dir = {
        "ppo": "ppo_agent",
        "dqn": "dqn_agent",
        "sac": "sac_agent",
        "td3": "td3_agent",
    }[args.agent]

    os.makedirs(f"{agent_dir}/instant_runs", exist_ok=True)
    os.makedirs(f"{agent_dir}/summary", exist_ok=True)

    model = load_model(args.agent, agent_dir)

    base_env, config = build_env(args.agent, args.render)

    # PPO uses VecNormalize during training; load it for evaluation.
    if args.agent == "ppo":
        vec_env = DummyVecEnv([lambda: base_env])
        norm_path = os.path.join(agent_dir, "vec_normalize.pkl")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Missing VecNormalize file: {norm_path}")
        env = VecNormalize.load(norm_path, vec_env)
        env.training = False
        env.norm_reward = False
        if args.seed is not None:
            env.seed(args.seed)
        obs = env.reset()
        eval_env = env.envs[0]
    else:
        env = base_env
        obs, _ = env.reset(seed=args.seed)
        eval_env = env

    print("\n" + "=" * 70)
    print(f"🚗 Evaluating {args.agent.upper()} Agent")
    print("=" * 70)
    print("Environment: highway-v0")
    print(
        f"Action Space: {'Continuous (Box)' if args.agent in ('sac', 'td3') else 'Discrete(5)'}"
    )
    print(f"Vehicles: {config['vehicles_count']}")
    print(f"Duration: {config['duration']}s")
    print(f"Policy frequency: {config['policy_frequency']} Hz")
    print("=" * 70 + "\n")

    all_episode_stats = []
    SAFE_TTC_THRESHOLD = 2.0

    for ep in range(args.episodes):
        if args.agent == "ppo":
            if args.seed is not None:
                env.seed(args.seed)
            obs = env.reset()[0]
        else:
            obs, _ = env.reset(seed=args.seed)
        done = truncated = False

        ep_reward = 0.0
        ep_steps = 0
        ep_speed_sum = 0.0
        ep_lane_changes = 0

        previous_speed = 0.0
        previous_acc = 0.0
        jerk_values = []
        ttc_values = []

        ego_vehicle = eval_env.unwrapped.vehicle
        previous_lane_index = (
            ego_vehicle.lane_index[2] if ego_vehicle.lane_index is not None else None
        )

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            if args.agent == "ppo":
                if not hasattr(action, "__len__") or np.shape(action) == ():
                    action = [action]
                obs, rewards, dones, infos = env.step(action)
                reward = float(rewards[0])
                done = bool(dones[0])
                info = infos[0]
                truncated = False
            else:
                obs, reward, done, truncated, info = env.step(action)
                reward = float(reward)
            ep_reward += reward
            ep_steps += 1

            ego_vehicle = eval_env.unwrapped.vehicle
            ego_pos = ego_vehicle.position
            ego_speed = ego_vehicle.speed
            ep_speed_sum += ego_speed

            acc = ego_speed - previous_speed
            jerk = acc - previous_acc
            jerk_values.append(abs(jerk))
            previous_speed = ego_speed
            previous_acc = acc

            min_ttc = float("inf")
            for other in eval_env.unwrapped.road.vehicles:
                if other is ego_vehicle:
                    continue
                if ego_vehicle.lane_index is None or other.lane_index is None:
                    continue
                if ego_vehicle.lane_index[2] != other.lane_index[2]:
                    continue
                rel_x = other.position[0] - ego_pos[0]
                if rel_x <= 0:
                    continue
                rel_v = ego_speed - other.speed
                if rel_v > 0.01:
                    ttc = rel_x / rel_v
                    min_ttc = min(min_ttc, ttc)

            if min_ttc != float("inf"):
                ttc_values.append(min_ttc)

            current_lane_index = (
                ego_vehicle.lane_index[2]
                if ego_vehicle.lane_index is not None
                else None
            )
            if (
                previous_lane_index is not None
                and current_lane_index is not None
                and current_lane_index != previous_lane_index
            ):
                ep_lane_changes += 1
            previous_lane_index = current_lane_index

        avg_speed = ep_speed_sum / (ep_steps + 1e-6)
        avg_jerk = float(np.mean(jerk_values)) if jerk_values else 0.0
        max_jerk = float(np.max(jerk_values)) if jerk_values else 0.0
        avg_ttc = float(np.mean(ttc_values)) if ttc_values else -1.0
        min_ttc = float(np.min(ttc_values)) if ttc_values else -1.0
        ttc_violations = sum(t < SAFE_TTC_THRESHOLD for t in ttc_values)
        ttc_violation_rate = ttc_violations / (ep_steps + 1e-6)

        all_episode_stats.append(
            {
                "episode": ep + 1,
                "total_reward": ep_reward,
                "steps": ep_steps,
                "avg_speed_ms": avg_speed,
                "lane_changes": ep_lane_changes,
                "avg_jerk": avg_jerk,
                "max_jerk": max_jerk,
                "avg_ttc": avg_ttc,
                "min_ttc": min_ttc,
                "ttc_violation_rate": ttc_violation_rate,
                "collision": info.get("crashed", False),
                "success": not info.get("crashed", False),
            }
        )

        print(
            f"Episode {ep+1}/{args.episodes}: Reward={ep_reward:.2f}, "
            f"Steps={ep_steps}, Crashed={info.get('crashed', False)}"
        )

    if args.agent == "ppo":
        env.close()
    else:
        env.close()

    df = pd.DataFrame(all_episode_stats)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df.to_csv(f"{agent_dir}/instant_runs/run_{timestamp}.csv", index=False)
    df.mean(numeric_only=True).to_frame("Value").to_csv(
        f"{agent_dir}/summary/summary_{timestamp}.csv"
    )

    print(f"\n{'='*70}")
    print(f"✅ {args.agent.upper()} Evaluation Complete!")
    print(f"📁 Results saved to: {agent_dir}/instant_runs/run_{timestamp}.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
