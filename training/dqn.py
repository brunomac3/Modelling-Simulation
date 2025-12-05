from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import highway_env
import os

AGENT_NAME = "dqn_agent"
TOTAL_TIMESTEPS = 400_000
os.makedirs(AGENT_NAME, exist_ok=True)

def make_env():
    env = gym.make("highway-v0")
    env.unwrapped.config.update({
            "observation": {"type": "Kinematics"},
            "vehicles_count": 35,
            "duration": 300,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "reward_speed_range": [20, 32],
            "collision_penalty": -8,
            "right_lane_reward": 0.1,
            "lane_change_reward": -0.03,
            "high_speed_reward": 0.6,
    })
    return env

env = DummyVecEnv([make_env])

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    batch_size=64,
    buffer_size=100_000,
    learning_starts=5_000,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    target_update_interval=500,
)

model.learn(total_timesteps=TOTAL_TIMESTEPS)

model.save(os.path.join(AGENT_NAME, "model"))

print("DQN training complete!")