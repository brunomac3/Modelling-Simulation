from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import highway_env
import os

AGENT_NAME = "sac_agent"
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
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = SAC(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,
    tau=0.02,
    train_freq=64,
    gradient_steps=64,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
)

model.learn(total_timesteps=TOTAL_TIMESTEPS)

model.save(os.path.join(AGENT_NAME, "model"))
env.save(os.path.join(AGENT_NAME, "vec_normalize.pkl"))

print("SAC training complete!")