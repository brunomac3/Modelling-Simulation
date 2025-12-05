from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import highway_env
import os

TOTAL_TIMESTEPS = 500_000          # ↑ Increased
AGENT_NAME = "ppo_agent"
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
        "high_speed_reward": 0.6
    })
    return env

# Vectorized + normalization
env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# === Improved PPO hyperparameters ===
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,                 # ↑ much better with PPO
    batch_size=256,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=dict(
        net_arch=[256, 256, 128]  # ↑ bigger network
    ),
)

# Train
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Save
model.save(os.path.join(AGENT_NAME, "model"))
env.save(os.path.join(AGENT_NAME, "vec_normalize.pkl"))

print("Training complete!")