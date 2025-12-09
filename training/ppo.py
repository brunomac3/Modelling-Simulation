from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import highway_env
import os

# Training configuration
TOTAL_TIMESTEPS = 200_000  # Balanced: ~20-30 min training, decent performance
AGENT_NAME = "ppo_agent"
os.makedirs(AGENT_NAME, exist_ok=True)

def make_env():
    env = gym.make("highway-v0")
    env.unwrapped.config.update({
        # Observation
        "observation": {"type": "Kinematics"},
        
        # Traffic density
        "vehicles_count": 50,  # More traffic = more realistic + harder
        "duration": 60,  # Shorter episodes = faster learning cycles
        
        # Simulation parameters
        "simulation_frequency": 15,
        "policy_frequency": 5,
        
        # Reward shaping (balanced for safety + efficiency)
        "reward_speed_range": [20, 30],
        "collision_penalty": -10,  # Strong penalty for crashes
        "right_lane_reward": 0.1,
        "lane_change_reward": -0.05,  # Discourage excessive lane changes
        "high_speed_reward": 0.3,  # Moderate speed reward (not too aggressive)
    })
    return env

# Vectorized + normalization
env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# === Improved PPO hyperparameters ===
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,  # Show training progress
    n_steps=2048,
    batch_size=256,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=dict(
        net_arch=[256, 256]  # Simplified network for faster training
    ),
)

# Train
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Save
model.save(os.path.join(AGENT_NAME, "model"))
env.save(os.path.join(AGENT_NAME, "vec_normalize.pkl"))

print("Training complete!")