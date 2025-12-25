"""
SAC Training - Based on Official highway-env Configuration

Uses the same environment config as DQN but with SAC-specific hyperparameters.
"""

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium
import highway_env
import os
import sys

# Import shared environment config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env_config import get_env_config

# Training configuration
AGENT_NAME = "sac_agent"
TOTAL_TIMESTEPS = 150_000  # SAC typically needs more steps
os.makedirs(AGENT_NAME, exist_ok=True)

print("="*70)
print("🚗 SAC Training - Official highway-env Baseline")
print("="*70)

def make_env():
    env = gymnasium.make("highway-v0")
    env.unwrapped.config.update(get_env_config())
    return env

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

print(f"\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...")
print("="*70)

# SAC hyperparameters (optimized for continuous control)
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,
    tau=0.02,
    train_freq=64,
    gradient_steps=64,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log=f"{AGENT_NAME}/tensorboard/"
)

model.learn(total_timesteps=TOTAL_TIMESTEPS)

model.save(os.path.join(AGENT_NAME, "model"))
env.save(os.path.join(AGENT_NAME, "vec_normalize.pkl"))

print("\n" + "="*70)
print("✅ SAC training complete!")
print(f"📁 Model saved to: {AGENT_NAME}/model.zip")
print(f"📁 Normalizer saved to: {AGENT_NAME}/vec_normalize.pkl")
print("="*70)