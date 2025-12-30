"""
DQN Training - Based on Official highway-env Documentation
Source: https://highway-env.farama.org/quickstart/

This uses the exact same configuration as the official example
to ensure consistent, quality baseline performance.
"""

from stable_baselines3 import DQN
import gymnasium
import highway_env
import os
import sys

# Import shared environment config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env_config import get_env_config

# Training configuration
AGENT_NAME = "dqn_agent"
TOTAL_TIMESTEPS = 150_000  # 150k timesteps for fair comparison
os.makedirs(AGENT_NAME, exist_ok=True)

print("="*70)
print("🚗 DQN Training - Official highway-env Baseline")
print("="*70)

# Create environment with shared config
env = gymnasium.make("highway-fast-v0")
env.unwrapped.config.update(get_env_config())

print(f"\nEnvironment Configuration:")
print(f"  - Vehicles: {env.unwrapped.config['vehicles_count']}")
print(f"  - Duration: {env.unwrapped.config['duration']}s")
print(f"  - Policy Frequency: {env.unwrapped.config['policy_frequency']} Hz")
print(f"  - Collision Penalty: {env.unwrapped.config['collision_reward']}")
print(f"  - High Speed Reward: {env.unwrapped.config['high_speed_reward']}")
print(f"\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...")
print("="*70)

# Official DQN configuration (from documentation)
model = DQN(
    'MlpPolicy',
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=15000,
    learning_starts=200,
    batch_size=32,
    gamma=0.8,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=50,
    verbose=1,
    tensorboard_log=f"{AGENT_NAME}/tensorboard/"
)

# Train
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Save
model.save(os.path.join(AGENT_NAME, "model"))

print("\n" + "="*70)
print("✅ DQN training complete!")
print(f"📁 Model saved to: {AGENT_NAME}/model.zip")
print("="*70)