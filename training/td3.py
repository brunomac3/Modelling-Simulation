"""
TD3 Training - Continuous Action Space (Baseline)

TD3 is a strong continuous-control baseline that can be more stable than SAC
with similar performance when properly tuned.
"""

from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import gymnasium
import highway_env
import numpy as np
import os
import sys

# Import shared environment config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env_config import get_continuous_env_config
from reward_wrappers import LaneCenteringOvertakeReward

# Training configuration
AGENT_NAME = "td3_agent"
TOTAL_TIMESTEPS = 400_000
os.makedirs(AGENT_NAME, exist_ok=True)

print("=" * 70)
print("🚗 TD3 Training - Continuous Action Space")
print("=" * 70)

# Create environment and configure for continuous actions
env = gymnasium.make("highway-v0")
config = get_continuous_env_config()
# Slightly reduce speed pressure for TD3 to avoid aggressive passes
config["high_speed_reward"] = 0.5
config["reward_speed_range"] = [22, 32]
config["collision_reward"] = -6
env.unwrapped.config.update(config)
env = LaneCenteringOvertakeReward(
    env,
    overtake_reward=0.1,
    lane_change_penalty_weight=0.1,
)
env.reset()

print("\nEnvironment Configuration:")
print(f"  - Action Space: {env.action_space}")
print(f"  - Action Type: {type(env.action_space).__name__}")
print(f"  - Vehicles: {env.unwrapped.config['vehicles_count']}")
print(f"  - Duration: {env.unwrapped.config['duration']}s")
print(f"  - Policy Frequency: {env.unwrapped.config['policy_frequency']} Hz")

# Verify continuous action space
if not hasattr(env.action_space, "shape"):
    raise ValueError(
        f"TD3 requires continuous (Box) action space, got {type(env.action_space)}"
    )
print("✅ Continuous action space verified!")

print(f"\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...")
print("This can take a while; check your machine speed for ETA...")
print("=" * 70)

# Action noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# TD3 hyperparameters for highway-env
model = TD3(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    policy_delay=2,
    action_noise=action_noise,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log=f"{AGENT_NAME}/tensorboard/",
)

model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(os.path.join(AGENT_NAME, "model"))

print("\n" + "=" * 70)
print("✅ TD3 training complete!")
print(f"📁 Model saved to: {AGENT_NAME}/model.zip")
print("=" * 70)
