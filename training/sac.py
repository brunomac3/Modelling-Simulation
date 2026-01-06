"""
SAC Training - Continuous Action Space (Required for SAC)

SAC requires continuous actions. Uses ContinuousAction type with constrained ranges.
Note: This makes direct comparison with DQN/PPO harder, but showcases SAC's 
strengths in smooth, continuous control.
"""

from stable_baselines3 import SAC
import gymnasium
import highway_env
import os
import sys

# Import shared environment config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from env_config import get_env_config

# Training configuration
AGENT_NAME = "sac_agent"
TOTAL_TIMESTEPS = 2_000
os.makedirs(AGENT_NAME, exist_ok=True)

print("="*70)
print("🚗 SAC Training - Continuous Action Space")
print("="*70)

# Create environment and configure for continuous actions
env = gymnasium.make("highway-fast-v0")

# Get base config and add continuous action configuration
config = get_env_config()
config["action"] = {
    "type": "ContinuousAction"
}

# Apply configuration
env.unwrapped.config.update(config)
env.reset()

print(f"\nEnvironment Configuration:")
print(f"  - Action Space: {env.action_space}")
print(f"  - Action Type: {type(env.action_space).__name__}")
print(f"  - Vehicles: {env.unwrapped.config['vehicles_count']}")
print(f"  - Duration: {env.unwrapped.config['duration']}s")
print(f"  - Policy Frequency: {env.unwrapped.config['policy_frequency']} Hz")

# Verify continuous action space
if not hasattr(env.action_space, 'shape'):
    raise ValueError(f"SAC requires continuous (Box) action space, got {type(env.action_space)}")
print(f"✅ Continuous action space verified!")

print(f"\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...")
print("This will take ~30-40 minutes...")
print("="*70)

# SAC hyperparameters for highway-env
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=50000,
    learning_starts=200,  # Start training after collecting some experience
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log=f"{AGENT_NAME}/tensorboard/"
)

model.learn(total_timesteps=TOTAL_TIMESTEPS)

model.save(os.path.join(AGENT_NAME, "model"))

print("\n" + "="*70)
print("✅ SAC training complete!")
print(f"📁 Model saved to: {AGENT_NAME}/model.zip")
print("="*70)