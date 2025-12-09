from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import highway_env
import os

# Training configuration
AGENT_NAME = "sac_agent"
TOTAL_TIMESTEPS = 250_000  # Balanced: ~25-35 min training, decent performance
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

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,  # Show training progress
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,
    tau=0.02,
    train_freq=64,
    gradient_steps=64,
    policy_kwargs=dict(net_arch=[256, 256]),  # Simplified network
)

model.learn(total_timesteps=TOTAL_TIMESTEPS)

model.save(os.path.join(AGENT_NAME, "model"))
env.save(os.path.join(AGENT_NAME, "vec_normalize.pkl"))

print("SAC training complete!")