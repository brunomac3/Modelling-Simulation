from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import highway_env
import os

# Training configuration
AGENT_NAME = "dqn_agent"
TOTAL_TIMESTEPS = 20_000  # Reccomended for DQN: 20_000
os.makedirs(AGENT_NAME, exist_ok=True)

def make_env():
    env = gym.make("highway-v0")
    """ env.unwrapped.config.update({
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
    }) """

    # opted to take off custom configuration to compare default DQN performance
    return env

env = DummyVecEnv([make_env])

model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1
            )

model.learn(total_timesteps=TOTAL_TIMESTEPS)

model.save(os.path.join(AGENT_NAME, "model"))

print("DQN training complete!")