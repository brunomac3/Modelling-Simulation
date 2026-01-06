"""
Shared environment configuration for training and evaluation.
Based on official highway-env documentation baseline.

This ensures the agent is tested in the same environment it was trained in.
Source: https://highway-env.farama.org/quickstart/
"""

import numpy as np

# Official highway-v0 default configuration (from documentation)
# This is the EXACT config used in the official examples
ENV_CONFIG = {
    # Observation - Kinematics (default)
    "observation": {
        "type": "Kinematics"
    },
    
    # Traffic - moderate density for smooth, visible driving
    "vehicles_count": 20,        # Fewer cars = easier to see what's happening
    "vehicles_density": 1,
    
    # Episode duration - 40 seconds (documentation default)
    "duration": 40,
    
    # Simulation timing (documentation defaults)
    "simulation_frequency": 15,  # 15 Hz physics simulation
    "policy_frequency": 1,       # 1 Hz decision making
    
    # Rendering - disabled for smooth, fast visualization
    "real_time_rendering": False,  # False = render as fast as possible (smooth!)
    
    # Rewards (documentation defaults)
    "collision_reward": -1,        # -1 for collision (NOT -10)
    "right_lane_reward": 0.1,      # Reward for right lane
    "high_speed_reward": 0.4,      # 0.4 for high speed (NOT 0.3)
    "reward_speed_range": [20, 30],  # Speed range for rewards
    "lane_change_reward": 0,       # 0 (neutral, NOT penalized)
    "normalize_reward": True,      # Normalize rewards
    
    # Terminal conditions
    "offroad_terminal": False,
    
    # Control
    "controlled_vehicles": 1,
    "manual_control": False,
}

def get_env_config():
    """Returns a copy of the environment configuration."""
    return ENV_CONFIG.copy()

def get_continuous_env_config():
    """Returns environment config for continuous action space (for SAC)."""
    config = ENV_CONFIG.copy()
    # For highway-env, the action config must be a dict with 'type' key
    config["action"] = {
        "type": "ContinuousAction",
        "acceleration_range": [-5.0, 5.0],
        "steering_range": [-0.5, 0.5],  # More conservative steering
    }
    # Safety constraints
    config["offroad_terminal"] = True  # STOP if going off-road
    config["collision_reward"] = -10  # Stronger penalty for crashes
    config["normalize_reward"] = False  # Don't normalize to see real penalties
    config["policy_frequency"] = 2  # Smoother continuous control for SAC
    return config
