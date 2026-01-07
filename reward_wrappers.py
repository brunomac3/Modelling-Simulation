"""
Reward shaping wrappers for highway-env.
"""

from __future__ import annotations

import gymnasium as gym


class LaneCenteringOvertakeReward(gym.Wrapper):
    """
    Add lane-centering penalty and overtake bonus to the base reward.

    - Lane-centering penalty encourages staying near the lane center.
    - Overtake bonus rewards clean passes when the ego goes ahead of another car.
    """

    def __init__(
        self,
        env: gym.Env,
        lane_center_weight: float = 0.2,
        overtake_reward: float = 0.2,
        steering_penalty_weight: float = 0.05,
    ) -> None:
        super().__init__(env)
        self.lane_center_weight = lane_center_weight
        self.overtake_reward = overtake_reward
        self.steering_penalty_weight = steering_penalty_weight
        self._last_rel_x = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_rel_x = {}
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        ego = self.env.unwrapped.vehicle
        road = self.env.unwrapped.road

        lane_penalty = 0.0
        if ego.lane_index is not None:
            lane = road.network.get_lane(ego.lane_index)
            longitudinal, lateral = lane.local_coordinates(ego.position)
            width = lane.width_at(longitudinal)
            if width > 0:
                # Normalize lateral deviation by half-lane width.
                lateral_ratio = abs(lateral) / (width / 2.0)
                lane_penalty = -self.lane_center_weight * lateral_ratio

        overtake_bonus = 0.0
        for other in road.vehicles:
            if other is ego:
                continue
            key = id(other)
            rel_x = other.position[0] - ego.position[0]
            prev_rel_x = self._last_rel_x.get(key)
            if prev_rel_x is not None and prev_rel_x > 0 and rel_x <= 0:
                overtake_bonus += self.overtake_reward
            self._last_rel_x[key] = rel_x

        steering_penalty = 0.0
        if hasattr(self.env.unwrapped, "action_type"):
            last_action = getattr(self.env.unwrapped.action_type, "last_action", None)
            if last_action is not None and len(last_action) > 1:
                steering_penalty = -self.steering_penalty_weight * abs(last_action[1])

        reward += lane_penalty + overtake_bonus + steering_penalty
        info["lane_center_penalty"] = lane_penalty
        info["overtake_bonus"] = overtake_bonus
        info["steering_penalty"] = steering_penalty
        return obs, reward, done, truncated, info
