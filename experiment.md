# Experiment Log (Summary)

This is a concise record of the changes made and why, in the order they happened.

1) Initial observation
- SAC behavior was unstable and did not match visual expectations.
- Evaluation metrics were inconsistent with what was seen on screen.

2) Environment and training consistency
- Switched SAC training to `highway-v0` and matched eval environment.
- Increased SAC steps from 2k to 200k, later to 400k, to stabilize learning.
- Set continuous control `policy_frequency = 2` for smoother control.

Result: SAC improved but still drifted and did not center in lane.

3) Fix evaluation correctness
- Metrics were using normalized observations (incorrect speeds/TTC).
- Updated evaluation to use real env state (vehicle speed, lane index).
- Fixed success definition (time-limit is not failure).
- Added a CLI evaluator (`scripts/eval.py`) to ensure reproducibility.
- Added PPO VecNormalize loading in CLI and notebook evaluation.
- Adjusted notebook PPO stepping to handle vec-env shapes.

Result: metrics now reflect actual behavior and match visuals.

4) Continuous-control shaping (SAC/TD3)
- Added `reward_wrappers.py` with:
  - lane-centering penalty
  - overtake bonus
  - steering-magnitude penalty
- Applied the wrapper to SAC and TD3 training/evaluation.
- Tightened steering range to reduce drifting.
- Adjusted SAC reward balance:
  - less severe collision penalty
  - reduced right-lane bias
  - stronger high-speed reward and higher target speed range

Result: SAC/TD3 became faster and more assertive, but comfort still needed tuning.

5) TD3 baseline
- Added `training/td3.py` (continuous baseline).
- Wired TD3 into evaluation.

Result: TD3 reached high speed and avoided crashes, but was still twitchy.

6) PPO improvements
- Added reward wrapper for PPO to encourage lane centering and overtakes.
- Reduced PPO right-lane bias.
- Increased PPO training to 400k steps to adapt to new shaping.

Result: PPO expected to become less passive after retraining.

7) Additional comfort tuning
- Tightened continuous steering range to [-0.2, 0.2].
- Increased lane-centering penalty to 0.3.
- Increased steering penalty to 0.08.

Result: Intended to reduce drifting and smooth transitions for TD3/SAC.

8) TD3 re-evaluation after pull + tuning
- Re-ran TD3 evaluation with the current config and reward shaping.
- Summary (2026-01-07): avg_speed ~31.97, lane_changes ~16.4, collision 0.2, success 0.8.

Result: TD3 stayed fast but still showed aggressive behavior (high lane changes, low TTC).

9) TD3 de-aggression tweaks
- Reduced overtake reward to 0.1 for TD3.
- Added lane-change penalty (0.1) to discourage multi-lane drift.
- Lowered high-speed pressure and raised collision penalty slightly.

Result: Expected to reduce unsafe overtakes and improve comfort after retraining.

10) PPO speed/overtake boost
- Increased PPO high-speed reward and target speed range.
- Increased PPO overtake reward.

Result: Intended to make PPO less passive and more willing to pass.

11) Evaluation snapshots (after fixes)
- DQN (2026-01-07): avg_speed ~28.0, lane_changes ~12.3, collision 0.40, success 0.60.
- PPO (2026-01-07): avg_speed ~20.4, lane_changes ~2.5, collision 0.00, success 1.00.
- SAC (2026-01-06 → 2026-01-07): avg_speed ~19.4 → ~31.3, collision 0.05 → 0.25.
- TD3 (2026-01-07): avg_speed ~31.9, lane_changes ~16.4, collision 0.20, success 0.80.

