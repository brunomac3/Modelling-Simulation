# Evaluation Analysis (Mean ± Std)

Results below are from the latest multi-run evaluation (3 runs × 50 episodes).
Tables were generated from `results_mean.csv` and `results_std.csv`.

## Summary

- **Fastest**: SAC/TD3 (avg_speed ~31.55–31.59 m/s).
- **Safest**: PPO (collision 0.0, success 1.0), but slowest.
- **Most aggressive**: DQN (highest crash rate, low success).
- **Best balance** (speed + safety): TD3 currently looks strongest (fast with low collisions).

## Key Metrics (Mean ± Std)

**DQN**
- avg_speed_ms: 27.70 ± 0.30
- collision: 0.46 ± 0.03
- success: 0.54 ± 0.03
- lane_changes: 10.20 ± 0.85
- avg_jerk: 4.54 ± 0.51

**PPO**
- avg_speed_ms: 20.75 ± 0.02
- collision: 0.00 ± 0.00
- success: 1.00 ± 0.00
- lane_changes: 3.09 ± 0.30
- avg_jerk: 1.67 ± 0.04

**SAC**
- avg_speed_ms: 31.55 ± 0.12
- collision: 0.23 ± 0.03
- success: 0.77 ± 0.03
- lane_changes: 12.44 ± 1.58
- avg_jerk: 1.84 ± 0.23

**TD3**
- avg_speed_ms: 31.59 ± 0.11
- collision: 0.10 ± 0.05
- success: 0.90 ± 0.05
- lane_changes: 12.29 ± 1.52
- avg_jerk: 2.23 ± 0.28

## Interpretation

- **PPO** is extremely stable but too conservative for overtaking-heavy scenarios. It keeps large TTC margins and rarely changes lanes, which is good for safety but limits efficiency.
- **DQN** is fast but unsafe (crashes ~46% of episodes). Its higher speed and frequent lane changes lead to aggressive behavior and more collisions.
- **SAC** is fast but still too risky (crash rate ~23%). It gains speed but has shorter TTC and more lane changes, indicating tight overtakes.
- **TD3** is fast and currently the best safety/speed trade-off. It matches SAC speed with fewer crashes and higher success.

Overall, the results show a clear safety–efficiency trade-off: PPO maximizes safety and comfort, while TD3 and SAC maximize speed, with TD3 providing the best balance.

## Recommendation

For presentation:
- Use **TD3** as the best overall performer.
- Use **PPO** to demonstrate safety/comfort baseline.
- Use **DQN** as the aggressive/unstable baseline.
