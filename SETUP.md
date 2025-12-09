# Highway-Env Setup

## 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install stable-baselines3[extra] highway-env jupyterlab matplotlib pandas
```

## 3. Train Models (Python Scripts)

Train the reinforcement learning agents (run from project root):

```bash
# Train DQN agent (~15-20 minutes, 150k timesteps)
python training/dqn.py

# Train PPO agent (~20-30 minutes, 200k timesteps)
python training/ppo.py

# Train SAC agent (~25-35 minutes, 250k timesteps)
python training/sac.py
```

Each script will create a folder (`ppo_agent/`, `dqn_agent/`, `sac_agent/`) containing the trained model.

**Note**: These are balanced configs for quick iteration. If you need better performance, you can increase `TOTAL_TIMESTEPS` in each file (e.g., 500k+ for production models).

## 4. Evaluate Models (Jupyter Notebook)

```bash
jupyter-lab
```

Open `notebook1.ipynb` and:
1. Set `AGENT_TYPE = "ppo"` (or "dqn", "sac", "random")
2. Run the evaluation cells to simulate episodes and calculate performance indicators
3. After testing all models, set `RUN_COMPARISON = True` to compare them side-by-side