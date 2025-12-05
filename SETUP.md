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
# Train PPO agent (~30-60 minutes, 500k timesteps)
python training/ppo.py

# Train DQN agent (~20-40 minutes, 400k timesteps)
python training/dqn.py

# Train SAC agent (~60-90 minutes, 600k timesteps)
python training/sqn.py
```

Each script will create a folder (`ppo_agent/`, `dqn_agent/`, `sac_agent/`) containing the trained model.

## 4. Evaluate Models (Jupyter Notebook)

```bash
jupyter-lab
```

Open `notebook1.ipynb` and:
1. Set `AGENT_TYPE = "ppo"` (or "dqn", "sac", "random")
2. Run the evaluation cells to simulate episodes and calculate performance indicators
3. After testing all models, set `RUN_COMPARISON = True` to compare them side-by-side