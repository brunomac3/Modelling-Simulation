# Reinforcement Learning for Autonomous Highway Navigation

A comparative study of discrete and continuous control RL algorithms (DQN, PPO, SAC, TD3) for autonomous highway driving using the Highway-Env simulation platform.

---

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install stable-baselines3[extra] highway-env jupyterlab matplotlib pandas
```

---

## Training the Models

Train reinforcement learning agents from the project root:

```bash
# DQN - Discrete actions (~15-20 min, 150k timesteps)
python training/dqn.py

# PPO - Discrete actions (~30-40 min, 400k timesteps)
python training/ppo.py

# SAC - Continuous actions (~30-40 min, 400k timesteps)
python training/sac.py

# TD3 - Continuous actions (~30-40 min, 400k timesteps)
python training/td3.py
```

Each script saves the trained model to its respective directory (`dqn_agent/`, `ppo_agent/`, `sac_agent/`, `td3_agent/`).

**Action Spaces:**
- **DQN & PPO**: Discrete actions (LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER)
- **SAC & TD3**: Continuous actions (steering angle + acceleration)

---

## Running Evaluation

### Option 1: Jupyter Notebook (Recommended for Visualization)

```bash
jupyter-lab
```

Open `model_evaluation.ipynb` and run the cells sequentially. The notebook provides:
- Episode simulation with visual rendering
- Performance indicator calculation (SI, EI, CI, RCI, GPS)
- Multi-model comparison and visualization

### Option 2: Command Line Evaluation

```bash
# Evaluate individual agents (20 episodes per run)
python scripts/eval.py --agent dqn --episodes 20
python scripts/eval.py --agent ppo --episodes 20
python scripts/eval.py --agent sac --episodes 20
python scripts/eval.py --agent td3 --episodes 20

# With rendering enabled
python scripts/eval.py --agent td3 --episodes 5 --render

# Multiple runs for statistical analysis
python scripts/eval.py --agent sac --episodes 20 --runs 3
```

### Aggregating Results and Plotting

When we created the plots with the scripts ```aggregate_results.py```, ```plot_results.py``` and 
```plot_indicators.py```, now placed in ```/scripts```, they were at the project root, so please consider the plots at the report (also at ```/final_results/plots```) as the definitive statistics for this project - we did not modify the scripts to handle this final codebase structure.


---

## Codebase Structure

```
├── training/                    # Model training scripts
│   ├── dqn.py                   # DQN training (discrete, 150k steps)
│   ├── ppo.py                   # PPO training (discrete, 400k steps)
│   ├── sac.py                   # SAC training (continuous, 400k steps)
│   └── td3.py                   # TD3 training (continuous, 400k steps)
│
├── scripts/                     # Evaluation and analysis utilities
│   ├── env_config.py            # Shared environment configuration
│   ├── reward_wrappers.py       # Custom reward shaping wrappers
│   ├── eval.py                  # CLI evaluation script
│   ├── aggregate_results.py     # Aggregate per-run summaries into mean/std
│   ├── plot_results.py          # Plot raw metrics comparison
│   ├── plot_indicators.py       # Plot performance indices (SI, EI, CI, GPS)
│   └── run_all_evals.py         # Batch evaluation runner
│
├── model_evaluation.ipynb       # Interactive evaluation notebook
│
├── dqn_agent/                   # DQN model and evaluation results
│   ├── model.zip                # Trained DQN model
│   ├── instant_runs/            # Per-episode CSV logs
│   └── summary/                 # Per-run summary statistics
│
├── ppo_agent/                   # PPO model and evaluation results
│   ├── model.zip
│   ├── vec_normalize.pkl        # Observation normalization stats
│   ├── instant_runs/
│   └── summary/
│
├── sac_agent/                   # SAC model and evaluation results
│   ├── model.zip
│   ├── instant_runs/
│   ├── summary/
│   └── tensorboard/             # Training logs
│
├── td3_agent/                   # TD3 model and evaluation results
│   ├── model.zip
│   ├── instant_runs/
│   ├── summary/
│   └── tensorboard/
│
├── final_results/               # Aggregated results and plots
│   ├── data/                    # CSV files (mean, std, indicators)
│   └── plots/                   # Generated comparison plots
│
├── model_comparison/            # Cross-model comparison CSVs
│
└── dev_docs/                    # Development documentation
```

---