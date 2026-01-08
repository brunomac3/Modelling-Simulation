# Training and Testing Guide

## 🎯 Goal
Train all 3 models (DQN, PPO, SAC) with the same timesteps (150k) and compare their performance.

---

## 📋 Complete Workflow

### Step 1: Train All Models

Run these commands **one at a time** (each takes ~30-45 minutes):

```bash
# Train DQN (150k timesteps)
python training/dqn.py

# Train PPO (150k timesteps)  
python training/ppo.py

# Train SAC (150k timesteps)
python training/sac.py
```

**What happens:**
- Each model trains for 150,000 timesteps
- Progress is shown in the terminal
- Model is saved to `{model}_agent/model.zip`
- Training takes ~30-45 minutes per model

**You can train them in any order!**

---

### Step 2: Evaluate Each Model

Open `highway_gym.ipynb` in Jupyter Lab:

```bash
jupyter-lab
```

Then for **each model**:

1. **Change `AGENT_TYPE` in Cell 1** to the model you want to test:
   ```python
   AGENT_TYPE = "dqn"   # or "ppo" or "sac"
   ```

2. **Run Cell 1** (PART 1: Evaluation)
   - Simulation window opens
   - Watch your agent drive for 20 episodes
   - Results automatically saved

3. **Run Cell 2** (PART 2: Indicators)
   - See performance breakdown (SI, EI, CI, RCI, GPS)
   - Understand strengths/weaknesses

4. **Repeat for other models**
   - Change `AGENT_TYPE` to `"ppo"`
   - Run Cell 1 and Cell 2 again
   - Change to `"sac"`
   - Run Cell 1 and Cell 2 again

---

## ✅ Practical CLI Evaluation (Recommended)

For reproducible metrics, use the evaluation script instead of the notebook:

```bash
source venv/bin/activate
python scripts/eval.py --agent dqn
python scripts/eval.py --agent ppo
python scripts/eval.py --agent sac
python scripts/eval.py --agent td3
```

Notes:
- The script automatically loads PPO's `VecNormalize`.
- SAC/TD3 use the same reward shaping as training.
- Results are saved to `{agent}_agent/instant_runs/` and `{agent}_agent/summary/`.

### Step 3: Compare All Models

After evaluating all 3 models:

1. **Run Cell 4** (PART 4: Multi-Model Comparison)
   - Automatically loads latest results for each model
   - Shows side-by-side comparison table
   - Displays winner for each category
   - Creates visualization chart

---

## 📊 What to Look For

### Performance Indicators

- **GPS (Global Performance Score)**: Overall winner
- **Safety Index (SI)**: Which model crashes least?
- **Efficiency Index (EI)**: Which model drives fastest + completes episodes?
- **Comfort Index (CI)**: Which model drives smoothest?
- **Rule Compliance (RCI)**: Which follows speed limits best?

### Expected Results

Based on highway-env research:
- **SAC**: Usually best overall (highest GPS)
- **PPO**: Good balance, stable learning
- **DQN**: Can struggle with continuous control

---

## ⏱️ Time Estimate

- **Training all 3 models**: 90-135 minutes total (can run overnight)
- **Evaluating each model**: 5-10 minutes per model
- **Total workflow**: ~2-3 hours

---

## 🔧 Troubleshooting

**Model not found?**
```
ERROR: Can't find dqn_agent/model.zip
```
→ Train that model first with `python training/dqn.py`

**Simulation too fast/slow?**
→ Edit `env_config.py` and change `real_time_rendering` (False = fast, True = slow-motion)

**Want to retrain a model?**
→ Just run the training script again - it will overwrite the old model

---

## 🎓 Scientific Method

This workflow follows proper ML research methodology:

1. **Fair Comparison**: All models trained with same timesteps
2. **Consistent Environment**: Same config for training & testing
3. **Statistical Validity**: 20 episodes per evaluation
4. **Multiple Metrics**: GPS + 4 sub-indices for comprehensive analysis
5. **Reproducibility**: Shared config ensures anyone can replicate results

---

## 💡 Next Steps After Comparison

Once you know which model wins:

1. **Increase training time** for winner (300k-500k timesteps)
2. **Tune hyperparameters** (learning rate, network size, etc.)
3. **Analyze failure cases** (when does it crash? why?)
4. **Document findings** in your project report

Good luck! 🚀
