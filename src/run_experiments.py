"""
Dopamine and Temporal-Difference Learning
Reproducible computational reproduction of reward-prediction-error experiments.

Run from the project root:
    python src/run_experiments.py
Outputs are written to results/.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SEED = 7
ALPHA = 0.15
GAMMA = 0.95
N_TRIALS = 100
STATES = ["baseline", "cue", "delay_1", "delay_2", "reward"]

def td_episode(values, reward=1.0, alpha=ALPHA, gamma=GAMMA, omit=False):
    """Run one baseline->cue->delay1->delay2->reward sequence using TD(0)."""
    deltas = []
    transitions = list(zip(STATES[:-1], STATES[1:]))
    for s, nxt in transitions:
        r = 0.0
        if s == "delay_2":
            r = 0.0 if omit else reward
        target = r + gamma * values[nxt]
        delta = target - values[s]
        if s != "baseline":
            values[s] += alpha * delta
        deltas.append(delta)
    return deltas

def train():
    values = {s: 0.0 for s in STATES}
    cue_errors, reward_errors, value_curve, rmse_curve = [], [], [], []
    baseline_rmse = []
    for trial in range(N_TRIALS):
        errors = td_episode(values)
        cue_errors.append(errors[0])
        reward_errors.append(errors[-1])
        value_curve.append(values["cue"])
        # Prediction at cue is the cue state's learned value; target is 1.
        rmse_curve.append((1.0 - values["cue"]) ** 2)
        # Heuristic baseline: never learns and always predicts zero reward.
        baseline_rmse.append(1.0)
    return values, np.array(cue_errors), np.array(reward_errors), np.array(value_curve), np.array(rmse_curve), np.array(baseline_rmse)

def make_figures():
    results = Path("results")
    results.mkdir(exist_ok=True)
    values, cue, reward, value, mse, base = train()
    trials = np.arange(1, N_TRIALS + 1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(trials, value, label="TD(0): predicted value of cue")
    plt.axhline(1.0, linestyle="--", label="Actual reward value")
    plt.xlabel("Trial"); plt.ylabel("Predicted value")
    plt.title("Learning curve: cue value increases with experience")
    plt.legend(); plt.tight_layout()
    plt.savefig(results/"learning_curve.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(trials, cue, label="TD error at cue onset")
    plt.plot(trials, reward, label="TD error at reward delivery")
    plt.axhline(0, color="black", linewidth=.7)
    plt.xlabel("Trial"); plt.ylabel("TD error (prediction error)")
    plt.title("Temporal shift in reward-prediction error")
    plt.legend(); plt.tight_layout()
    plt.savefig(results/"td_error_curve.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(trials, np.sqrt(mse), label="TD(0) learner")
    plt.plot(trials, np.sqrt(base), "--", label="Heuristic baseline (always predicts 0)")
    plt.xlabel("Trial"); plt.ylabel("Reward prediction RMSE")
    plt.title("TD learner compared with a no-learning baseline")
    plt.legend(); plt.tight_layout()
    plt.savefig(results/"baseline_comparison.png", dpi=180); plt.close()

    # Omission test after learning: expected reward is withheld.
    omission_values = values.copy()
    omission_deltas = td_episode(omission_values, omit=True)
    plt.figure(figsize=(8, 4.5))
    labels = ["Cue onset", "Cue → delay 1", "Delay 1 → delay 2", "Expected reward omitted"]
    plt.bar(labels, omission_deltas)
    plt.axhline(0, color="black", linewidth=.7)
    plt.ylabel("TD error"); plt.title("Reward omission after learning")
    plt.tight_layout(); plt.savefig(results/"omission_test.png", dpi=180); plt.close()

    summary = {
        "trials": N_TRIALS, "alpha": ALPHA, "gamma": GAMMA,
        "final_cue_value": float(value[-1]),
        "final_reward_delivery_td_error": float(reward[-1]),
        "final_cue_td_error": float(cue[-1]),
        "final_td_rmse": float(np.sqrt(mse[-1])),
        "baseline_rmse": float(np.sqrt(base[-1])),
        "omission_td_errors": [float(x) for x in omission_deltas]
    }
    (results/"metrics.json").write_text(__import__("json").dumps(summary, indent=2))
    print(__import__("json").dumps(summary, indent=2))

if __name__ == "__main__":
    make_figures()
