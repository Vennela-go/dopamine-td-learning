# Dopamine and Temporal-Difference Learning

A small, reproducible Python project that models the classic reward-prediction-error findings associated with dopamine activity.

## Research question
Can a simple TD(0) learner reproduce the qualitative pattern in which prediction error moves from unexpected reward delivery to the reward-predicting cue, and becomes negative when an expected reward is omitted?

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiments.py
```
Figures and `metrics.json` are saved in `results/`.

## Algorithm
TD(0) updates the value of the current state:
`delta = reward + gamma * V(next_state) - V(state)`
`V(state) = V(state) + alpha * delta`

The simulated sequence is baseline → cue → delay 1 → delay 2 → reward. A reward of 1 is delivered at the final transition. The omission test uses the learned values but withholds reward.

## Baseline
The heuristic baseline is a no-learning predictor that always predicts zero reward. Its reward-prediction RMSE remains 1 for this unit-reward task. TD(0) should reduce prediction error as cue value is learned.

## Expected qualitative findings
1. Early trials: large positive TD error at reward delivery.
2. Learning: cue-onset TD error increases while reward-delivery error decreases.
3. After learning: reward delivery is largely predicted, so its TD error approaches zero.
4. Omission: withholding the expected reward produces a negative TD error.

## Important scientific caveat
This is a simplified computational demonstration, not a biological model of dopamine firing and not a replication of the original animal experiment. The correspondence between dopamine activity and TD error is influential but does not imply that all dopamine neurons encode one identical scalar signal. The simulation uses a deterministic chain, fixed learning rate, and a single reward.

## Deliverables
- `problem_formulation.md`: one-page state/action/reward/environment formulation.
- `src/run_experiments.py`: runnable TD(0) implementation.
- `results/`: learning curves, baseline comparison, omission test, and metrics.
- `presentation/`: five-slide presentation.
