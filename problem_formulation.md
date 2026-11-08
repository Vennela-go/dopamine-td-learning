# Problem Formulation: Dopamine and Temporal-Difference Learning

## Problem statement and motivation
The project investigates whether a standard temporal-difference learning algorithm can reproduce the qualitative reward-prediction-error (RPE) patterns reported in classic dopamine-neuron experiments. In the influential work of Schultz, Dayan and Montague (1997), dopamine responses were reported to be strong for unexpected rewards, to shift toward a reward-predicting cue after learning, and to decrease when an expected reward is omitted. The computational question is whether TD error provides a compact explanation for these patterns.

## Environment
A simulated, deterministic Pavlovian conditioning task. Each episode follows the state sequence:

`baseline → cue → delay_1 → delay_2 → reward`

A scalar reward of +1 is delivered on the final transition. In the omission condition, the expected reward is withheld. The environment is fully observable and resets to `baseline` after each episode.

## State
The state is the current stage in the conditioning sequence: `baseline`, `cue`, `delay_1`, `delay_2`, or `reward`. The state representation is discrete and tabular. The value function V(s) estimates expected discounted future reward from each stage.

## Action
There is no meaningful agent-controlled action in this Pavlovian conditioning setup: transitions are controlled by the environment. TD(0) is therefore used for state-value prediction rather than action selection. This is appropriate because the scientific focus is reward prediction error, not control. If a strict action-based RL extension is required, the same chain can be represented as a one-action MDP, with the sole action `wait/advance` at every non-terminal state.

## Reward
The agent receives +1 when the reward is delivered at the final transition and 0 otherwise. During the omission test, the reward is 0 at the expected delivery time. The discount factor is γ = 0.95.

## Learning algorithm
TD(0) updates the value estimate using:

`δ_t = r_(t+1) + γ V(s_(t+1)) − V(s_t)`

`V(s_t) ← V(s_t) + α δ_t`

The learning rate is α = 0.15. The TD error δ is treated as a computational analogue of reward-prediction error.

## Baseline and evaluation
The baseline is a heuristic no-learning predictor that always predicts zero reward. Its RMSE remains 1 on the unit-reward task. The TD learner is evaluated using (1) cue-value learning curve, (2) TD error at cue onset and reward delivery, (3) reward-prediction RMSE against the baseline, and (4) a post-learning reward-omission test. The desired outcome is increasing cue value, decreasing reward-time TD error, and a negative error when reward is omitted.

## Assumptions, limitations and ethics
This is a simplified deterministic simulation, not a biological measurement or exact reproduction of the original animal experiments. It omits neural heterogeneity, stochastic reward timing, physiological dynamics and experimental noise. It uses published findings as scientific context and does not involve human or animal subjects, personal data, or clinical claims. Results must be described as qualitative agreement with the TD account, not proof that dopamine is identical to TD error.
