# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
AIREN Rubrics — OpenEnv RFC 004 compliant reward rubrics.

Rubrics decouple reward logic from environment logic, making it easy to
swap reward functions for different training objectives without changing
the environment itself.

Usage:
    from airen_env.rubrics import AIRENRubric, ResolutionRubric, DiagnosisRubric

    rubric = AIRENRubric()                    # default composite
    reward = rubric(obs, action, prev_obs)

    rubric = AIRENRubric(outcome=ResolutionRubric(), process=DiagnosisRubric())
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class RubricResult:
    reward: float
    outcome_reward: float
    process_reward: float
    explanation: str


class ResolutionRubric:
    """
    Outcome rubric: reward based on whether the incident was resolved.

    Terminal steps only (done=True):
      - 1.0 if incident resolved in ≤ 3 steps
      - 0.8 if resolved in ≤ 5 steps
      - 0.6 if resolved at any step
      - 0.0 if not resolved (timeout)

    Non-terminal steps: 0.0
    """

    def __call__(self, obs, action=None, prev_obs=None) -> float:
        if not getattr(obs, "done", False):
            return 0.0
        if not getattr(obs, "incident_resolved", False):
            return 0.0
        steps = getattr(obs, "step_number", 10)
        if steps <= 3:
            return 1.0
        if steps <= 5:
            return 0.8
        return 0.6


class DiagnosisRubric:
    """
    Process rubric: per-step reward for correct diagnostic behaviour.

      +0.2  for any diagnostic action (inspect_logs, run_diagnostic, inspect_metrics)
      +0.3  for diagnostic action on the correct service
      -0.1  for destructive action on wrong service (hallucination penalty)
      -0.05 for code execution errors / action failures
       0.0  otherwise
    """

    _DIAGNOSTIC = {"inspect_logs", "inspect_metrics", "run_diagnostic", "acknowledge_incident"}

    def __call__(self, obs, action=None, prev_obs=None) -> float:
        if action is None:
            return 0.0
        action_type = getattr(action, "action_type", "")
        target      = getattr(action, "target", "")
        success     = getattr(obs, "action_success", True)

        if not success and action_type not in self._DIAGNOSTIC:
            return -0.05

        diagnosis_score = getattr(obs, "diagnosis_score", None)
        if diagnosis_score is not None:
            # Use environment's own diagnosis score as process signal
            return round(diagnosis_score * 0.3, 3)

        if action_type in self._DIAGNOSTIC:
            return 0.2
        return 0.0


class HealthDeltaRubric:
    """
    Process rubric: reward proportional to system health improvement this step.
    Provides immediate dense signal — agent learns that health matters every step.
    """

    def __call__(self, obs, action=None, prev_obs=None) -> float:
        health_now  = getattr(obs, "system_health", 0.0)
        health_prev = getattr(prev_obs, "system_health", health_now) if prev_obs else health_now
        delta = health_now - health_prev
        return round(max(-0.5, min(1.0, delta * 3.0)), 3)


class CustomMetricRubric:
    """
    Rubric backed by a user-provided metric function.

    Args:
        metric_fn: Callable(obs, action, prev_obs) -> float
                   Must return a value in [-1.0, 1.0].

    Example:
        def my_metric(obs, action, prev_obs):
            return 1.0 if obs.incident_resolved else 0.0

        rubric = AIRENRubric(outcome=CustomMetricRubric(my_metric))
    """

    def __init__(self, metric_fn: Callable):
        self._fn = metric_fn

    def __call__(self, obs, action=None, prev_obs=None) -> float:
        return float(self._fn(obs, action, prev_obs))


class AIRENRubric:
    """
    Composite rubric combining outcome + process rewards.
    Default rubric used by the environment.

    Follows OpenEnv RFC 004:
      - outcome_reward: terminal signal (resolution quality)
      - process_reward: per-step signal (diagnosis quality, health delta)
      - failure_reward: penalty when max steps exhausted without resolution

    Args:
        outcome:         Rubric for terminal reward. Default: ResolutionRubric.
        process:         Rubric for per-step reward. Default: DiagnosisRubric.
        outcome_weight:  Weight for outcome component. Default: 0.6.
        process_weight:  Weight for process component. Default: 0.4.
        failure_reward:  Reward when episode times out. Default: -0.1.
        gamma:           Discount factor for temporal credit assignment. Default: 0.99.
    """

    def __init__(
        self,
        outcome: Optional[object] = None,
        process: Optional[object] = None,
        outcome_weight: float = 0.6,
        process_weight: float = 0.4,
        failure_reward: float = -0.1,
        gamma: float = 0.99,
    ):
        self.outcome        = outcome or ResolutionRubric()
        self.process        = process or DiagnosisRubric()
        self.outcome_weight = outcome_weight
        self.process_weight = process_weight
        self.failure_reward = failure_reward
        self.gamma          = gamma

    def __call__(self, obs, action=None, prev_obs=None) -> RubricResult:
        done = getattr(obs, "done", False)
        step = getattr(obs, "step_number", 0)
        max_steps = getattr(obs, "max_steps", 10)

        # Failure: timed out without resolution
        if done and not getattr(obs, "incident_resolved", False):
            return RubricResult(
                reward=self.failure_reward,
                outcome_reward=self.failure_reward,
                process_reward=0.0,
                explanation=f"timeout after {step} steps → {self.failure_reward}",
            )

        outcome_r = self.outcome(obs, action, prev_obs)
        process_r = self.process(obs, action, prev_obs)

        # Apply temporal discount to outcome reward
        if done and outcome_r > 0:
            discount = self.gamma ** step
            outcome_r = round(outcome_r * discount, 4)

        total = round(
            self.outcome_weight * outcome_r + self.process_weight * process_r, 4
        )

        return RubricResult(
            reward=total,
            outcome_reward=outcome_r,
            process_reward=process_r,
            explanation=(
                f"outcome={outcome_r:.3f}×{self.outcome_weight} + "
                f"process={process_r:.3f}×{self.process_weight} "
                f"→ {total:.4f}"
            ),
        )
