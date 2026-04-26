# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Multi-objective reward for AIREN — TRUE RL-grade reward shaping.

Reward components (per step):
  +recovery      — health delta this step (immediate feedback)
  +diagnosis     — right service + right action type
  +efficiency    — acting early in the episode
  +threat_mitig  — reducing active threat level
  -hallucination — acting on wrong service with high confidence
  -security_viol — ignoring critical alerts
  -cost          — destructive actions that cause downtime
  -downtime      — cumulative time system was unhealthy
  +resolve_bonus — large bonus for fully resolving the incident

This is the difference between a benchmark harness and a real RL env.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RewardBreakdown:
    total: float
    recovery_score: float
    diagnosis_score: float
    efficiency_score: float
    threat_mitigation: float
    hallucination_penalty: float
    security_violation_penalty: float
    cost_penalty: float
    downtime_penalty: float
    resolve_bonus: float
    explanation: str


# Action cost table — destructive actions that cause real downtime
_ACTION_COSTS = {
    "restart_service":    0.30,
    "rollback_deployment": 0.20,
    "scale_service":      0.10,
    "ignore_alert":       0.50,   # highest — ignoring is always bad
}

# Actions that gather information (neutral, low cost)
_DIAGNOSTIC_ACTIONS = {
    "inspect_logs", "inspect_metrics", "run_diagnostic", "acknowledge_incident"
}


def compute_reward(
    action_type: str,
    target: str,
    action_success: bool,
    health_before: float,
    health_after: float,
    step_number: int,
    max_steps: int,
    correct_targets: List[str],
    correct_actions: List[str],
    incident_resolved: bool,
    total_downtime_steps: int,
    # New RL-grade inputs
    threat_level_before: float = 0.0,
    threat_level_after: float = 0.0,
    active_alerts_count: int = 0,
    wrong_actions_so_far: int = 0,
    severity: str = "high",
) -> RewardBreakdown:

    # ── 1. Recovery — immediate health delta ─────────────────────────────────
    delta = health_after - health_before
    recovery = max(0.0, min(1.0, delta * 3.0))   # amplified — health changes matter
    if incident_resolved:
        recovery = min(1.0, recovery + 0.4)

    # ── 2. Diagnosis — right service + right action ───────────────────────────
    correct_target = target in correct_targets
    correct_action = action_type in correct_actions
    if correct_target and correct_action:
        diagnosis = 1.0
    elif correct_target and action_type in _DIAGNOSTIC_ACTIONS:
        diagnosis = 0.5   # inspecting the right service is good
    elif correct_target:
        diagnosis = 0.4
    elif correct_action:
        diagnosis = 0.2
    else:
        diagnosis = 0.0

    # ── 3. Efficiency — reward acting early, penalise stalling ───────────────
    steps_remaining = max_steps - step_number
    efficiency = max(0.0, steps_remaining / max_steps)
    # Extra bonus for resolving quickly
    if incident_resolved and step_number <= 3:
        efficiency = min(1.0, efficiency + 0.3)

    # ── 4. Threat mitigation — did we reduce the threat level? ───────────────
    threat_delta = threat_level_before - threat_level_after
    threat_mitigation = max(0.0, min(1.0, threat_delta * 2.0))

    # ── 5. Hallucination penalty — wrong target with destructive action ───────
    hallucination = 0.0
    if not correct_target and action_type not in _DIAGNOSTIC_ACTIONS:
        # Agent acted destructively on the wrong service
        hallucination = 0.3 + (wrong_actions_so_far * 0.05)
        hallucination = min(0.6, hallucination)

    # ── 6. Security violation — ignoring critical alerts ─────────────────────
    security_viol = 0.0
    if action_type == "ignore_alert":
        security_viol = 0.4
        if severity == "critical":
            security_viol = 0.7
    elif active_alerts_count > 3 and action_type in _DIAGNOSTIC_ACTIONS and not correct_target:
        # Inspecting irrelevant service while critical alerts fire
        security_viol = 0.1

    # ── 7. Action cost — destructive wrong actions ────────────────────────────
    cost = 0.0
    if not action_success and action_type in _ACTION_COSTS:
        cost = _ACTION_COSTS[action_type]
    elif action_type == "ignore_alert":
        cost = _ACTION_COSTS["ignore_alert"]

    # ── 8. Downtime penalty — cumulative time unhealthy ───────────────────────
    downtime = min(0.5, total_downtime_steps * 0.05)

    # ── 9. Resolution bonus — big reward for actually fixing it ───────────────
    resolve_bonus = 0.0
    if incident_resolved:
        resolve_bonus = 0.5
        # Bigger bonus for fast resolution
        if step_number <= 3:
            resolve_bonus = 0.8
        elif step_number <= 5:
            resolve_bonus = 0.6

    # ── Weighted total ────────────────────────────────────────────────────────
    total = (
        0.25 * recovery
        + 0.20 * diagnosis
        + 0.10 * efficiency
        + 0.10 * threat_mitigation
        + 0.15 * resolve_bonus
        - 0.10 * hallucination
        - 0.10 * security_viol
        - 0.05 * cost
        - downtime
    )
    total = max(0.0, min(1.0, round(total, 3)))

    return RewardBreakdown(
        total=total,
        recovery_score=round(recovery, 3),
        diagnosis_score=round(diagnosis, 3),
        efficiency_score=round(efficiency, 3),
        threat_mitigation=round(threat_mitigation, 3),
        hallucination_penalty=round(hallucination, 3),
        security_violation_penalty=round(security_viol, 3),
        cost_penalty=round(cost, 3),
        downtime_penalty=round(downtime, 3),
        resolve_bonus=round(resolve_bonus, 3),
        explanation=(
            f"recovery={recovery:.2f}*0.25 "
            f"diagnosis={diagnosis:.2f}*0.20 "
            f"efficiency={efficiency:.2f}*0.10 "
            f"threat={threat_mitigation:.2f}*0.10 "
            f"resolve={resolve_bonus:.2f}*0.15 "
            f"halluc=-{hallucination:.2f}*0.10 "
            f"secviol=-{security_viol:.2f}*0.10 "
            f"cost=-{cost:.2f}*0.05 "
            f"downtime=-{downtime:.2f} "
            f"-> {total:.3f}"
        ),
    )
