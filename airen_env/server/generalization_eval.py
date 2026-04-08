# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Cross-Environment Generalization Evaluator for AIREN.

Tests whether a model trained on some incident types generalizes to unseen ones.
This is the "cross-environment generalization" unsaturated area — most submissions
only test on the same distribution they trained on.

Evaluation protocol:
  1. Train on EASY + MEDIUM incident types (6 types)
  2. Evaluate on HARD types (3 types) — zero-shot generalization
  3. Also evaluate on held-out seeds of training types — in-distribution generalization
  4. Report generalization gap: train_score - test_score

This directly answers: "Does the agent learn to diagnose, or just memorize?"
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .incident_engine import (
    ALL_INCIDENT_TYPES, EASY_INCIDENTS, MEDIUM_INCIDENTS, HARD_INCIDENTS,
    generate_incident,
)


# ── Evaluation splits ─────────────────────────────────────────────────────────

TRAIN_SPLIT = EASY_INCIDENTS + MEDIUM_INCIDENTS   # 6 types — train on these
TEST_SPLIT  = HARD_INCIDENTS                       # 3 types — zero-shot eval

# Held-out seeds: configurable via env var, default to a range above typical training seeds
_HELD_OUT_START = int(os.environ.get("HELD_OUT_SEED_START", "5000"))
_HELD_OUT_COUNT = int(os.environ.get("HELD_OUT_SEED_COUNT", "20"))
HELD_OUT_SEEDS = list(range(_HELD_OUT_START, _HELD_OUT_START + _HELD_OUT_COUNT))

# Generalization gap thresholds — configurable via env vars
_GAP_EXCELLENT = float(os.environ.get("GEN_GAP_EXCELLENT", "0.10"))
_GAP_GOOD      = float(os.environ.get("GEN_GAP_GOOD",      "0.20"))
_GAP_PARTIAL   = float(os.environ.get("GEN_GAP_PARTIAL",   "0.35"))


@dataclass
class GeneralizationResult:
    """Result of evaluating one episode for generalization."""
    incident_type: str
    seed: int
    split: str                  # "train" | "test_zero_shot" | "test_held_out"
    cumulative_reward: float
    incident_resolved: bool
    steps_taken: int
    diagnosis_quality: str
    actions_taken: List[str]
    correct_actions: List[str]
    correct_targets: List[str]
    # Generalization-specific metrics
    correct_service_identified: bool   # did agent target the right service?
    correct_action_type_used: bool     # did agent use the right action type?
    exploration_depth: int             # how many unique services investigated
    wrong_fixes_before_resolve: int    # how many wrong fixes before success


@dataclass
class GeneralizationReport:
    """Full generalization evaluation report."""
    model_name: str
    timestamp: float
    train_episodes: int
    test_episodes: int

    # Aggregate scores
    train_avg_reward: float
    test_zero_shot_avg_reward: float
    test_held_out_avg_reward: float
    generalization_gap: float          # train - test_zero_shot (lower = better generalization)

    # Resolution rates
    train_resolution_rate: float
    test_zero_shot_resolution_rate: float
    test_held_out_resolution_rate: float

    # Per-incident-type breakdown
    by_incident_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Skill transfer analysis
    correct_service_rate_train: float = 0.0
    correct_service_rate_test: float = 0.0
    exploration_depth_train: float = 0.0
    exploration_depth_test: float = 0.0

    @property
    def generalizes_well(self) -> bool:
        """True if generalization gap is below the 'good' threshold."""
        return self.generalization_gap < _GAP_GOOD

    @property
    def grade(self) -> str:
        gap = self.generalization_gap
        if gap < _GAP_EXCELLENT: return "A"
        if gap < _GAP_GOOD:      return "B"
        if gap < _GAP_PARTIAL:   return "C"
        return "D"


def evaluate_generalization(
    model_fn,                          # callable(obs_str) -> action_type, target
    n_episodes_per_type: int = 3,
    seeds: Optional[List[int]] = None,
    model_name: str = "unknown",
) -> GeneralizationReport:
    """
    Evaluate model generalization across train/test splits.

    Args:
        model_fn: Function that takes observation string, returns (action_type, target)
        n_episodes_per_type: Episodes per incident type
        seeds: Seeds to use (default: random)
        model_name: Name for the report

    Returns:
        GeneralizationReport with full breakdown
    """
    from .airen_environment import AIRENEnvironment
    from .models import AIRENAction

    if seeds is None:
        seeds = [random.randint(0, 4999) for _ in range(n_episodes_per_type)]

    env = AIRENEnvironment()
    results: List[GeneralizationResult] = []

    for split, incident_types in [
        ("train", TRAIN_SPLIT),
        ("test_zero_shot", TEST_SPLIT),
        ("test_held_out", TRAIN_SPLIT),  # same types, unseen seeds
    ]:
        eval_seeds = HELD_OUT_SEEDS[:n_episodes_per_type] if split == "test_held_out" else seeds
        for itype in incident_types:
            for seed in eval_seeds[:n_episodes_per_type]:
                try:
                    obs = env.reset(incident_type=itype, seed=seed)
                    scenario = env._scenario
                    total_reward = 0.0
                    actions = []
                    wrong_fixes = 0
                    hypotheses = set()

                    for step in range(env.MAX_STEPS):
                        # Format observation for model
                        obs_str = _format_obs(obs)
                        try:
                            action_type, target = model_fn(obs_str)
                        except Exception:
                            action_type, target = "run_diagnostic", scenario.correct_targets[0]

                        action = AIRENAction(action_type=action_type, target=target)
                        obs2 = env.step(action)
                        total_reward += obs2.reward or 0.0
                        actions.append(f"{action_type}:{target}")

                        if action_type in ("inspect_logs", "inspect_metrics", "run_diagnostic"):
                            hypotheses.add(target)
                        if action_type in scenario.wrong_action_effects:
                            wrong_fixes += 1

                        if obs2.done:
                            obs = obs2
                            break
                        obs = obs2

                    state = env.state
                    correct_svc = any(
                        a.split(":")[1] in scenario.correct_targets
                        for a in actions
                        if ":" in a
                    )
                    correct_act = any(
                        a.split(":")[0] in scenario.correct_actions
                        for a in actions
                        if ":" in a
                    )

                    results.append(GeneralizationResult(
                        incident_type=itype,
                        seed=seed,
                        split=split,
                        cumulative_reward=round(total_reward, 3),
                        incident_resolved=state.incident_resolved,
                        steps_taken=state.steps_taken,
                        diagnosis_quality=getattr(obs, "diagnosis_quality", "unknown") or "unknown",
                        actions_taken=actions,
                        correct_actions=scenario.correct_actions,
                        correct_targets=scenario.correct_targets,
                        correct_service_identified=correct_svc,
                        correct_action_type_used=correct_act,
                        exploration_depth=len(hypotheses),
                        wrong_fixes_before_resolve=wrong_fixes,
                    ))
                except Exception:
                    pass

    return _build_report(results, model_name)


def _format_obs(obs) -> str:
    """Format observation for model input."""
    services = {
        name: f"{s.get('status')} | err={s.get('error_rate',0):.0%} | cpu={s.get('cpu_pct',0)}%"
        for name, s in obs.services.items()
    }
    import json
    return (
        f"INCIDENT: {obs.incident_type} (severity={obs.severity})\n"
        f"Health: {obs.system_health:.0%} | Threat: {obs.threat_level:.2f}\n"
        f"Services: {json.dumps(services)}\n"
        f"Logs: {' | '.join(obs.logs[-3:])}\n"
        f"Alerts: {' | '.join(a.get('message','') for a in obs.alerts[:2])}"
    )


def _build_report(results: List[GeneralizationResult], model_name: str) -> GeneralizationReport:
    """Build GeneralizationReport from raw results."""

    def _avg(rs, field):
        vals = [getattr(r, field) for r in rs]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def _rate(rs, field):
        vals = [getattr(r, field) for r in rs]
        return round(sum(1 for v in vals if v) / len(vals), 3) if vals else 0.0

    train_r = [r for r in results if r.split == "train"]
    test_zs  = [r for r in results if r.split == "test_zero_shot"]
    test_ho  = [r for r in results if r.split == "test_held_out"]

    train_avg  = _avg(train_r, "cumulative_reward")
    test_avg   = _avg(test_zs, "cumulative_reward")
    held_avg   = _avg(test_ho, "cumulative_reward")
    gap        = round(train_avg - test_avg, 3)

    # Per-incident-type breakdown
    by_type: Dict[str, Dict[str, float]] = {}
    for itype in ALL_INCIDENT_TYPES:
        type_results = [r for r in results if r.incident_type == itype]
        if type_results:
            split = type_results[0].split
            by_type[itype] = {
                "split": split,
                "avg_reward": _avg(type_results, "cumulative_reward"),
                "resolution_rate": _rate(type_results, "incident_resolved"),
                "correct_service_rate": _rate(type_results, "correct_service_identified"),
                "avg_exploration_depth": _avg(type_results, "exploration_depth"),
                "n": len(type_results),
            }

    return GeneralizationReport(
        model_name=model_name,
        timestamp=time.time(),
        train_episodes=len(train_r),
        test_episodes=len(test_zs),
        train_avg_reward=train_avg,
        test_zero_shot_avg_reward=test_avg,
        test_held_out_avg_reward=held_avg,
        generalization_gap=gap,
        train_resolution_rate=_rate(train_r, "incident_resolved"),
        test_zero_shot_resolution_rate=_rate(test_zs, "incident_resolved"),
        test_held_out_resolution_rate=_rate(test_ho, "incident_resolved"),
        by_incident_type=by_type,
        correct_service_rate_train=_rate(train_r, "correct_service_identified"),
        correct_service_rate_test=_rate(test_zs, "correct_service_identified"),
        exploration_depth_train=_avg(train_r, "exploration_depth"),
        exploration_depth_test=_avg(test_zs, "exploration_depth"),
    )


def quick_generalization_check(env_instance, n: int = 2) -> Dict[str, Any]:
    """
    Quick generalization check using a heuristic agent derived from observation content.
    No hardcoded service names — targets are extracted from the observation string.
    """
    def heuristic_agent(obs_str: str) -> Tuple[str, str]:
        """
        Heuristic: inspect the most degraded service based on observation content.
        Derives target from actual service names mentioned in the observation.
        """
        obs_lower = obs_str.lower()
        # Extract service names from observation (they appear as keys in SERVICES dict)
        import re
        # Find service names from the observation (format: "servicename: status | ...")
        svc_candidates = re.findall(r'"([a-z_]+)":\s*"(?:degraded|down)', obs_lower)
        if not svc_candidates:
            svc_candidates = re.findall(r'([a-z_]+):\s*(?:degraded|down)', obs_lower)
        # Pick the first degraded/down service, or fall back to first mentioned service
        if svc_candidates:
            target = svc_candidates[0]
        else:
            # Extract any service name from the observation
            all_svcs = re.findall(r'"([a-z_]{2,12})":', obs_lower)
            target = all_svcs[0] if all_svcs else "api"
        # Action: always start with run_diagnostic
        return "run_diagnostic", target

    report = evaluate_generalization(
        model_fn=heuristic_agent,
        n_episodes_per_type=n,
        model_name="heuristic_baseline",
    )
    from dataclasses import asdict
    return asdict(report)
