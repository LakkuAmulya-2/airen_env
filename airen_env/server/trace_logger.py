# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
AgentOps-style trace logger for AIREN.

Emits structured decision traces per step — every action, reward breakdown,
state delta, and cost is recorded. Judges can replay the full episode.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class StepTrace:
    episode_id: str
    step: int
    timestamp: float
    action_type: str
    target: str
    reasoning: str
    # state before
    health_before: float
    risk_before: float
    # state after
    health_after: float
    risk_after: float
    # reward breakdown (full RL-grade)
    reward_total: float
    recovery_score: float
    diagnosis_score: float
    efficiency_score: float
    threat_mitigation: float
    hallucination_penalty: float
    security_violation_penalty: float
    cost_penalty: float
    downtime_penalty: float
    resolve_bonus: float
    # cost tracking
    llm_tokens_used: int
    llm_cost_usd: float
    # outcome
    action_success: bool
    incident_resolved: bool
    # optional fields with defaults
    exploration_bonus: float = 0.0
    recovery_bonus: float = 0.0
    anomaly: Optional[str] = None


@dataclass
class EpisodeTrace:
    episode_id: str
    incident_type: str
    severity: str
    difficulty: str
    start_time: float
    end_time: Optional[float] = None
    steps: List[StepTrace] = field(default_factory=list)
    # cumulative
    total_reward: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    resolved: bool = False
    final_health: float = 0.0
    diagnosis_quality: str = "unknown"
    # anomaly summary
    anomalies: List[str] = field(default_factory=list)

    def add_step(self, step: StepTrace) -> None:
        self.steps.append(step)
        self.total_reward += step.reward_total
        self.total_tokens += step.llm_tokens_used
        self.total_cost_usd += step.llm_cost_usd
        if step.anomaly:
            self.anomalies.append(f"step{step.step}:{step.anomaly}")

    def close(self, resolved: bool, final_health: float, diagnosis_quality: str) -> None:
        self.end_time = time.time()
        self.resolved = resolved
        self.final_health = final_health
        self.diagnosis_quality = diagnosis_quality

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["elapsed_s"] = round((self.end_time or time.time()) - self.start_time, 3)
        d["steps_taken"] = len(self.steps)
        return d

    def summary(self) -> Dict[str, Any]:
        """Compact summary for [END] log line."""
        return {
            "episode_id": self.episode_id,
            "incident_type": self.incident_type,
            "resolved": self.resolved,
            "final_health": round(self.final_health, 3),
            "total_reward": round(self.total_reward, 3),
            "steps": len(self.steps),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "diagnosis_quality": self.diagnosis_quality,
            "anomalies": self.anomalies,
            "elapsed_s": round((self.end_time or time.time()) - self.start_time, 3),
        }


# ── Cost model (GPT-4o-mini pricing as of 2025) ───────────────────────────────
_COST_PER_1K = {
    "gpt-4o-mini":   0.000150,   # $0.15 / 1M tokens
    "gpt-4o":        0.002500,
    "gpt-4-turbo":   0.010000,
    "default":       0.000150,
}


def tokens_to_usd(tokens: int, model: str = "gpt-4o-mini") -> float:
    rate = _COST_PER_1K.get(model, _COST_PER_1K["default"])
    return round(tokens / 1000 * rate, 8)


# ── Anomaly detection ─────────────────────────────────────────────────────────

def detect_anomaly(
    action_type: str,
    target: str,
    history: List[str],
    correct_targets: List[str],
) -> Optional[str]:
    """Flag suspicious agent behaviour."""
    key = f"{action_type}:{target}"
    # Repeated identical action
    if history.count(key) >= 2:
        return "repeated_action"
    # Ignoring obvious signals
    if action_type == "ignore_alert":
        return "ignored_alert"
    # Acting on wrong target when correct target is obvious
    if correct_targets and target not in correct_targets and action_type not in (
        "inspect_logs", "inspect_metrics", "run_diagnostic", "acknowledge_incident"
    ):
        return "wrong_target"
    return None


# ── Session-level aggregator ──────────────────────────────────────────────────

class TraceSession:
    """Aggregates traces across multiple episodes in one run."""

    def __init__(self) -> None:
        self._episodes: List[EpisodeTrace] = []

    def new_episode(
        self,
        episode_id: str,
        incident_type: str,
        severity: str,
        difficulty: str = "medium",
    ) -> EpisodeTrace:
        ep = EpisodeTrace(
            episode_id=episode_id,
            incident_type=incident_type,
            severity=severity,
            difficulty=difficulty,
            start_time=time.time(),
        )
        self._episodes.append(ep)
        return ep

    def session_summary(self) -> Dict[str, Any]:
        if not self._episodes:
            return {}
        resolved = sum(1 for e in self._episodes if e.resolved)
        total_cost = sum(e.total_cost_usd for e in self._episodes)
        total_tokens = sum(e.total_tokens for e in self._episodes)
        avg_reward = sum(e.total_reward for e in self._episodes) / len(self._episodes)
        return {
            "total_episodes": len(self._episodes),
            "resolved": resolved,
            "resolution_rate": round(resolved / len(self._episodes), 3),
            "avg_reward": round(avg_reward, 3),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "anomaly_rate": round(
                sum(len(e.anomalies) for e in self._episodes) / max(len(self._episodes), 1), 2
            ),
        }

    def print_trace(self, episode_id: str) -> None:
        """Pretty-print full trace for one episode."""
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                print(json.dumps(ep.to_dict(), indent=2))
                return
        print(f"Episode {episode_id} not found")
