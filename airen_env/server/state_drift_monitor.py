# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
State Drift Monitor — AIREN's Production Stability Layer

Based on:
- "The 90-Day Rebuild Problem" (cybernative.ai, 2026)
- "Agents degrade over time even when no one touches them" (Salesforce, 2026)
- "Audited Skill-Graph Self-Improvement" (arxiv 2512.23760)
- Amazon's 90-day code safety reset after AI agent retail outages (March 2026)

Real production finding: 70% of enterprises rebuild AI agent systems every
90 days. The #1 cause: agents degrade over time due to:
  1. Distribution shift: real-world data drifts from training distribution
  2. Reward drift: reward function assumptions become stale
  3. Behavioral drift: agent develops subtle policy changes over time
  4. Environment drift: the environment itself changes (new incident types)

AIREN's State Drift Monitor:
  - Tracks agent performance over time with statistical tests
  - Detects when performance degrades beyond acceptable threshold
  - Triggers automatic retraining recommendation
  - Provides drift report for human review

This is what prevents the 90-day rebuild cycle:
  Instead of waiting for catastrophic failure, detect drift early
  and retrain proactively before production incidents occur.

Meta pain point: Amazon's March 2026 retail outages were caused by an
AI agent that had drifted from its training distribution — nobody noticed
until it caused a SEV1.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DriftAlert:
    drift_type: str             # performance | reward | behavioral | environment
    severity: str               # CRITICAL | HIGH | MEDIUM | LOW
    metric: str                 # which metric drifted
    baseline_value: float
    current_value: float
    drift_magnitude: float      # relative change
    recommendation: str
    detected_at: float = field(default_factory=time.time)


@dataclass
class DriftReport:
    alerts: List[DriftAlert]
    overall_drift_score: float  # 0.0 = no drift, 1.0 = severe drift
    recommendation: str         # retrain | monitor | ok
    baseline_episodes: int
    current_episodes: int
    drift_detected: bool


class StateDriftMonitor:
    """
    Monitors agent performance for drift over time.

    Uses statistical process control (SPC) to detect when performance
    metrics deviate significantly from baseline.

    Four drift types:
    1. Performance drift: resolution rate / avg reward declining
    2. Reward drift: reward distribution shifting
    3. Behavioral drift: action distribution changing
    4. Environment drift: new incident types appearing

    Detection method: CUSUM (Cumulative Sum) control chart
    - Sensitive to gradual drift (not just sudden changes)
    - Low false positive rate
    - Standard in production monitoring systems
    """

    # Drift thresholds
    PERFORMANCE_DRIFT_THRESHOLD = 0.15   # 15% relative decline
    REWARD_DRIFT_THRESHOLD = 0.20        # 20% reward distribution shift
    BEHAVIORAL_DRIFT_THRESHOLD = 0.25    # 25% action distribution change
    CUSUM_THRESHOLD = 5.0                # CUSUM control limit

    # Baseline window
    BASELINE_WINDOW = 20                 # episodes for baseline
    MONITORING_WINDOW = 10               # episodes for current performance

    def __init__(self) -> None:
        # Performance tracking
        self._all_rewards: deque = deque(maxlen=200)
        self._all_resolved: deque = deque(maxlen=200)
        self._action_counts: Dict[str, int] = {}
        self._incident_type_counts: Dict[str, int] = {}

        # CUSUM state
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0
        self._cusum_target: float = 0.0
        self._cusum_initialized: bool = False

        # Drift history
        self._drift_alerts: List[DriftAlert] = []
        self._episode_count: int = 0
        self._last_report_time: float = time.time()

    def record_episode(
        self,
        reward: float,
        resolved: bool,
        action_types: List[str],
        incident_type: str,
    ) -> None:
        """Record episode metrics for drift monitoring."""
        self._episode_count += 1
        self._all_rewards.append(reward)
        self._all_resolved.append(1.0 if resolved else 0.0)

        for action in action_types:
            self._action_counts[action] = self._action_counts.get(action, 0) + 1

        self._incident_type_counts[incident_type] = (
            self._incident_type_counts.get(incident_type, 0) + 1
        )

        # Update CUSUM
        self._update_cusum(reward)

    def check_drift(self) -> DriftReport:
        """
        Run full drift detection analysis.
        Returns DriftReport with alerts and recommendations.
        """
        if self._episode_count < self.BASELINE_WINDOW + self.MONITORING_WINDOW:
            return DriftReport(
                alerts=[],
                overall_drift_score=0.0,
                recommendation="ok",
                baseline_episodes=min(self._episode_count, self.BASELINE_WINDOW),
                current_episodes=self._episode_count,
                drift_detected=False,
            )

        alerts = []
        rewards = list(self._all_rewards)
        resolved = list(self._all_resolved)

        # Split into baseline and current windows
        baseline_rewards = rewards[:self.BASELINE_WINDOW]
        current_rewards = rewards[-self.MONITORING_WINDOW:]
        baseline_resolved = resolved[:self.BASELINE_WINDOW]
        current_resolved = resolved[-self.MONITORING_WINDOW:]

        # 1. Performance drift
        perf_alert = self._check_performance_drift(
            baseline_rewards, current_rewards,
            baseline_resolved, current_resolved
        )
        if perf_alert:
            alerts.append(perf_alert)

        # 2. Reward distribution drift
        reward_alert = self._check_reward_drift(baseline_rewards, current_rewards)
        if reward_alert:
            alerts.append(reward_alert)

        # 3. CUSUM drift
        cusum_alert = self._check_cusum_drift()
        if cusum_alert:
            alerts.append(cusum_alert)

        # 4. Behavioral drift
        behav_alert = self._check_behavioral_drift()
        if behav_alert:
            alerts.append(behav_alert)

        # Compute overall drift score
        if not alerts:
            drift_score = 0.0
        else:
            severity_weights = {"CRITICAL": 1.0, "HIGH": 0.7, "MEDIUM": 0.4, "LOW": 0.2}
            drift_score = min(1.0, sum(
                severity_weights.get(a.severity, 0.3) for a in alerts
            ) / 2.0)

        # Recommendation
        if drift_score >= 0.7:
            recommendation = "retrain"
        elif drift_score >= 0.4:
            recommendation = "monitor"
        else:
            recommendation = "ok"

        self._drift_alerts.extend(alerts)
        self._last_report_time = time.time()

        return DriftReport(
            alerts=alerts,
            overall_drift_score=round(drift_score, 3),
            recommendation=recommendation,
            baseline_episodes=self.BASELINE_WINDOW,
            current_episodes=self._episode_count,
            drift_detected=len(alerts) > 0,
        )

    def _check_performance_drift(
        self,
        baseline_rewards: List[float],
        current_rewards: List[float],
        baseline_resolved: List[float],
        current_resolved: List[float],
    ) -> Optional[DriftAlert]:
        """Check if resolution rate or avg reward has declined."""
        baseline_avg = sum(baseline_rewards) / len(baseline_rewards)
        current_avg = sum(current_rewards) / len(current_rewards)
        baseline_rate = sum(baseline_resolved) / len(baseline_resolved)
        current_rate = sum(current_resolved) / len(current_resolved)

        reward_drift = (baseline_avg - current_avg) / max(abs(baseline_avg), 0.01)
        rate_drift = (baseline_rate - current_rate) / max(baseline_rate, 0.01)

        if reward_drift > self.PERFORMANCE_DRIFT_THRESHOLD:
            severity = "CRITICAL" if reward_drift > 0.30 else "HIGH"
            return DriftAlert(
                drift_type="performance",
                severity=severity,
                metric="avg_reward",
                baseline_value=round(baseline_avg, 3),
                current_value=round(current_avg, 3),
                drift_magnitude=round(reward_drift, 3),
                recommendation=(
                    f"Avg reward dropped {reward_drift:.0%} from baseline. "
                    f"Consider retraining on recent episodes."
                ),
            )

        if rate_drift > self.PERFORMANCE_DRIFT_THRESHOLD:
            severity = "HIGH" if rate_drift > 0.25 else "MEDIUM"
            return DriftAlert(
                drift_type="performance",
                severity=severity,
                metric="resolution_rate",
                baseline_value=round(baseline_rate, 3),
                current_value=round(current_rate, 3),
                drift_magnitude=round(rate_drift, 3),
                recommendation=(
                    f"Resolution rate dropped {rate_drift:.0%} from baseline. "
                    f"Agent may be encountering new incident patterns."
                ),
            )

        return None

    def _check_reward_drift(
        self,
        baseline: List[float],
        current: List[float],
    ) -> Optional[DriftAlert]:
        """Check if reward distribution has shifted (KL divergence proxy)."""
        # Use mean + std comparison as KL divergence proxy
        b_mean = sum(baseline) / len(baseline)
        c_mean = sum(current) / len(current)
        b_std = _std(baseline)
        c_std = _std(current)

        # Normalized distance
        mean_shift = abs(c_mean - b_mean) / max(b_std, 0.01)

        if mean_shift > 2.0:  # > 2 standard deviations
            return DriftAlert(
                drift_type="reward",
                severity="MEDIUM",
                metric="reward_distribution",
                baseline_value=round(b_mean, 3),
                current_value=round(c_mean, 3),
                drift_magnitude=round(mean_shift, 3),
                recommendation=(
                    f"Reward distribution shifted {mean_shift:.1f}σ from baseline. "
                    f"Check if environment dynamics have changed."
                ),
            )
        return None

    def _check_cusum_drift(self) -> Optional[DriftAlert]:
        """Check CUSUM control chart for gradual drift."""
        if not self._cusum_initialized:
            return None

        if self._cusum_pos > self.CUSUM_THRESHOLD:
            return DriftAlert(
                drift_type="performance",
                severity="HIGH",
                metric="cusum_positive",
                baseline_value=self._cusum_target,
                current_value=self._cusum_pos,
                drift_magnitude=self._cusum_pos / self.CUSUM_THRESHOLD,
                recommendation=(
                    f"CUSUM control chart: positive drift detected (CUSUM={self._cusum_pos:.2f}). "
                    f"Performance improving — consider harder curriculum."
                ),
            )

        if self._cusum_neg > self.CUSUM_THRESHOLD:
            return DriftAlert(
                drift_type="performance",
                severity="CRITICAL",
                metric="cusum_negative",
                baseline_value=self._cusum_target,
                current_value=-self._cusum_neg,
                drift_magnitude=self._cusum_neg / self.CUSUM_THRESHOLD,
                recommendation=(
                    f"CUSUM control chart: negative drift detected (CUSUM={self._cusum_neg:.2f}). "
                    f"Performance declining — retrain immediately."
                ),
            )

        return None

    def _check_behavioral_drift(self) -> Optional[DriftAlert]:
        """Check if action distribution has changed significantly."""
        if not self._action_counts or self._episode_count < 30:
            return None

        total = sum(self._action_counts.values())
        if total == 0:
            return None

        # Check if any action type dominates (>60% of actions)
        for action, count in self._action_counts.items():
            ratio = count / total
            if ratio > 0.60 and action in ("acknowledge_incident", "inspect_metrics"):
                return DriftAlert(
                    drift_type="behavioral",
                    severity="MEDIUM",
                    metric=f"action_dominance_{action}",
                    baseline_value=0.20,  # expected ~20% per action type
                    current_value=round(ratio, 3),
                    drift_magnitude=round(ratio - 0.20, 3),
                    recommendation=(
                        f"Action '{action}' dominates at {ratio:.0%} of all actions. "
                        f"Agent may be stuck in a behavioral rut."
                    ),
                )
        return None

    def _update_cusum(self, reward: float) -> None:
        """Update CUSUM control chart."""
        if not self._cusum_initialized and self._episode_count >= self.BASELINE_WINDOW:
            rewards = list(self._all_rewards)[:self.BASELINE_WINDOW]
            self._cusum_target = sum(rewards) / len(rewards)
            self._cusum_initialized = True

        if self._cusum_initialized:
            k = 0.5  # allowance parameter
            deviation = reward - self._cusum_target
            self._cusum_pos = max(0, self._cusum_pos + deviation - k)
            self._cusum_neg = max(0, self._cusum_neg - deviation - k)

    def get_stats(self) -> Dict[str, Any]:
        recent_rewards = list(self._all_rewards)[-20:]
        recent_resolved = list(self._all_resolved)[-20:]
        return {
            "total_episodes_monitored": self._episode_count,
            "recent_avg_reward": round(sum(recent_rewards) / max(len(recent_rewards), 1), 3),
            "recent_resolution_rate": round(sum(recent_resolved) / max(len(recent_resolved), 1), 3),
            "cusum_positive": round(self._cusum_pos, 3),
            "cusum_negative": round(self._cusum_neg, 3),
            "cusum_threshold": self.CUSUM_THRESHOLD,
            "total_drift_alerts": len(self._drift_alerts),
            "recent_alerts": [
                {
                    "type": a.drift_type,
                    "severity": a.severity,
                    "metric": a.metric,
                    "magnitude": a.drift_magnitude,
                }
                for a in self._drift_alerts[-5:]
            ],
            "meta_pain_point": "Amazon 90-day reset: agents degrade without drift detection",
            "solution": "CUSUM control charts + statistical drift detection",
        }


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 1.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


# Module-level singleton
_monitor: Optional[StateDriftMonitor] = None


def get_drift_monitor() -> StateDriftMonitor:
    global _monitor
    if _monitor is None:
        _monitor = StateDriftMonitor()
    return _monitor
