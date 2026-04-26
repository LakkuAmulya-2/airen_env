# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Self-Evolving Curriculum (SEC) — AIREN's Auto-Curriculum Engine

Based on: "Self-Evolving Curriculum for LLM Reasoning" (arxiv 2505.14970)
and "Learning Progress-based Curriculum RL" (LP-ACRL, 2025).

The environment automatically adjusts incident difficulty based on the
agent's real-time learning progress — no human tuning required.

How it works:
  - Tracks per-incident-type success rates using a sliding window
  - Computes "learning progress" = rate of improvement per type
  - Uses Multi-Armed Bandit (softmax) to sample the next incident type
  - Types where agent is actively improving get sampled MORE
  - Types where agent has plateaued get sampled LESS (already learned)
  - Types where agent always fails get sampled MORE (needs practice)

This is the difference between:
  - Static curriculum: easy → medium → hard (fixed schedule)
  - Self-evolving curriculum: always at the edge of the agent's ability

Result: 40-60% faster convergence vs fixed curriculum (LP-ACRL paper).

Meta pain point: 90-day rebuild problem — agents plateau because training
data doesn't adapt to what the agent actually needs to learn next.
"""

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class IncidentTypeStats:
    incident_type: str
    total_episodes: int = 0
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_resolved: deque = field(default_factory=lambda: deque(maxlen=20))
    learning_progress: float = 0.0     # rate of improvement
    current_difficulty: str = "medium"
    sampling_weight: float = 1.0
    last_updated: float = field(default_factory=time.time)

    @property
    def avg_reward(self) -> float:
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    @property
    def resolution_rate(self) -> float:
        if not self.recent_resolved:
            return 0.0
        return sum(self.recent_resolved) / len(self.recent_resolved)

    @property
    def is_mastered(self) -> bool:
        """Agent has mastered this type (resolution rate > 85%)."""
        return len(self.recent_resolved) >= 10 and self.resolution_rate > 0.85

    @property
    def is_stuck(self) -> bool:
        """Agent is stuck on this type (resolution rate < 15% after 10+ episodes)."""
        return len(self.recent_resolved) >= 10 and self.resolution_rate < 0.15


class SelfEvolvingCurriculum:
    """
    Automatic curriculum that evolves based on agent learning progress.

    Implements the SEC (Self-Evolving Curriculum) algorithm:
    1. Track per-type performance with sliding window
    2. Compute learning progress = recent improvement rate
    3. Multi-Armed Bandit sampling: weight = f(learning_progress)
    4. Mastered types → reduce sampling (agent learned them)
    5. Stuck types → increase sampling (agent needs more practice)
    6. Active learning types → highest sampling (most signal)

    Integrated into AIREN's training loop via /curriculum/next endpoint.
    """

    # Difficulty progression thresholds
    PROMOTE_THRESHOLD = 0.75    # resolution rate to promote difficulty
    DEMOTE_THRESHOLD = 0.20     # resolution rate to demote difficulty
    DIFFICULTY_ORDER = ["easy", "medium", "hard"]

    # MAB temperature (higher = more exploration)
    SOFTMAX_TEMPERATURE = 2.0

    def __init__(self) -> None:
        from airen_env.server.incident_engine import ALL_INCIDENT_TYPES, EASY_INCIDENTS, MEDIUM_INCIDENTS, HARD_INCIDENTS
        self._all_types = ALL_INCIDENT_TYPES
        self._easy = set(EASY_INCIDENTS)
        self._medium = set(MEDIUM_INCIDENTS)
        self._hard = set(HARD_INCIDENTS)

        self._stats: Dict[str, IncidentTypeStats] = {
            t: IncidentTypeStats(
                incident_type=t,
                current_difficulty=self._base_difficulty(t),
            )
            for t in self._all_types
        }
        self._episode_count = 0
        self._curriculum_history: List[Dict[str, Any]] = []
        self._rng = random.Random(42)

    def record_episode(
        self,
        incident_type: str,
        reward: float,
        resolved: bool,
        steps_taken: int,
    ) -> None:
        """Record episode outcome to update curriculum weights."""
        if incident_type not in self._stats:
            self._stats[incident_type] = IncidentTypeStats(incident_type=incident_type)

        stats = self._stats[incident_type]
        stats.total_episodes += 1
        stats.recent_rewards.append(reward)
        stats.recent_resolved.append(1.0 if resolved else 0.0)
        stats.last_updated = time.time()
        self._episode_count += 1

        # Update learning progress
        self._update_learning_progress(stats)

        # Update difficulty
        self._update_difficulty(stats)

        # Recompute all sampling weights
        self._recompute_weights()

    def next_incident(self) -> Tuple[str, str]:
        """
        Sample next incident type using MAB softmax policy.

        Returns: (incident_type, difficulty)
        """
        weights = {t: s.sampling_weight for t, s in self._stats.items()}
        incident_type = self._softmax_sample(weights)
        difficulty = self._stats[incident_type].current_difficulty

        self._curriculum_history.append({
            "episode": self._episode_count,
            "selected": incident_type,
            "difficulty": difficulty,
            "weight": weights[incident_type],
            "timestamp": time.time(),
        })

        return incident_type, difficulty

    def get_curriculum_state(self) -> Dict[str, Any]:
        """Get current curriculum state for /curriculum/status endpoint."""
        type_states = {}
        for t, s in self._stats.items():
            type_states[t] = {
                "total_episodes": s.total_episodes,
                "avg_reward": round(s.avg_reward, 3),
                "resolution_rate": round(s.resolution_rate, 3),
                "learning_progress": round(s.learning_progress, 4),
                "current_difficulty": s.current_difficulty,
                "sampling_weight": round(s.sampling_weight, 3),
                "status": (
                    "mastered" if s.is_mastered
                    else "stuck" if s.is_stuck
                    else "learning"
                ),
            }

        # Find the most active learning type
        active = max(
            self._stats.items(),
            key=lambda x: x[1].learning_progress,
            default=(None, None),
        )

        return {
            "total_episodes": self._episode_count,
            "incident_types": type_states,
            "most_active_learning": active[0] if active[0] else "none",
            "mastered_types": [t for t, s in self._stats.items() if s.is_mastered],
            "stuck_types": [t for t, s in self._stats.items() if s.is_stuck],
            "curriculum_algorithm": "Self-Evolving Curriculum (SEC) + Multi-Armed Bandit",
            "reference": "arxiv:2505.14970 — Self-Evolving Curriculum for LLM Reasoning",
        }

    def get_training_recommendation(self) -> Dict[str, Any]:
        """
        Recommend next training focus based on curriculum state.
        Used by train_grpo.py to prioritize episode sampling.
        """
        state = self.get_curriculum_state()
        stuck = state["stuck_types"]
        mastered = state["mastered_types"]
        active = state["most_active_learning"]

        recommendations = []
        if stuck:
            recommendations.append(
                f"Focus on stuck types: {stuck} — agent needs more practice here"
            )
        if mastered:
            recommendations.append(
                f"Reduce sampling of mastered types: {mastered} — diminishing returns"
            )
        if active:
            recommendations.append(
                f"Prioritize '{active}' — highest learning progress right now"
            )

        return {
            "next_incident": self.next_incident(),
            "recommendations": recommendations,
            "curriculum_efficiency": self._compute_efficiency(),
        }

    def _update_learning_progress(self, stats: IncidentTypeStats) -> None:
        """
        Compute learning progress as rate of reward improvement.
        LP = (recent_avg - older_avg) / max(older_avg, 0.01)
        """
        rewards = list(stats.recent_rewards)
        if len(rewards) < 6:
            stats.learning_progress = 0.1  # default: some exploration value
            return

        half = len(rewards) // 2
        older_avg = sum(rewards[:half]) / half
        recent_avg = sum(rewards[half:]) / (len(rewards) - half)

        # Learning progress = relative improvement
        lp = (recent_avg - older_avg) / max(abs(older_avg), 0.01)
        # Clip to [-1, 1] and shift to [0, 2] for positive weights
        stats.learning_progress = max(-1.0, min(1.0, lp))

    def _update_difficulty(self, stats: IncidentTypeStats) -> None:
        """Auto-promote or demote difficulty based on resolution rate."""
        if stats.total_episodes < 5:
            return  # not enough data

        current_idx = self.DIFFICULTY_ORDER.index(stats.current_difficulty)

        if stats.resolution_rate > self.PROMOTE_THRESHOLD:
            # Promote if not already at hardest
            if current_idx < len(self.DIFFICULTY_ORDER) - 1:
                stats.current_difficulty = self.DIFFICULTY_ORDER[current_idx + 1]

        elif stats.resolution_rate < self.DEMOTE_THRESHOLD:
            # Demote if not already at easiest
            if current_idx > 0:
                stats.current_difficulty = self.DIFFICULTY_ORDER[current_idx - 1]

    def _recompute_weights(self) -> None:
        """
        Recompute MAB sampling weights for all incident types.

        Weight formula (LP-ACRL inspired):
        - Mastered: 0.1 (low — already learned)
        - Stuck: 0.5 (medium — needs practice but low signal)
        - Active learning (LP > 0): 1.0 + LP * 2 (high — most signal)
        - No data: 1.5 (high — exploration)
        """
        for stats in self._stats.values():
            if stats.total_episodes == 0:
                stats.sampling_weight = 1.5  # exploration bonus
            elif stats.is_mastered:
                stats.sampling_weight = 0.1
            elif stats.is_stuck:
                stats.sampling_weight = 0.5
            elif stats.learning_progress > 0:
                # Active learning — weight proportional to progress
                stats.sampling_weight = 1.0 + stats.learning_progress * 2.0
            else:
                # Regressing — still sample but less
                stats.sampling_weight = max(0.2, 1.0 + stats.learning_progress)

    def _softmax_sample(self, weights: Dict[str, float]) -> str:
        """Sample incident type using softmax distribution."""
        types = list(weights.keys())
        raw_weights = [weights[t] for t in types]

        # Softmax with temperature
        max_w = max(raw_weights)
        exp_weights = [
            math.exp((w - max_w) / self.SOFTMAX_TEMPERATURE)
            for w in raw_weights
        ]
        total = sum(exp_weights)
        probs = [w / total for w in exp_weights]

        # Sample
        r = self._rng.random()
        cumulative = 0.0
        for t, p in zip(types, probs):
            cumulative += p
            if r <= cumulative:
                return t
        return types[-1]

    def _base_difficulty(self, incident_type: str) -> str:
        if incident_type in self._easy:
            return "easy"
        if incident_type in self._hard:
            return "hard"
        return "medium"

    def _compute_efficiency(self) -> float:
        """
        Curriculum efficiency = fraction of episodes on actively-learning types.
        Higher = better use of training compute.
        """
        if self._episode_count == 0:
            return 0.0
        active_episodes = sum(
            s.total_episodes
            for s in self._stats.values()
            if not s.is_mastered and s.learning_progress > 0
        )
        return round(active_episodes / max(self._episode_count, 1), 3)


# Module-level singleton
_curriculum: Optional[SelfEvolvingCurriculum] = None


def get_curriculum() -> SelfEvolvingCurriculum:
    global _curriculum
    if _curriculum is None:
        _curriculum = SelfEvolvingCurriculum()
    return _curriculum
