# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Incident Replay Forensics — Upgrade #10

Deterministic replay of failed incidents for root cause analysis.
Like an airplane black box — every incident can be replayed exactly
to understand where the agent went wrong.

Real pain point: Post-mortem analysis is standard practice at every
company. Without replay, you can't understand why an agent failed.
This automates the post-mortem.

Usage:
    # After a failed production incident
    replayer = IncidentReplayForensics()
    replayer.record_episode(episode_id, incident_type, seed, actions, rewards)
    analysis = replayer.analyze_failure(episode_id)
    print(analysis.root_cause_step)
    print(analysis.suggested_alternative)
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EpisodeRecord:
    episode_id: str
    incident_type: str
    seed: int
    attacker_seed: int
    actions: List[Dict[str, Any]]       # [{action_type, target, reasoning, step}]
    rewards: List[float]
    health_trajectory: List[float]
    resolved: bool
    final_health: float
    root_cause: str
    correct_actions: List[str]
    correct_targets: List[str]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureAnalysis:
    episode_id: str
    incident_type: str
    resolved: bool
    root_cause_step: Optional[int]          # step where agent made critical mistake
    critical_action: Optional[str]
    critical_target: Optional[str]
    critical_reward: Optional[float]
    suggested_alternative: str
    failure_pattern: str                    # wrong_target | wrong_action | loop | timeout
    health_at_failure: float
    recovery_possible: bool
    lessons_learned: List[str]
    replay_available: bool


class IncidentReplayForensics:
    """
    Replay failed incidents for root cause analysis.

    Stores episode records and provides:
    1. Deterministic replay with full observability
    2. Root cause analysis (where did agent go wrong?)
    3. Alternative action suggestions
    4. Pattern detection across multiple failures
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
    ) -> None:
        self._episodes: Dict[str, EpisodeRecord] = {}
        self._storage_dir = Path(storage_dir or "completions")
        self._storage_dir.mkdir(exist_ok=True)
        self._analysis_cache: Dict[str, FailureAnalysis] = {}

        # Load existing episodes from disk
        self._load_from_disk()

    def record_episode(
        self,
        episode_id: str,
        incident_type: str,
        seed: int,
        actions: List[Dict[str, Any]],
        rewards: List[float],
        health_trajectory: List[float],
        resolved: bool,
        final_health: float,
        root_cause: str,
        correct_actions: List[str],
        correct_targets: List[str],
        attacker_seed: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a completed episode for future replay and analysis."""
        record = EpisodeRecord(
            episode_id=episode_id,
            incident_type=incident_type,
            seed=seed,
            attacker_seed=attacker_seed,
            actions=actions,
            rewards=rewards,
            health_trajectory=health_trajectory,
            resolved=resolved,
            final_health=final_health,
            root_cause=root_cause,
            correct_actions=correct_actions,
            correct_targets=correct_targets,
            metadata=metadata or {},
        )
        self._episodes[episode_id] = record
        self._persist_episode(record)

    def analyze_failure(
        self,
        episode_id: str,
    ) -> Optional[FailureAnalysis]:
        """
        Analyze a failed episode to find root cause.
        Returns None if episode not found or was successful.
        """
        if episode_id in self._analysis_cache:
            return self._analysis_cache[episode_id]

        record = self._episodes.get(episode_id)
        if record is None:
            return None

        analysis = self._run_analysis(record)
        self._analysis_cache[episode_id] = analysis
        return analysis

    def replay_episode(
        self,
        episode_id: str,
        debug_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        Replay a recorded episode step by step.
        Returns full replay with internal state at each step.
        """
        record = self._episodes.get(episode_id)
        if record is None:
            return {"error": f"Episode '{episode_id}' not found"}

        replay_steps = []
        for i, (action, reward) in enumerate(zip(record.actions, record.rewards)):
            health = record.health_trajectory[i] if i < len(record.health_trajectory) else 0.0
            health_prev = record.health_trajectory[i - 1] if i > 0 else health
            health_delta = health - health_prev

            step_info: Dict[str, Any] = {
                "step": i + 1,
                "action_type": action.get("action_type", ""),
                "target": action.get("target", ""),
                "reasoning": action.get("reasoning", ""),
                "reward": round(reward, 4),
                "health": round(health, 3),
                "health_delta": round(health_delta, 3),
            }

            if debug_mode:
                # Annotate with correctness
                correct_target = action.get("target", "") in record.correct_targets
                correct_action = action.get("action_type", "") in record.correct_actions
                step_info["correct_target"] = correct_target
                step_info["correct_action"] = correct_action
                step_info["annotation"] = _annotate_step(
                    action, reward, correct_target, correct_action, record
                )

            replay_steps.append(step_info)

        analysis = self._run_analysis(record)

        return {
            "episode_id": episode_id,
            "incident_type": record.incident_type,
            "seed": record.seed,
            "resolved": record.resolved,
            "final_health": round(record.final_health, 3),
            "root_cause": record.root_cause,
            "steps": replay_steps,
            "root_cause_analysis": {
                "failure_step": analysis.root_cause_step,
                "critical_action": analysis.critical_action,
                "critical_target": analysis.critical_target,
                "failure_pattern": analysis.failure_pattern,
                "suggested_alternative": analysis.suggested_alternative,
                "lessons_learned": analysis.lessons_learned,
            } if analysis else None,
        }

    def get_failure_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns across all failed episodes.
        Identifies systemic weaknesses in agent behavior.
        """
        failed = [r for r in self._episodes.values() if not r.resolved]
        if not failed:
            return {"message": "No failed episodes recorded yet", "total_episodes": len(self._episodes)}

        # Count failure patterns
        pattern_counts: Dict[str, int] = {}
        incident_failure_rates: Dict[str, Dict[str, int]] = {}

        for record in failed:
            analysis = self._run_analysis(record)
            pattern = analysis.failure_pattern
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

            itype = record.incident_type
            if itype not in incident_failure_rates:
                incident_failure_rates[itype] = {"failed": 0, "total": 0}
            incident_failure_rates[itype]["failed"] += 1

        for record in self._episodes.values():
            itype = record.incident_type
            if itype not in incident_failure_rates:
                incident_failure_rates[itype] = {"failed": 0, "total": 0}
            incident_failure_rates[itype]["total"] += 1

        # Most common wrong targets
        wrong_targets: Dict[str, int] = {}
        for record in failed:
            for action in record.actions:
                target = action.get("target", "")
                if target and target not in record.correct_targets:
                    wrong_targets[target] = wrong_targets.get(target, 0) + 1

        return {
            "total_episodes": len(self._episodes),
            "failed_episodes": len(failed),
            "failure_rate": round(len(failed) / max(len(self._episodes), 1), 3),
            "failure_patterns": pattern_counts,
            "most_common_pattern": max(pattern_counts, key=pattern_counts.get) if pattern_counts else None,
            "incident_failure_rates": {
                itype: {
                    "failure_rate": round(
                        v["failed"] / max(v["total"], 1), 3
                    ),
                    **v,
                }
                for itype, v in incident_failure_rates.items()
            },
            "most_common_wrong_targets": sorted(
                wrong_targets.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def list_episodes(
        self,
        incident_type: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List recorded episodes with optional filters."""
        episodes = list(self._episodes.values())

        if incident_type:
            episodes = [e for e in episodes if e.incident_type == incident_type]
        if resolved is not None:
            episodes = [e for e in episodes if e.resolved == resolved]

        # Sort by timestamp descending
        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        episodes = episodes[:limit]

        return [
            {
                "episode_id": e.episode_id,
                "incident_type": e.incident_type,
                "seed": e.seed,
                "resolved": e.resolved,
                "final_health": round(e.final_health, 3),
                "steps": len(e.actions),
                "timestamp": e.timestamp,
            }
            for e in episodes
        ]

    def _run_analysis(self, record: EpisodeRecord) -> FailureAnalysis:
        """Run root cause analysis on an episode record."""
        if not record.actions:
            return FailureAnalysis(
                episode_id=record.episode_id,
                incident_type=record.incident_type,
                resolved=record.resolved,
                root_cause_step=None,
                critical_action=None,
                critical_target=None,
                critical_reward=None,
                suggested_alternative="No actions taken",
                failure_pattern="timeout",
                health_at_failure=record.final_health,
                recovery_possible=False,
                lessons_learned=["Agent took no actions — check environment connectivity"],
                replay_available=True,
            )

        # Find first catastrophic action (reward < -0.3)
        root_cause_step = None
        critical_action = None
        critical_target = None
        critical_reward = None

        for i, (action, reward) in enumerate(zip(record.actions, record.rewards)):
            if reward < -0.3:
                root_cause_step = i + 1
                critical_action = action.get("action_type", "")
                critical_target = action.get("target", "")
                critical_reward = round(reward, 4)
                break

        # Determine failure pattern
        failure_pattern = _classify_failure_pattern(
            record.actions, record.rewards, record.correct_targets, record.correct_actions
        )

        # Generate suggestion
        suggestion = _generate_suggestion(
            failure_pattern, critical_action, critical_target,
            record.correct_actions, record.correct_targets, record.incident_type
        )

        # Lessons learned
        lessons = _extract_lessons(
            failure_pattern, record.incident_type,
            record.correct_actions, record.correct_targets
        )

        return FailureAnalysis(
            episode_id=record.episode_id,
            incident_type=record.incident_type,
            resolved=record.resolved,
            root_cause_step=root_cause_step,
            critical_action=critical_action,
            critical_target=critical_target,
            critical_reward=critical_reward,
            suggested_alternative=suggestion,
            failure_pattern=failure_pattern,
            health_at_failure=record.final_health,
            recovery_possible=record.final_health > 0.3,
            lessons_learned=lessons,
            replay_available=True,
        )

    def _persist_episode(self, record: EpisodeRecord) -> None:
        """Persist episode to disk for cross-session replay."""
        try:
            path = self._storage_dir / f"{record.episode_id}.json"
            data = {
                "episode_id": record.episode_id,
                "incident_type": record.incident_type,
                "seed": record.seed,
                "attacker_seed": record.attacker_seed,
                "actions": record.actions,
                "rewards": record.rewards,
                "health_trajectory": record.health_trajectory,
                "resolved": record.resolved,
                "final_health": record.final_health,
                "root_cause": record.root_cause,
                "correct_actions": record.correct_actions,
                "correct_targets": record.correct_targets,
                "timestamp": record.timestamp,
                "metadata": record.metadata,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Persistence is best-effort

    def _load_from_disk(self) -> None:
        """Load existing episode records from disk."""
        try:
            for path in self._storage_dir.glob("*.json"):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if "episode_id" in data and "actions" in data:
                        record = EpisodeRecord(**{
                            k: v for k, v in data.items()
                            if k in EpisodeRecord.__dataclass_fields__
                        })
                        self._episodes[record.episode_id] = record
                except Exception:
                    pass
        except Exception:
            pass


def _classify_failure_pattern(
    actions: List[Dict[str, Any]],
    rewards: List[float],
    correct_targets: List[str],
    correct_actions: List[str],
) -> str:
    """Classify the primary failure pattern."""
    if not actions:
        return "timeout"

    # Check for loop pattern
    tool_sequence = [a.get("action_type", "") for a in actions]
    if len(tool_sequence) >= 4:
        last_4 = tool_sequence[-4:]
        if len(set(last_4)) == 1:
            return "loop"

    # Check for wrong target pattern
    wrong_target_count = sum(
        1 for a in actions
        if a.get("target", "") not in correct_targets
        and a.get("action_type", "") not in ("inspect_logs", "inspect_metrics", "run_diagnostic")
    )
    if wrong_target_count > len(actions) * 0.5:
        return "wrong_target"

    # Check for wrong action pattern
    wrong_action_count = sum(
        1 for a in actions
        if a.get("action_type", "") not in correct_actions
        and a.get("action_type", "") not in ("inspect_logs", "inspect_metrics", "run_diagnostic", "acknowledge_incident")
    )
    if wrong_action_count > len(actions) * 0.5:
        return "wrong_action"

    # Check for timeout (ran out of steps without resolving)
    if len(actions) >= 8:
        return "timeout"

    return "wrong_target"


def _generate_suggestion(
    pattern: str,
    critical_action: Optional[str],
    critical_target: Optional[str],
    correct_actions: List[str],
    correct_targets: List[str],
    incident_type: str,
) -> str:
    correct_action_str = correct_actions[0] if correct_actions else "run_diagnostic"
    correct_target_str = correct_targets[0] if correct_targets else "the affected service"

    if pattern == "wrong_target":
        return (
            f"Agent targeted '{critical_target}' but the root cause was in '{correct_target_str}'. "
            f"For {incident_type}, always run_diagnostic on '{correct_target_str}' first. "
            f"Look for the service with highest error_rate or CPU in the observation."
        )
    elif pattern == "wrong_action":
        return (
            f"Agent used '{critical_action}' but the correct action was '{correct_action_str}'. "
            f"For {incident_type}, the fix sequence is: "
            f"run_diagnostic({correct_target_str}) → {correct_action_str}({correct_target_str})."
        )
    elif pattern == "loop":
        return (
            f"Agent got stuck repeating the same action. "
            f"When an action doesn't improve health, try a different approach. "
            f"For {incident_type}: {correct_action_str}({correct_target_str})."
        )
    elif pattern == "timeout":
        return (
            f"Agent ran out of steps without resolving. "
            f"Act faster: diagnose first, then fix immediately. "
            f"For {incident_type}: step 1 = run_diagnostic({correct_target_str}), "
            f"step 2 = {correct_action_str}({correct_target_str})."
        )
    return f"For {incident_type}: {correct_action_str}({correct_target_str})"


def _annotate_step(
    action: Dict[str, Any],
    reward: float,
    correct_target: bool,
    correct_action: bool,
    record: EpisodeRecord,
) -> str:
    """Generate human-readable annotation for a replay step."""
    atype = action.get("action_type", "")
    target = action.get("target", "")

    if reward < -0.3:
        return f"❌ CRITICAL MISTAKE: {atype}({target}) caused severe penalty ({reward:.3f})"
    elif reward < 0:
        return f"⚠️ Wrong action: {atype}({target}) — negative reward ({reward:.3f})"
    elif correct_target and correct_action:
        return f"✅ Correct: {atype}({target}) — right service, right action"
    elif correct_target:
        return f"🔍 Partial: {atype}({target}) — right service, but {record.correct_actions[0] if record.correct_actions else 'different action'} needed"
    elif reward > 0.3:
        return f"✅ Good: {atype}({target}) — positive reward ({reward:.3f})"
    else:
        return f"➡️ {atype}({target}) — reward: {reward:.3f}"


def _extract_lessons(
    pattern: str,
    incident_type: str,
    correct_actions: List[str],
    correct_targets: List[str],
) -> List[str]:
    """Extract actionable lessons from failure analysis."""
    lessons = []
    correct_target = correct_targets[0] if correct_targets else "the root cause service"
    correct_action = correct_actions[0] if correct_actions else "the correct fix"

    if pattern == "wrong_target":
        lessons.append(
            f"For {incident_type}: always investigate '{correct_target}' first — "
            f"it shows the highest anomaly signals"
        )
        lessons.append(
            "Read ALL service metrics before acting — the highest error_rate service is usually the root cause"
        )
    elif pattern == "wrong_action":
        lessons.append(
            f"For {incident_type}: '{correct_action}' is the correct fix, not restart/rollback"
        )
        lessons.append(
            "Match action type to incident type: overload→apply_fix, leak→restart, deployment→rollback"
        )
    elif pattern == "loop":
        lessons.append("If an action doesn't improve health after 2 tries, switch approach")
        lessons.append("The ARL circuit breaker will block you after 3 repeats — vary your actions")
    elif pattern == "timeout":
        lessons.append("Diagnose in step 1, fix in step 2 — don't waste steps on exploration")
        lessons.append(f"For {incident_type}: run_diagnostic({correct_target}) → {correct_action}({correct_target})")

    lessons.append(
        "Always provide detailed reasoning — it improves diagnosis score and helps the LLM judge"
    )
    return lessons


# Module-level singleton
_replayer: Optional[IncidentReplayForensics] = None


def get_replay_forensics() -> IncidentReplayForensics:
    global _replayer
    if _replayer is None:
        _replayer = IncidentReplayForensics()
    return _replayer
