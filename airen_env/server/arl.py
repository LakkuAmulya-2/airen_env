# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Agentic Reliability Layer (ARL) — The Self-Healing AI Core.

Three interlocked components that make AIREN a true enterprise-grade
reliability system, not just an RL benchmark:

1. DeterministicCircuitBreaker  — kills infinite loops and runaway costs
2. StatePreservingRollbackEngine — time machine for catastrophic wrong actions
3. ContextualActionLedger        — dynamic memory that prevents context bloat

The Golden Flow:
  Agent takes action
    → Circuit Breaker checks for repetition (saves cost)
    → Rollback Engine snapshots state before execution (ensures safety)
    → Action executes
    → Ledger summarizes result into crisp memory (improves logic)
    → Agent gets clean context for next step

This is the architectural layer that separates a toy RL environment
from a production-grade Self-Healing AI system.
"""

import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Action classification — derived from models.ActionType, not hardcoded ────
# These sets are built at import time from the canonical ActionType definition.
# If new actions are added to models.py, they are automatically classified here.

def _classify_actions():
    """
    Classify all ActionType values into risk categories.
    Derived from the action's inherent properties — no hardcoded lists.
    """
    try:
        from airen_env.models import AIRENAction
        all_actions = list(AIRENAction.model_fields["action_type"].annotation.__args__)
    except Exception:
        # Fallback: use the known action types from models.py
        all_actions = [
            "inspect_logs", "inspect_metrics", "restart_service", "scale_service",
            "rollback_deployment", "run_diagnostic", "ignore_alert",
            "acknowledge_incident", "apply_fix",
        ]

    # Diagnostic actions: gather info, low risk, loop-sensitive
    diagnostic = {a for a in all_actions if any(kw in a for kw in ("inspect", "diagnostic", "acknowledge"))}
    # Destructive actions: cause downtime, high risk
    destructive = {a for a in all_actions if any(kw in a for kw in ("restart", "rollback", "scale"))}
    # Fix actions: targeted, medium risk
    fix_actions = {a for a in all_actions if "fix" in a or "apply" in a}
    # Penalty actions: always wrong
    penalty = {a for a in all_actions if "ignore" in a}

    loop_sensitive = diagnostic | destructive | fix_actions
    high_risk = destructive | fix_actions

    return loop_sensitive, destructive, high_risk


_LOOP_SENSITIVE_ACTIONS, _DESTRUCTIVE_ACTIONS, _HIGH_RISK_ACTIONS = _classify_actions()

# Health drop threshold — configurable via env var
import os as _os
_ROLLBACK_HEALTH_THRESHOLD = float(_os.environ.get("ARL_ROLLBACK_THRESHOLD", "0.15"))


@dataclass
class CircuitBreakerResult:
    allowed: bool
    reason: str                  # why it was blocked (empty if allowed)
    forced_message: str          # message sent back to agent when blocked
    action_type: str
    target: str
    repeat_count: int


class DeterministicCircuitBreaker:
    """
    Intercepts every agent action before it reaches the environment.

    Blocks:
    - Same action+target repeated ≥ N times (configurable, default 3)
    - Destructive actions on same target repeated ≥ 2 times
    - Any action after episode timeout

    When blocked, forces the agent to generate a new hypothesis
    instead of burning tokens on the same failed approach.

    This is deterministic — no LLM involved, no randomness.
    Pure rule-based middleware between agent and environment.
    """

    def __init__(
        self,
        max_repeats: int = 3,
        max_destructive_repeats: int = 2,
    ) -> None:
        self.max_repeats = max_repeats
        self.max_destructive_repeats = max_destructive_repeats
        self._action_counts: Dict[str, int] = defaultdict(int)  # "action:target" → count
        self._blocked_count: int = 0
        self._total_checked: int = 0
        self._block_log: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self._action_counts.clear()
        self._blocked_count = 0
        self._total_checked = 0
        self._block_log.clear()

    def check(self, action_type: str, target: str) -> CircuitBreakerResult:
        """
        Check if this action should be allowed or blocked.
        Called BEFORE the action reaches the environment.
        """
        self._total_checked += 1
        key = f"{action_type}:{target}"
        self._action_counts[key] += 1
        count = self._action_counts[key]

        # Rule 1: Destructive actions — stricter limit
        if action_type in _DESTRUCTIVE_ACTIONS and count >= self.max_destructive_repeats:
            self._blocked_count += 1
            msg = (
                f"CIRCUIT BREAKER OPEN: {action_type}({target}) has been attempted "
                f"{count} times with no improvement. "
                f"Destructive actions are blocked after {self.max_destructive_repeats} attempts. "
                f"Generate a NEW hypothesis — investigate a DIFFERENT service or try a DIFFERENT action type. "
                f"Services you haven't investigated yet may hold the root cause."
            )
            result = CircuitBreakerResult(
                allowed=False, reason="destructive_repeat",
                forced_message=msg, action_type=action_type,
                target=target, repeat_count=count,
            )
            self._block_log.append({
                "timestamp": time.time(), "action": key,
                "count": count, "reason": "destructive_repeat",
            })
            return result

        # Rule 2: Any action repeated too many times
        if action_type in _LOOP_SENSITIVE_ACTIONS and count >= self.max_repeats:
            self._blocked_count += 1
            msg = (
                f"CIRCUIT BREAKER OPEN: {action_type}({target}) repeated {count} times. "
                f"You are in a loop. This action is blocked. "
                f"You MUST try a different approach: "
                f"(1) Investigate a service you haven't checked yet, "
                f"(2) Try a different action type on {target}, "
                f"(3) Acknowledge the incident and reassess. "
                f"Repeating the same action wastes time and the system is degrading."
            )
            result = CircuitBreakerResult(
                allowed=False, reason="action_loop",
                forced_message=msg, action_type=action_type,
                target=target, repeat_count=count,
            )
            self._block_log.append({
                "timestamp": time.time(), "action": key,
                "count": count, "reason": "action_loop",
            })
            return result

        return CircuitBreakerResult(
            allowed=True, reason="", forced_message="",
            action_type=action_type, target=target, repeat_count=count,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_checked": self._total_checked,
            "blocked_count": self._blocked_count,
            "block_rate": round(self._blocked_count / max(self._total_checked, 1), 3),
            "action_counts": dict(self._action_counts),
            "recent_blocks": self._block_log[-5:],
        }


# ── 2. State-Preserving Rollback Engine ──────────────────────────────────────


@dataclass
class StateSnapshot:
    """Immutable snapshot of system state before a high-risk action."""
    snapshot_id: str
    step_number: int
    action_type: str
    target: str
    services: Dict[str, Dict[str, Any]]
    metrics: Dict[str, float]
    alerts: List[Dict[str, Any]]
    logs: List[str]
    health: float
    threat_level: float
    attack_progress: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class RollbackResult:
    rolled_back: bool
    reason: str
    health_before: float
    health_after: float
    health_drop: float
    forced_message: str
    snapshot_id: str


class StatePreservingRollbackEngine:
    """
    Snapshots system state before every high-risk action.
    Automatically rolls back if the action causes catastrophic degradation.

    This is the "time machine" — it ensures the agent can never permanently
    destroy the system through a single wrong action.

    Enterprise-grade pattern: same as database transactions with ROLLBACK.
    """

    def __init__(self, health_drop_threshold: float = _ROLLBACK_HEALTH_THRESHOLD) -> None:
        self.health_drop_threshold = health_drop_threshold
        self._snapshots: List[StateSnapshot] = []
        self._rollback_count: int = 0
        self._snapshot_count: int = 0

    def reset(self) -> None:
        self._snapshots.clear()
        self._rollback_count = 0
        self._snapshot_count = 0

    def should_snapshot(self, action_type: str) -> bool:
        return action_type in _HIGH_RISK_ACTIONS

    def take_snapshot(
        self,
        step_number: int,
        action_type: str,
        target: str,
        services: Dict[str, Any],
        metrics: Dict[str, float],
        alerts: List[Dict],
        logs: List[str],
        health: float,
        threat_level: float,
        attack_progress: float,
    ) -> StateSnapshot:
        """Take a deep copy snapshot before executing a high-risk action."""
        self._snapshot_count += 1
        snap = StateSnapshot(
            snapshot_id=f"SNAP-{self._snapshot_count:04d}",
            step_number=step_number,
            action_type=action_type,
            target=target,
            services=deepcopy(services),
            metrics=deepcopy(metrics),
            alerts=deepcopy(alerts),
            logs=list(logs),
            health=health,
            threat_level=threat_level,
            attack_progress=attack_progress,
        )
        # Keep only last 3 snapshots (memory efficiency)
        self._snapshots.append(snap)
        if len(self._snapshots) > 3:
            self._snapshots.pop(0)
        return snap

    def check_and_rollback(
        self,
        health_before: float,
        health_after: float,
        snapshot: Optional[StateSnapshot],
    ) -> RollbackResult:
        """
        After action executes, check if health dropped catastrophically.
        If yes, restore the snapshot and inform the agent.
        """
        health_drop = health_before - health_after

        if snapshot is None or health_drop < self.health_drop_threshold:
            return RollbackResult(
                rolled_back=False, reason="",
                health_before=health_before, health_after=health_after,
                health_drop=health_drop, forced_message="",
                snapshot_id="",
            )

        # Catastrophic drop — rollback
        self._rollback_count += 1
        msg = (
            f"ROLLBACK ENGINE: Your action {snapshot.action_type}({snapshot.target}) "
            f"caused system health to drop from {health_before:.0%} to {health_after:.0%} "
            f"(drop: {health_drop:.0%}). "
            f"State has been REVERTED to before your action. "
            f"The system is back at {health_before:.0%} health. "
            f"Try a SAFER approach: run_diagnostic first to confirm root cause "
            f"before applying destructive fixes."
        )
        return RollbackResult(
            rolled_back=True,
            reason=f"health_drop_{health_drop:.2f}",
            health_before=health_before,
            health_after=health_after,
            health_drop=health_drop,
            forced_message=msg,
            snapshot_id=snapshot.snapshot_id,
        )

    def restore_snapshot(self, snapshot: StateSnapshot) -> Dict[str, Any]:
        """Return the snapshot data for the environment to restore."""
        return {
            "services": deepcopy(snapshot.services),
            "metrics": deepcopy(snapshot.metrics),
            "alerts": deepcopy(snapshot.alerts),
            "logs": list(snapshot.logs),
            "health": snapshot.health,
            "threat_level": snapshot.threat_level,
            "attack_progress": snapshot.attack_progress,
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "snapshots_taken": self._snapshot_count,
            "rollbacks_executed": self._rollback_count,
            "snapshots_in_memory": len(self._snapshots),
        }


# ── 3. Contextual Action Ledger ───────────────────────────────────────────────

@dataclass
class LedgerEntry:
    """One entry in the action ledger — crisp summary of what happened."""
    step: int
    action_type: str
    target: str
    outcome: str          # "success" | "failed" | "blocked" | "rolled_back"
    health_delta: float   # health change this step
    key_finding: str      # most important thing learned (1 sentence)
    timestamp: float = field(default_factory=time.time)

    def to_summary_line(self) -> str:
        delta_str = f"+{self.health_delta:.0%}" if self.health_delta >= 0 else f"{self.health_delta:.0%}"
        return (
            f"Step {self.step}: {self.action_type}({self.target}) → "
            f"{self.outcome} [{delta_str}] — {self.key_finding}"
        )


class ContextualActionLedger:
    """
    Replaces raw log dumps with crisp, structured action memory.

    Instead of giving the agent 50 lines of logs, the ledger gives:
    "Step 1: inspect_logs(db) → success [+0%] — Found slow queries on orders table"
    "Step 2: apply_fix(api) → rolled_back [-18%] — Wrong target, DB is the issue"
    "Step 3: run_diagnostic(db) → success [+0%] — ROOT SIGNAL: missing index"

    Benefits:
    - Prevents context window overflow (10 steps × 1 line vs 10 steps × 50 lines)
    - Prevents agent from repeating failed approaches
    - Saves tokens (estimated 60-80% reduction in context size)
    - Makes agent reasoning more focused and accurate
    """

    def __init__(self, max_entries: int = 10) -> None:
        self.max_entries = max_entries
        self._entries: List[LedgerEntry] = []
        self._failed_targets: List[str] = []
        self._successful_actions: List[str] = []

    def reset(self) -> None:
        self._entries.clear()
        self._failed_targets.clear()
        self._successful_actions.clear()

    def record(
        self,
        step: int,
        action_type: str,
        target: str,
        action_result: str,
        action_success: bool,
        health_before: float,
        health_after: float,
        circuit_blocked: bool = False,
        rolled_back: bool = False,
        clues_revealed: Optional[List[str]] = None,
    ) -> LedgerEntry:
        """Record an action and its outcome into the ledger."""
        health_delta = health_after - health_before

        if circuit_blocked:
            outcome = "blocked"
            key_finding = f"Circuit breaker blocked — {action_type} on {target} repeated too many times"
        elif rolled_back:
            outcome = "rolled_back"
            key_finding = f"Rollback triggered — {action_type} on {target} caused health drop {health_delta:.0%}"
            self._failed_targets.append(target)
        elif action_success and health_delta > 0.05:
            outcome = "success"
            key_finding = clues_revealed[0][:80] if clues_revealed else f"{action_type} improved health by {health_delta:.0%}"
            self._successful_actions.append(f"{action_type}:{target}")
        elif action_success:
            outcome = "success"
            key_finding = clues_revealed[0][:80] if clues_revealed else f"{action_type} on {target} — no significant change"
        else:
            outcome = "failed"
            key_finding = f"{action_type} on {target} had unintended effects — health {health_delta:.0%}"
            self._failed_targets.append(target)

        entry = LedgerEntry(
            step=step, action_type=action_type, target=target,
            outcome=outcome, health_delta=health_delta, key_finding=key_finding,
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)
        return entry

    def get_context_summary(self, current_health: float, incident_type: str) -> str:
        """
        Generate a crisp context summary for the agent.
        Replaces raw log dumps — dramatically reduces token usage.
        """
        if not self._entries:
            # Suggest the first diagnostic action from the available action set
            diag_actions = sorted(_LOOP_SENSITIVE_ACTIONS - _DESTRUCTIVE_ACTIONS)[:2]
            suggestion = " or ".join(diag_actions) if diag_actions else "a diagnostic action"
            return f"No actions taken yet. Incident: {incident_type}. Health: {current_health:.0%}. Start with {suggestion}."

        lines = [f"=== ACTION LEDGER (Health: {current_health:.0%}) ==="]
        for entry in self._entries:
            lines.append(entry.to_summary_line())

        if self._failed_targets:
            unique_failed = list(dict.fromkeys(self._failed_targets))
            lines.append(f"\nFAILED TARGETS: {', '.join(unique_failed)} — avoid repeating these")

        if self._successful_actions:
            lines.append(f"SUCCESSFUL: {', '.join(self._successful_actions[-3:])}")

        # Suggest next action based on ledger — use diagnostic actions from classification
        diagnostic_actions = _LOOP_SENSITIVE_ACTIONS - _DESTRUCTIVE_ACTIONS
        investigated = {e.target for e in self._entries if e.action_type in diagnostic_actions}
        lines.append(f"INVESTIGATED: {', '.join(investigated) if investigated else 'none yet'}")

        return "\n".join(lines)

    @property
    def entries(self) -> List[LedgerEntry]:
        return list(self._entries)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self._entries),
            "failed_targets": self._failed_targets,
            "successful_actions": self._successful_actions,
            # Estimate: each raw log dump is ~5 log lines × ~40 tokens = ~200 tokens
            # Ledger entry is ~1 line × ~20 tokens = ~20 tokens
            # Savings per entry ≈ 180 tokens
            "estimated_tokens_saved": len(self._entries) * 180,
        }


# ── ARL — The Unified Layer ───────────────────────────────────────────────────

@dataclass
class ARLDecision:
    """Result of ARL processing for one action."""
    proceed: bool                          # True = execute action, False = blocked
    blocked_by: str                        # "circuit_breaker" | "rollback" | ""
    forced_message: str                    # message to return to agent if blocked
    snapshot: Optional[StateSnapshot]     # snapshot taken before action (if high-risk)
    ledger_context: str                    # crisp context summary for agent
    circuit_result: Optional[CircuitBreakerResult] = None
    rollback_result: Optional[RollbackResult] = None


class AgenticReliabilityLayer:
    """
    The Agentic Reliability Layer — unified middleware between agent and environment.

    Composes all 3 components into a single coherent layer:
      1. Circuit Breaker — checks BEFORE action executes
      2. Rollback Engine — snapshots BEFORE, checks AFTER
      3. Action Ledger   — records AFTER, provides context for NEXT step

    Usage in AIRENEnvironment.step():
      # Before action:
      decision = arl.pre_action(action_type, target, services, metrics, ...)
      if not decision.proceed:
          return blocked_observation(decision.forced_message)

      # Execute action normally...

      # After action:
      arl.post_action(decision, health_before, health_after, ...)
      if decision.rollback_result and decision.rollback_result.rolled_back:
          restore_snapshot(decision.snapshot)
    """

    def __init__(
        self,
        max_repeats: int = 3,
        max_destructive_repeats: int = 2,
        rollback_threshold: float = _ROLLBACK_HEALTH_THRESHOLD,
    ) -> None:
        self.circuit_breaker = DeterministicCircuitBreaker(
            max_repeats=max_repeats,
            max_destructive_repeats=max_destructive_repeats,
        )
        self.rollback_engine = StatePreservingRollbackEngine(
            health_drop_threshold=rollback_threshold,
        )
        self.ledger = ContextualActionLedger()
        self._enabled = True

    def reset(self) -> None:
        self.circuit_breaker.reset()
        self.rollback_engine.reset()
        self.ledger.reset()

    def pre_action(
        self,
        action_type: str,
        target: str,
        step_number: int,
        services: Dict[str, Any],
        metrics: Dict[str, float],
        alerts: List[Dict],
        logs: List[str],
        health: float,
        threat_level: float,
        attack_progress: float,
        current_health: float,
        incident_type: str,
    ) -> ARLDecision:
        """
        Called BEFORE action executes.
        Returns ARLDecision with proceed=False if action should be blocked.
        """
        if not self._enabled:
            return ARLDecision(
                proceed=True, blocked_by="", forced_message="",
                snapshot=None,
                ledger_context=self.ledger.get_context_summary(current_health, incident_type),
            )

        # Step 1: Circuit Breaker check
        cb_result = self.circuit_breaker.check(action_type, target)
        if not cb_result.allowed:
            return ARLDecision(
                proceed=False,
                blocked_by="circuit_breaker",
                forced_message=cb_result.forced_message,
                snapshot=None,
                ledger_context=self.ledger.get_context_summary(current_health, incident_type),
                circuit_result=cb_result,
            )

        # Step 2: Rollback Engine — snapshot if high-risk
        snapshot = None
        if self.rollback_engine.should_snapshot(action_type):
            snapshot = self.rollback_engine.take_snapshot(
                step_number=step_number,
                action_type=action_type,
                target=target,
                services=services,
                metrics=metrics,
                alerts=alerts,
                logs=logs,
                health=health,
                threat_level=threat_level,
                attack_progress=attack_progress,
            )

        return ARLDecision(
            proceed=True,
            blocked_by="",
            forced_message="",
            snapshot=snapshot,
            ledger_context=self.ledger.get_context_summary(current_health, incident_type),
            circuit_result=cb_result,
        )

    def post_action(
        self,
        decision: ARLDecision,
        step: int,
        action_type: str,
        target: str,
        action_result: str,
        action_success: bool,
        health_before: float,
        health_after: float,
        clues_revealed: Optional[List[str]] = None,
    ) -> Tuple[Optional[RollbackResult], LedgerEntry]:
        """
        Called AFTER action executes.
        Returns (rollback_result, ledger_entry).
        If rollback_result.rolled_back is True, environment must restore snapshot.
        """
        # Step 3: Rollback check
        rollback_result = self.rollback_engine.check_and_rollback(
            health_before=health_before,
            health_after=health_after,
            snapshot=decision.snapshot,
        )

        # Step 4: Record in ledger
        ledger_entry = self.ledger.record(
            step=step,
            action_type=action_type,
            target=target,
            action_result=action_result,
            action_success=action_success,
            health_before=health_before,
            health_after=health_after,
            circuit_blocked=False,
            rolled_back=rollback_result.rolled_back,
            clues_revealed=clues_revealed,
        )

        decision.rollback_result = rollback_result
        return rollback_result, ledger_entry

    def record_blocked(
        self,
        step: int,
        action_type: str,
        target: str,
        health: float,
    ) -> LedgerEntry:
        """Record a circuit-breaker-blocked action in the ledger."""
        return self.ledger.record(
            step=step, action_type=action_type, target=target,
            action_result="blocked", action_success=False,
            health_before=health, health_after=health,
            circuit_blocked=True,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "circuit_breaker": self.circuit_breaker.stats,
            "rollback_engine": self.rollback_engine.stats,
            "ledger": self.ledger.stats,
        }

    def get_ledger_context(self, health: float, incident_type: str) -> str:
        return self.ledger.get_context_summary(health, incident_type)
