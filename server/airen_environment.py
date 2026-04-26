# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
AIREN Environment — True RL-grade Core Logic.

What makes this a REAL RL environment (not just an evaluator):

1. AUTONOMOUS WORLD EVOLUTION — the system degrades every step even without
   agent action. Waiting costs health. Time pressure is real.

2. RICH ACTION FEEDBACK — inspect_logs/inspect_metrics return actual clues
   from the scenario. The agent learns to gather information before acting.

3. PROPER STATE MACHINE — threat_level and attack_progress evolve each step.
   Wrong actions increase both. Correct actions reduce both.

4. DENSE REWARD SIGNAL — every step returns a shaped reward, not just at
   episode end. Agent gets immediate feedback on each decision.

5. CASCADING FAILURES — wrong actions trigger wrong_action_effects AND
   increase attack_progress, which can trigger secondary cascades.

6. PARTIAL OBSERVABILITY — agent sees symptoms (services, metrics, logs,
   alerts, threat_level) but NOT root_cause or attack_progress internals.

7. MULTI-AGENT MODE — optional AttackerAgent runs each step, actively
   worsening the incident. Defender must outpace the attacker.

8. FULL TRACING — every step is traced with AgentOps-style structured logs,
   cost tracking, and anomaly detection.
"""

import os
import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import AIRENAction, AIRENObservation, AIRENState
    from .dynamic_generator import get_generator
    from .incident_engine import ALL_INCIDENT_TYPES, IncidentScenario, generate_incident
    from .llm_judge import get_judge
    from .reward import compute_reward
    from .trace_logger import (
        EpisodeTrace, StepTrace, TraceSession,
        detect_anomaly, tokens_to_usd,
    )
    from .attacker_agent import AttackerAgent
except ImportError:
    from models import AIRENAction, AIRENObservation, AIRENState
    from server.dynamic_generator import get_generator
    from server.incident_engine import ALL_INCIDENT_TYPES, IncidentScenario, generate_incident
    from server.llm_judge import get_judge
    from server.reward import compute_reward
    from server.trace_logger import (
        EpisodeTrace, StepTrace, TraceSession,
        detect_anomaly, tokens_to_usd,
    )
    from server.attacker_agent import AttackerAgent


# Inspect actions return real clues from the scenario — this is what makes
# the action space deep. Agent must gather info before acting.
_INSPECT_CLUES: Dict[str, Dict[str, List[str]]] = {
    "db_overload": {
        "inspect_logs":    ["[DB] SLOW QUERY detected: full table scan on orders (8.4s)", "[DB] Connection pool: 487/500 used"],
        "inspect_metrics": ["db_cpu_pct=95.0", "db_connections=487", "api_p99_latency_ms=2400"],
        "run_diagnostic":  ["ROOT SIGNAL: Missing index on orders.user_id — 2.3M rows scanned per query"],
    },
    "memory_leak": {
        "inspect_logs":    ["[WORKER] Buffer pool: 2.1GB (expected 512MB)", "[WORKER] GC pause: 450ms"],
        "inspect_metrics": ["worker_memory_pct=91.0", "worker_memory_growth_mb_per_min=2.1"],
        "run_diagnostic":  ["ROOT SIGNAL: Job handler not releasing buffers after completion"],
    },
    "bad_deployment": {
        "inspect_logs":    ["[PAYMENT] NullPointerException in PaymentHandler.process() line 247", "[DEPLOY] v2.4.1 deployed 20min ago"],
        "inspect_metrics": ["payment_error_rate=0.35", "deployment_age_minutes=20"],
        "run_diagnostic":  ["ROOT SIGNAL: v2.4.1 introduced null pointer in CardValidator — rollback recommended"],
    },
    "network_partition": {
        "inspect_logs":    ["[API] connect ECONNREFUSED db:5432", "[INFRA] Firewall rules updated 8min ago"],
        "inspect_metrics": ["api_error_rate=1.0", "db_connections=0", "network_packet_loss_pct=100"],
        "run_diagnostic":  ["ROOT SIGNAL: Firewall rule change blocked port 5432 — API cannot reach DB"],
    },
    "cache_stampede": {
        "inspect_logs":    ["[CACHE] 52,847 keys expired simultaneously", "[DB] Query rate: 12,400/s (normal: 1,200/s)"],
        "inspect_metrics": ["cache_hit_rate=0.12", "db_cpu_pct=88", "db_queries_per_sec=12400"],
        "run_diagnostic":  ["ROOT SIGNAL: TTL batch expiry causing thundering herd — warm cache and rate-limit DB"],
    },
}


class AIRENEnvironment(Environment):
    """
    AI Production Incident Response & Recovery Environment.

    True RL environment: the world evolves every step, actions have real
    consequences, and the agent must learn a multi-step policy.

    Episode flow:
      reset(incident_type=...) → initial degraded system state
      step(AIRENAction)         → world evolves + reward + next state
      state                     → full state including hidden root cause

    Optional multi-agent mode:
      multi_agent=True → AttackerAgent runs each step, actively worsening.
      aggression=0.0-1.0 → controls attacker intensity.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS = 10

    _LLM_MODEL: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    def __init__(
        self,
        multi_agent: bool = False,
        aggression: float = 0.5,
    ) -> None:
        super().__init__()
        self._state = AIRENState()
        self._scenario: Optional[IncidentScenario] = None
        self._services: Dict[str, Dict] = {}
        self._metrics: Dict[str, float] = {}
        self._alerts: List[Dict] = []
        self._logs_buffer: List[str] = []   # rolling log window
        self._episode_count = 0

        # RL state variables (evolve each step)
        self._threat_level: float = 0.5
        self._attack_progress: float = 0.0
        self._diagnostic_clues_revealed: List[str] = []  # what agent has learned

        # Multi-agent mode
        self._multi_agent = multi_agent
        self._attacker = AttackerAgent(aggression=aggression) if multi_agent else None

        # Trace / cost tracking
        self._trace_session = TraceSession()
        self._current_trace: Optional[EpisodeTrace] = None
        self._episode_tokens: int = 0

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        incident_type: Optional[str] = None,
        difficulty: str = "medium",
        **kwargs: Any,
    ) -> AIRENObservation:
        """
        Start a new incident response episode.

        The world starts in a degraded state. Every step the system gets
        worse unless the agent acts correctly. Time pressure is real.
        """
        if seed is None:
            seed = random.randint(0, 9999)
        if incident_type is None:
            incident_type = ALL_INCIDENT_TYPES[self._episode_count % len(ALL_INCIDENT_TYPES)]

        self._episode_count += 1
        self._episode_tokens = 0
        self._diagnostic_clues_revealed = []

        generator = get_generator()
        scenario = generator.generate(
            incident_type=incident_type,
            seed=seed,
            difficulty=difficulty,
        )
        self._scenario = scenario
        self._services = deepcopy(scenario.initial_services)
        self._metrics = deepcopy(scenario.initial_metrics)
        self._alerts = deepcopy(scenario.initial_alerts)
        self._logs_buffer = list(scenario.initial_logs)

        # Initial RL state
        self._threat_level = scenario.threat_level
        self._attack_progress = 0.0

        if self._attacker:
            self._attacker.reset()

        health = self._health()
        eid = episode_id or str(uuid4())

        self._state = AIRENState(
            episode_id=eid,
            step_count=0,
            incident_id=scenario.incident_id,
            incident_type=scenario.incident_type,
            root_cause=scenario.root_cause,
            steps_taken=0,
            actions_taken=[],
            correct_actions=0,
            wrong_actions=0,
            system_health_history=[health],
            incident_resolved=False,
            total_downtime_steps=0,
            cumulative_reward=0.0,
            threat_level=self._threat_level,
            attack_progress=self._attack_progress,
        )

        self._current_trace = self._trace_session.new_episode(
            episode_id=eid,
            incident_type=scenario.incident_type,
            severity=scenario.severity,
            difficulty=difficulty,
        )

        return AIRENObservation(
            incident_id=scenario.incident_id,
            incident_type=scenario.incident_type,
            severity=scenario.severity,
            step_number=0,
            max_steps=self.MAX_STEPS,
            services=deepcopy(self._services),
            metrics=deepcopy(self._metrics),
            logs=self._logs_buffer[-5:],   # partial observability: last 5 lines
            alerts=list(self._alerts),
            system_health=health,
            threat_level=round(self._threat_level, 3),
            attack_progress=round(self._attack_progress, 3),
            done=False,
            reward=None,
        )

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: AIRENAction, **kwargs: Any) -> AIRENObservation:
        """
        Execute one incident response action.

        World evolution order each step:
          1. Attacker acts (multi-agent mode)
          2. World degrades autonomously (time pressure)
          3. Agent action executes
          4. Cascading failures checked
          5. Reward computed (dense, multi-objective)
          6. LLM judge at episode end
        """
        if self._scenario is None:
            raise RuntimeError("Call reset() before step()")

        self._state.step_count += 1
        self._state.steps_taken += 1
        self._state.actions_taken.append(f"{action.action_type}:{action.target}")

        scenario = self._scenario
        health_before = self._health()
        threat_before = self._threat_level
        risk_before = self._metrics.get("risk_level", self._threat_level)

        if health_before < 0.5:
            self._state.total_downtime_steps += 1

        # ── 1. Attacker acts first ────────────────────────────────────────────
        attack_log: Optional[str] = None
        if self._attacker:
            attack = self._attacker.act(
                incident_type=scenario.incident_type,
                services=self._services,
                step=self._state.steps_taken,
            )
            self._services, new_logs, self._metrics = self._attacker.apply(
                attack, self._services, [], self._metrics
            )
            if new_logs:
                attack_log = new_logs[0]
                self._logs_buffer.append(attack_log)

        # ── 2. World degrades autonomously ────────────────────────────────────
        self._evolve_world(scenario)

        # ── 3. Agent action executes ──────────────────────────────────────────
        result_msg, success, action_logs = self._execute(action, scenario)
        for log_line in action_logs:
            self._logs_buffer.append(log_line)

        # ── 4. Cascading failures ─────────────────────────────────────────────
        cascade_log = self._check_cascades(scenario)
        if cascade_log:
            self._logs_buffer.append(cascade_log)

        # Check resolution
        resolved = self._is_resolved(action, scenario)
        if resolved and not self._state.incident_resolved:
            self._state.incident_resolved = True
            self._state.resolution_step = self._state.steps_taken
            self._recover(scenario)

        health_after = self._health()
        threat_after = self._threat_level
        risk_after = self._metrics.get("risk_level", self._threat_level)
        self._state.system_health_history.append(health_after)
        self._state.threat_level = self._threat_level
        self._state.attack_progress = self._attack_progress

        # Track correct/wrong
        if action.target in scenario.correct_targets and action.action_type in scenario.correct_actions:
            self._state.correct_actions += 1
        else:
            self._state.wrong_actions += 1

        # ── 5. Dense multi-objective reward ───────────────────────────────────
        rb = compute_reward(
            action_type=action.action_type,
            target=action.target,
            action_success=success,
            health_before=health_before,
            health_after=health_after,
            step_number=self._state.steps_taken,
            max_steps=self.MAX_STEPS,
            correct_targets=scenario.correct_targets,
            correct_actions=scenario.correct_actions,
            incident_resolved=resolved,
            total_downtime_steps=self._state.total_downtime_steps,
            threat_level_before=threat_before,
            threat_level_after=threat_after,
            active_alerts_count=len(self._alerts),
            wrong_actions_so_far=self._state.wrong_actions,
            severity=scenario.severity,
        )
        self._state.cumulative_reward += rb.total

        done = self._state.incident_resolved or self._state.steps_taken >= self.MAX_STEPS

        # ── 6. LLM judge at episode end ───────────────────────────────────────
        judge_result = None
        judge_tokens = 0
        if done:
            judge = get_judge()
            judge_result = judge.judge(
                incident_type=scenario.incident_type,
                root_cause=scenario.root_cause,
                actions_taken=self._state.actions_taken,
                final_health=health_after,
                incident_resolved=self._state.incident_resolved,
                correct_actions=scenario.correct_actions,
                correct_targets=scenario.correct_targets,
                rule_score=rb.total,
            )
            judge_tokens = judge_result.tokens_used

        # ── Cost tracking ─────────────────────────────────────────────────────
        step_tokens = judge_tokens
        self._episode_tokens += step_tokens
        step_cost = tokens_to_usd(step_tokens, self._LLM_MODEL)

        # ── Anomaly detection ─────────────────────────────────────────────────
        anomaly = detect_anomaly(
            action.action_type,
            action.target,
            self._state.actions_taken[:-1],
            scenario.correct_targets,
        )

        # ── Trace ─────────────────────────────────────────────────────────────
        if self._current_trace is not None:
            step_trace = StepTrace(
                episode_id=self._state.episode_id,
                step=self._state.steps_taken,
                timestamp=time.time(),
                action_type=action.action_type,
                target=action.target,
                reasoning=action.reasoning or "",
                health_before=health_before,
                risk_before=risk_before,
                health_after=health_after,
                risk_after=risk_after,
                reward_total=rb.total,
                recovery_score=rb.recovery_score,
                diagnosis_score=rb.diagnosis_score,
                efficiency_score=rb.efficiency_score,
                threat_mitigation=rb.threat_mitigation,
                hallucination_penalty=rb.hallucination_penalty,
                security_violation_penalty=rb.security_violation_penalty,
                cost_penalty=rb.cost_penalty,
                downtime_penalty=rb.downtime_penalty,
                resolve_bonus=rb.resolve_bonus,
                llm_tokens_used=step_tokens,
                llm_cost_usd=step_cost,
                action_success=success,
                incident_resolved=resolved,
                anomaly=anomaly,
            )
            self._current_trace.add_step(step_trace)
            if done:
                self._current_trace.close(
                    resolved=self._state.incident_resolved,
                    final_health=health_after,
                    diagnosis_quality=judge_result.diagnosis_quality if judge_result else "unknown",
                )

        # Build metadata dict (test compatibility)
        metadata = {
            "judge_used": judge_result.judge_used if judge_result else None,
            "diagnosis_quality": judge_result.diagnosis_quality if judge_result else None,
            "anomaly": anomaly,
            "cost_usd": step_cost,
            "tokens": step_tokens,
            "attack_progress": round(self._attack_progress, 3),
            "clues_revealed": len(self._diagnostic_clues_revealed),
        }

        return AIRENObservation(
            incident_id=scenario.incident_id,
            incident_type=scenario.incident_type,
            severity=scenario.severity,
            step_number=self._state.steps_taken,
            max_steps=self.MAX_STEPS,
            services=deepcopy(self._services),
            metrics=deepcopy(self._metrics),
            logs=self._logs_buffer[-5:],   # partial observability: last 5 lines
            alerts=self._alerts if not resolved else [],
            system_health=health_after,
            threat_level=round(self._threat_level, 3),
            attack_progress=round(self._attack_progress, 3),
            action_result=result_msg,
            action_success=success,
            recovery_score=rb.recovery_score,
            diagnosis_score=rb.diagnosis_score,
            efficiency_score=rb.efficiency_score,
            threat_mitigation=rb.threat_mitigation,
            hallucination_penalty=rb.hallucination_penalty,
            security_violation_penalty=rb.security_violation_penalty,
            cost_penalty=rb.cost_penalty,
            downtime_penalty=rb.downtime_penalty,
            resolve_bonus=rb.resolve_bonus,
            incident_resolved=self._state.incident_resolved,
            cumulative_reward=round(self._state.cumulative_reward, 3),
            correct_actions_count=self._state.correct_actions,
            wrong_actions_count=self._state.wrong_actions,
            reward_explanation=rb.explanation,
            judge_used=judge_result.judge_used if judge_result else None,
            diagnosis_quality=judge_result.diagnosis_quality if judge_result else None,
            judge_reasoning=judge_result.reasoning_feedback if judge_result else None,
            metadata=metadata,
            done=done,
            reward=judge_result.final_score if judge_result else rb.total,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> AIRENState:
        return self._state

    @property
    def trace(self) -> Optional[EpisodeTrace]:
        return self._current_trace

    @property
    def cost_summary(self) -> Dict[str, Any]:
        return {
            "episode_tokens": self._episode_tokens,
            "episode_cost_usd": tokens_to_usd(self._episode_tokens, self._LLM_MODEL),
            "model": self._LLM_MODEL,
        }

    @property
    def session_summary(self) -> Dict[str, Any]:
        return self._trace_session.session_summary()

    # ── World evolution (THE KEY RL MECHANIC) ─────────────────────────────────

    def _evolve_world(self, scenario: IncidentScenario) -> None:
        """
        Autonomous world degradation — runs every step regardless of agent action.

        This is what makes it a real RL environment: the world gets worse
        over time. The agent must act, not just observe.
        """
        rate = scenario.degradation_rate

        # Threat level grows each step (incident spreads)
        self._threat_level = min(1.0, self._threat_level + rate * random.uniform(0.5, 1.5))

        # Attack progress grows (incident worsens)
        self._attack_progress = min(5.0, self._attack_progress + random.uniform(0.1, 0.3))

        # Services degrade based on incident type
        for name, svc in self._services.items():
            if svc.get("status") in ("degraded", "down"):
                svc["error_rate"] = min(1.0, svc["error_rate"] + rate * 0.5)
                svc["latency_ms"] = int(svc["latency_ms"] * (1 + rate * 0.3))
                svc["cpu_pct"] = min(100, svc.get("cpu_pct", 50) + random.randint(0, 3))

        # Update metrics to reflect degradation
        self._metrics["threat_level"] = round(self._threat_level, 3)
        self._metrics["attack_progress"] = round(self._attack_progress, 3)

    def _check_cascades(self, scenario: IncidentScenario) -> Optional[str]:
        """
        Check if attack_progress has crossed the cascade threshold.
        If so, trigger secondary failures on healthy services.
        """
        if self._attack_progress < scenario.cascade_threshold:
            return None

        # Find a healthy service to cascade into
        healthy = [
            name for name, svc in self._services.items()
            if svc.get("status") == "healthy"
        ]
        if not healthy:
            return None

        target = random.choice(healthy)
        self._services[target]["status"] = "degraded"
        self._services[target]["error_rate"] = min(1.0, 0.15 + random.uniform(0, 0.1))
        self._services[target]["latency_ms"] = int(self._services[target]["latency_ms"] * 2)

        # Add a new alert
        self._alerts.append({
            "service": target,
            "severity": "high",
            "message": f"Cascading failure: {target} degraded due to incident spread",
            "triggered_at": time.time(),
        })

        return f"[SYSTEM] CASCADE: {target} degraded — incident spreading (progress={self._attack_progress:.1f})"

    # ── Action execution (RICH FEEDBACK) ──────────────────────────────────────

    def _execute(
        self, action: AIRENAction, scenario: IncidentScenario
    ) -> Tuple[str, bool, List[str]]:
        """
        Execute action and return (result_msg, success, new_log_lines).

        Inspect actions return real diagnostic clues — this is what makes
        the action space deep. Agent learns to gather info before acting.
        """
        atype, target = action.action_type, action.target
        new_logs: List[str] = []

        # ── Inspect actions: return real clues ────────────────────────────────
        if atype in ("inspect_logs", "inspect_metrics", "run_diagnostic"):
            clues = _INSPECT_CLUES.get(scenario.incident_type, {}).get(atype, [])
            if clues:
                clue = random.choice(clues)
                self._diagnostic_clues_revealed.append(clue)
                new_logs.append(f"[{atype.upper()}:{target.upper()}] {clue}")
                # Reduce threat slightly — gathering info is productive
                self._threat_level = max(0.0, self._threat_level - 0.02)
                return f"Diagnostic: {clue[:80]}", True, new_logs
            new_logs.append(f"[{atype.upper()}:{target.upper()}] No additional signals on {target}")
            return f"Gathered information via {atype} on {target}", True, new_logs

        if atype == "acknowledge_incident":
            new_logs.append(f"[INCIDENT] Acknowledged by agent — tracking started")
            self._threat_level = max(0.0, self._threat_level - 0.05)
            return "Incident acknowledged", True, new_logs

        # ── Correct action → system improves ─────────────────────────────────
        if target in scenario.correct_targets and atype in scenario.correct_actions:
            for name, svc in self._services.items():
                if name == target or target in ("network", "infra"):
                    svc["error_rate"] = max(0.0, svc["error_rate"] * 0.4)
                    svc["latency_ms"] = max(50, int(svc["latency_ms"] * 0.5))
                    if svc["status"] in ("down", "degraded"):
                        svc["status"] = "recovering"
            # Correct action reduces threat and attack progress
            self._threat_level = max(0.0, self._threat_level - 0.25)
            self._attack_progress = max(0.0, self._attack_progress - 1.0)
            new_logs.append(f"[{target.upper()}] {atype} executed — system improving")
            return f"{atype} on {target} executed — system improving", True, new_logs
        # ── Wrong action → cascading side effects ─────────────────────────────
        if atype in scenario.wrong_action_effects:
            for svc_name, changes in scenario.wrong_action_effects[atype].items():
                if svc_name in self._services:
                    self._services[svc_name].update(changes)
            # Wrong action increases threat and attack progress
            self._threat_level = min(1.0, self._threat_level + 0.15)
            self._attack_progress = min(5.0, self._attack_progress + 0.5)
            new_logs.append(f"[{target.upper()}] {atype} had unintended side effects — situation worsening")
            return f"{atype} on {target} had unintended effects", False, new_logs

        # ── Neutral action ────────────────────────────────────────────────────
        new_logs.append(f"[{target.upper()}] {atype} completed — no significant change")
        return f"{atype} on {target} had no effect", False, new_logs

    # ── Resolution logic ──────────────────────────────────────────────────────

    def _is_resolved(self, action: AIRENAction, scenario: IncidentScenario) -> bool:
        """
        Incident is resolved when:
        1. Agent took the correct action on the correct target
        2. For multi-action scenarios: agent has taken at least 1 diagnostic step first
           (single-action scenarios like bad_deployment can resolve immediately)
        3. Threat level has been reduced below 0.8

        The diagnostic requirement only applies when the scenario itself requires
        multiple steps (len(correct_actions) > 1). Single-action scenarios like
        bad_deployment (just rollback_deployment) can resolve on step 1.
        """
        correct_action = action.action_type in scenario.correct_actions
        correct_target = action.target in scenario.correct_targets
        threat_manageable = self._threat_level < 0.8

        # Only require a prior diagnostic step for multi-action scenarios
        requires_diagnosis = len(scenario.correct_actions) > 1
        if requires_diagnosis:
            has_diagnosed = any(
                a.split(":")[0] in ("inspect_logs", "inspect_metrics", "run_diagnostic")
                for a in self._state.actions_taken[:-1]  # exclude current action
            )
        else:
            has_diagnosed = True  # single-action scenarios resolve immediately

        return correct_action and correct_target and has_diagnosed and threat_manageable

    def _recover(self, scenario: IncidentScenario) -> None:
        """Restore all services to healthy state after resolution."""
        for name in self._services:
            self._services[name].update({
                "status": "healthy",
                "error_rate": 0.01,
                "latency_ms": random.randint(40, 100),
                "cpu_pct": random.randint(25, 50),
            })
        self._alerts = []
        self._threat_level = 0.0
        self._attack_progress = 0.0
        self._metrics["threat_level"] = 0.0
        self._metrics["attack_progress"] = 0.0

    # ── Health calculation ────────────────────────────────────────────────────

    def _health(self) -> float:
        if not self._services:
            return 0.0
        scores = []
        for s in self._services.values():
            st = s.get("status", "healthy")
            er = s.get("error_rate", 0.0)
            if st == "down":
                scores.append(0.0)
            elif st == "degraded":
                scores.append(max(0.05, 0.5 - er * 0.5))
            elif st == "recovering":
                scores.append(0.75)
            else:
                scores.append(max(0.5, 1.0 - er * 2))
        return round(sum(scores) / len(scores), 3)
