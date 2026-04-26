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

7. MULTI-AGENT MODE — 3 concurrent agents: AttackerAgent (adversarial),
   MonitoringAgent (cooperative, 75% accuracy), AutoScalerAgent (autonomous).
   Defender must outpace attacker, coordinate with monitoring, account for scaler.

8. FAILURE + RECOVERY — wrong fix → worse state → agent must recover.
   Tracked via wrong_fixes_applied / recovery_attempts. Extra reward for recovery.

9. MULTI-HYPOTHESIS EXPLORATION — exploration bonus for testing multiple
   services before acting. Rewards real SRE behavior (not shortcut solving).

10. 9 INCIDENT TYPES — bad_deployment, db_overload, memory_leak,
    network_partition, cache_stampede, api_timeout, disk_full,
    ssl_cert_expired, ddos_attack. Easy → medium → hard curriculum.

11. FULL TRACING — every step is traced with AgentOps-style structured logs,
    cost tracking, and anomaly detection.
"""

import os
import random
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


def _clamp_score(score: float) -> float:
    return max(0.001, min(0.999, round(score, 3)))

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
    from .attacker_agent import AttackerAgent, MonitoringAgent, AutoScalerAgent
    from .digital_twin import enrich_observation, k8s_pod_state
    from .arl import AgenticReliabilityLayer, ARLDecision
    from .compliance_enforcer import StructuralComplianceEnforcer
    # Self-evolving upgrades
    from .self_evolving_curriculum import get_curriculum
    from .reward_hacking_detector import get_hack_detector
    from .agent_deadlock_detector import get_deadlock_detector
    from .persistent_agent_memory import get_agent_memory
    from .self_reward_judge import get_self_reward_judge
    from .state_drift_monitor import get_drift_monitor
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
    from server.attacker_agent import AttackerAgent, MonitoringAgent, AutoScalerAgent
    from server.digital_twin import enrich_observation, k8s_pod_state
    from server.arl import AgenticReliabilityLayer, ARLDecision
    from server.compliance_enforcer import StructuralComplianceEnforcer
    # Self-evolving upgrades
    from server.self_evolving_curriculum import get_curriculum
    from server.reward_hacking_detector import get_hack_detector
    from server.agent_deadlock_detector import get_deadlock_detector
    from server.persistent_agent_memory import get_agent_memory
    from server.self_reward_judge import get_self_reward_judge
    from server.state_drift_monitor import get_drift_monitor


def _build_inspect_clues(scenario: "IncidentScenario") -> Dict[str, Dict[str, List[str]]]:
    """
    Build inspect clues DYNAMICALLY from the scenario's actual metrics and logs.
    No hardcoded values — every clue reflects the real numbers in this episode.
    """
    itype   = scenario.incident_type
    metrics = scenario.initial_metrics
    logs    = scenario.initial_logs
    rc      = scenario.root_cause

    # Pull real log lines that contain actual symptoms (not red herrings)
    real_logs = [l for l in logs if "red herring" not in l.lower()
                 and any(kw in l.upper() for kw in
                         ["ERROR","SLOW","WARN","CRITICAL","FAIL","TIMEOUT",
                          "EXPIR","FLOOD","LEAK","FULL","PARTITION","STAMPEDE",
                          "OVERFLOW","REFUSED","EXPIRED","ATTACK","MISS"])]
    # Fallback: first 3 logs
    if not real_logs:
        real_logs = logs[:3]

    # Build metric summary from actual scenario metrics
    metric_lines = [f"{k}={v}" for k, v in list(metrics.items())[:4]]

    # Clue templates keyed by incident type — values come from scenario
    clue_map: Dict[str, Dict[str, List[str]]] = {}

    if itype == "db_overload":
        db_cpu  = metrics.get("db_cpu_pct", 95)
        db_conn = int(metrics.get("db_connections", 487))
        lat     = int(metrics.get("api_p99_latency_ms", 2400))
        clue_map[itype] = {
            "inspect_logs":    real_logs[:2] or [f"[DB] SLOW QUERY detected ({db_cpu:.0f}% CPU)"],
            "inspect_metrics": [f"db_cpu_pct={db_cpu}", f"db_connections={db_conn}",
                                f"api_p99_latency_ms={lat}"],
            "run_diagnostic":  [f"ROOT SIGNAL: {rc}"],
        }

    elif itype == "memory_leak":
        mem  = metrics.get("worker_memory_pct", 91)
        grow = metrics.get("worker_memory_growth_mb_per_min", 2.1)
        clue_map[itype] = {
            "inspect_logs":    real_logs[:2] or [f"[WORKER] Memory {mem:.0f}% — growing"],
            "inspect_metrics": [f"worker_memory_pct={mem}", f"growth_mb_per_min={grow}"],
            "run_diagnostic":  [f"ROOT SIGNAL: {rc}"],
        }

    elif itype == "bad_deployment":
        pay_err  = metrics.get("payment_error_rate", 0.35)
        dep_age  = metrics.get("deployment_age_minutes", 20)
        clue_map[itype] = {
            "inspect_logs":    real_logs[:2] or [f"[PAYMENT] Error rate {pay_err:.0%}"],
            "inspect_metrics": [f"payment_error_rate={pay_err}",
                                f"deployment_age_minutes={dep_age}"],
            "run_diagnostic":  [f"ROOT SIGNAL: {rc}"],
        }

    elif itype == "network_partition":
        failures = int(metrics.get("api_db_connection_failures", 847))
        clue_map[itype] = {
            "inspect_logs":    real_logs[:2] or ["[API] connect ECONNREFUSED db:5432"],
            "inspect_metrics": [f"api_error_rate=1.0", f"db_connections=0",
                                f"api_db_connection_failures={failures}"],
            "run_diagnostic":  [f"ROOT SIGNAL: {rc}"],
        }

    elif itype == "cache_stampede":
        hit  = metrics.get("cache_hit_rate", 0.12)
        qps  = int(metrics.get("db_queries_per_sec", 12400))
        cpu  = metrics.get("db_cpu_pct", 88)
        clue_map[itype] = {
            "inspect_logs":    real_logs[:2] or [f"[CACHE] Mass expiry — hit rate {hit:.0%}"],
            "inspect_metrics": [f"cache_hit_rate={hit}", f"db_queries_per_sec={qps}",
                                f"db_cpu_pct={cpu}"],
            "run_diagnostic":  [f"ROOT SIGNAL: {rc}"],
        }

    elif itype == "api_timeout":
        waiting = int(metrics.get("connection_pool_waiting", 48))
        pool_max = int(metrics.get("connection_pool_max", 50))
        clue_map[itype] = {
            "inspect_logs":    real_logs[:2] or ["[API] Timeout waiting for upstream"],
            "inspect_metrics": [f"connection_pool_waiting={waiting}/{pool_max}",
                                "upstream_timeout_rate=1.0"],
            "run_diagnostic":  [f"ROOT SIGNAL: {rc}"],
        }

    elif itype == "disk_full":
        disk_pct = metrics.get("disk_usage_pct", 97)
        log_gb   = metrics.get("log_dir_size_gb", 185)
        hours    = metrics.get("log_rotation_last_success_hours_ago", 48)
        clue_map[itype] = {
            "inspect_logs":    real_logs[:2] or [f"[INFRA] Disk {disk_pct:.0f}% full"],
            "inspect_metrics": [f"disk_usage_pct={disk_pct}",
                                f"log_dir_size_gb={log_gb}",
                                f"log_rotation_last_success_hours_ago={hours}"],
            "run_diagnostic":  [f"ROOT SIGNAL: {rc}"],
        }

    elif itype == "ssl_cert_expired":
        failures = int(metrics.get("tls_handshake_failures", 2400))
        hours    = metrics.get("cert_expiry_hours_ago", 2.0)
        clue_map[itype] = {
            "inspect_logs":    real_logs[:2] or ["[NGINX] SSL certificate expired"],
            "inspect_metrics": [f"tls_handshake_failures={failures}",
                                f"cert_expiry_hours_ago={hours}"],
            "run_diagnostic":  [f"ROOT SIGNAL: {rc}"],
        }

    elif itype == "ddos_attack":
        rps     = int(metrics.get("api_requests_per_sec", 85000))
        ips     = int(metrics.get("unique_attacker_ips", 8000))
        bw      = metrics.get("network_bandwidth_gbps", 9.8)
        clue_map[itype] = {
            "inspect_logs":    real_logs[:2] or [f"[SECURITY] {ips:,} unique IPs flooding"],
            "inspect_metrics": [f"api_requests_per_sec={rps:,}",
                                f"unique_attacker_ips={ips:,}",
                                f"network_bandwidth_gbps={bw}"],
            "run_diagnostic":  [f"ROOT SIGNAL: {rc}"],
        }

    else:
        # Unknown / LLM-generated type — derive clues from scenario data
        clue_map[itype] = {
            "inspect_logs":    real_logs[:2] or [f"[SYSTEM] Anomaly detected in {itype}"],
            "inspect_metrics": metric_lines or ["metrics unavailable"],
            "run_diagnostic":  [f"ROOT SIGNAL: {rc}"],
        }

    return clue_map


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
    MAX_STEPS = 10  # instance-level override happens in reset() per difficulty

    # Per-difficulty step budgets — configurable via env vars, no hardcoded magic
    _DIFFICULTY_STEPS: Dict[str, int] = {
        "easy":   int(os.environ.get("MAX_STEPS_EASY",   "12")),
        "medium": int(os.environ.get("MAX_STEPS_MEDIUM", "10")),
        "hard":   int(os.environ.get("MAX_STEPS_HARD",   "8")),
    }

    _LLM_MODEL: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    def __init__(
        self,
        multi_agent: bool = False,
        aggression: float = 0.5,
        monitoring_accuracy: float = 0.75,
        autoscaler_aggressiveness: float = 0.6,
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
        self._hypotheses_tested: List[str] = []          # services agent has investigated
        self._wrong_fixes_applied: int = 0               # failure+recovery tracking
        self._recovery_attempts: int = 0                 # how many times agent recovered

        # Multi-agent mode
        self._multi_agent = multi_agent
        self._attacker = AttackerAgent(aggression=aggression) if multi_agent else None
        self._monitor = MonitoringAgent(accuracy=monitoring_accuracy) if multi_agent else None
        self._autoscaler = AutoScalerAgent(aggressiveness=autoscaler_aggressiveness) if multi_agent else None

        # Trace / cost tracking
        self._trace_session = TraceSession()
        self._current_trace: Optional[EpisodeTrace] = None
        self._episode_tokens: int = 0
        # Digital twin — Kubernetes/Prometheus-style enrichment
        self._topology_info: Dict[str, Any] = {}
        self._k8s_pods: Dict[str, Any] = {}
        self._digital_twin_enabled: bool = os.environ.get("DIGITAL_TWIN", "1") == "1"

        # Agentic Reliability Layer — Circuit Breaker + Rollback + Ledger
        self._arl = AgenticReliabilityLayer(
            max_repeats=int(os.environ.get("ARL_MAX_REPEATS", "3")),
            max_destructive_repeats=int(os.environ.get("ARL_MAX_DESTRUCTIVE", "2")),
            rollback_threshold=float(os.environ.get("ARL_ROLLBACK_THRESHOLD", "0.15")),
        )
        self._arl_enabled: bool = os.environ.get("ARL_ENABLED", "1") == "1"

        # Structural Compliance Enforcer — EU AI Act, PCI-DSS, SOC2, HIPAA
        self._compliance = StructuralComplianceEnforcer()
        self._compliance_enabled: bool = os.environ.get("COMPLIANCE_ENABLED", "1") == "1"

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
        self._hypotheses_tested = []
        self._wrong_fixes_applied = 0
        self._recovery_attempts = 0
        self._last_seed = seed  # for replay forensics

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

        # Per-difficulty step budget — hard incidents degrade faster, need fewer steps
        # to keep time pressure real; easy incidents get more steps to allow exploration.
        self.MAX_STEPS = self._DIFFICULTY_STEPS.get(scenario.difficulty, 10)

        # ── Digital twin enrichment — Prometheus/K8s-style metrics ───────────
        if self._digital_twin_enabled:
            # Derive RNG from scenario content — no seed dependency
            rng_seed = hash(scenario.incident_id + scenario.root_cause) % (2**31)
            rng = random.Random(rng_seed)
            enriched_metrics, enriched_alerts, topology = enrich_observation(
                rng=rng,
                incident_type=scenario.incident_type,
                services=self._services,
                base_metrics=self._metrics,
                base_alerts=self._alerts,
            )
            self._metrics = enriched_metrics
            self._alerts = enriched_alerts
            self._topology_info = topology
            # Generate K8s pod states for each service
            self._k8s_pods = {
                svc: [p.__dict__ for p in k8s_pod_state(rng, svc, data)]
                for svc, data in self._services.items()
            }            # Add K8s pod restart counts to logs
            for svc, pods in self._k8s_pods.items():
                total_restarts = sum(p.get("restarts", 0) for p in pods)
                if total_restarts > 0:
                    self._logs_buffer.append(
                        f"[K8S] {svc}: {len(pods)} pods, {total_restarts} total restarts"
                    )

        # Initial RL state
        self._threat_level = scenario.threat_level
        self._attack_progress = 0.0

        if self._attacker:
            self._attacker.reset()
        if self._monitor:
            self._monitor.reset()
        if self._autoscaler:
            self._autoscaler.reset()
        self._arl.reset()
        self._compliance.reset()

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

        # ── Observability: track episode start ────────────────────────────────
        try:
            from .observability import get_observability_dashboard
            get_observability_dashboard().track_episode_start(
                incident_type=scenario.incident_type,
                severity=scenario.severity,
            )
        except Exception:
            pass

        # ── Persistent Memory: retrieve past experience ───────────────────────
        try:
            memory_result = get_agent_memory().retrieve_for_incident(
                incident_type=scenario.incident_type,
                current_health=health,
            )
            if memory_result.context_hint:
                self._logs_buffer.append(memory_result.context_hint)
        except Exception:
            pass

        # ── Reward Hacking Detector: reset episode ────────────────────────────
        try:
            get_hack_detector().reset_episode(eid)
        except Exception:
            pass

        # ── Deadlock Detector: reset ──────────────────────────────────────────
        try:
            get_deadlock_detector().reset()
        except Exception:
            pass

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

        # ── ARL Pre-Action: Circuit Breaker + Snapshot ────────────────────────
        arl_decision = None
        compliance_decision = None

        # ── Compliance check (runs before ARL) ───────────────────────────────
        if self._compliance_enabled:
            compliance_decision = self._compliance.check(
                action_type=action.action_type,
                target=action.target,
                reasoning=action.reasoning or "",
                step_number=self._state.steps_taken,
                prior_actions=self._state.actions_taken[:-1],
                incident_type=scenario.incident_type,
                severity=scenario.severity,
            )
            if not compliance_decision.allowed:
                blocked_v = [v for v in compliance_decision.violations if v.blocked]
                block_msg = blocked_v[0].reason if blocked_v else "Compliance violation"
                self._logs_buffer.append(f"[COMPLIANCE] BLOCKED: {block_msg[:100]}")
                # Track in observability
                try:
                    from .observability import get_observability_dashboard
                    for v in compliance_decision.violations:
                        get_observability_dashboard().track_compliance_violation(
                            v.framework, v.severity, v.blocked
                        )
                except Exception:
                    pass
                from .reward import compute_reward as _cr
                rb_comp = _cr(
                    action_type=action.action_type, target=action.target,
                    action_success=False, health_before=health_before,
                    health_after=health_before, step_number=self._state.steps_taken,
                    max_steps=self.MAX_STEPS, correct_targets=scenario.correct_targets,
                    correct_actions=scenario.correct_actions, incident_resolved=False,
                    total_downtime_steps=self._state.total_downtime_steps,
                    wrong_actions_so_far=self._state.wrong_actions,
                )
                self._state.wrong_actions += 1
                self._state.cumulative_reward += rb_comp.total
                self._state.wrong_fixes_applied = self._wrong_fixes_applied
                self._state.recovery_attempts = self._recovery_attempts
                self._state.hypotheses_tested = len(self._hypotheses_tested)
                return AIRENObservation(
                    incident_id=scenario.incident_id,
                    incident_type=scenario.incident_type,
                    severity=scenario.severity,
                    step_number=self._state.steps_taken,
                    max_steps=self.MAX_STEPS,
                    services=deepcopy(self._services),
                    metrics=deepcopy(self._metrics),
                    logs=self._logs_buffer[-5:] + [f"[COMPLIANCE] {block_msg[:120]}"],
                    alerts=self._alerts,
                    system_health=health_before,
                    threat_level=round(self._threat_level, 3),
                    attack_progress=round(self._attack_progress, 3),
                    action_result=f"COMPLIANCE BLOCKED: {block_msg[:120]}",
                    action_success=False,
                    recovery_score=0.0, diagnosis_score=0.0,
                    efficiency_score=0.0, threat_mitigation=0.0,
                    hallucination_penalty=0.0, security_violation_penalty=0.2,
                    cost_penalty=0.0, downtime_penalty=0.0,
                    resolve_bonus=0.0, exploration_bonus=0.0, recovery_bonus=0.0,
                    reward_explanation=f"Compliance blocked: {blocked_v[0].framework if blocked_v else 'unknown'}",
                    metadata={
                        "compliance_blocked": True,
                        "compliance_violations": [
                            {"framework": v.framework, "article": v.article, "reason": v.reason[:100]}
                            for v in compliance_decision.violations
                        ],
                        "compliance_stats": self._compliance.stats,
                        "arl_blocked": False,
                        "arl_stats": self._arl.stats if self._arl_enabled else {},
                        "ledger_context": self._arl.get_ledger_context(health_before, scenario.incident_type) if self._arl_enabled else "",
                        "wrong_fixes_applied": self._wrong_fixes_applied,
                        "recovery_attempts": self._recovery_attempts,
                        "hypotheses_tested": len(self._hypotheses_tested),
                        "multi_agent_active": self._multi_agent,
                        "monitoring_signals": 0,
                        "autoscaler_actions": 0,
                        "topology": self._topology_info,
                        "k8s_pods": self._k8s_pods,
                        "digital_twin_enabled": self._digital_twin_enabled,
                    },
                    done=False,
                    reward=rb_comp.total,
                )

        if self._arl_enabled:
            arl_decision = self._arl.pre_action(
                action_type=action.action_type,
                target=action.target,
                step_number=self._state.steps_taken,
                services=self._services,
                metrics=self._metrics,
                alerts=self._alerts,
                logs=self._logs_buffer,
                health=health_before,
                threat_level=self._threat_level,
                attack_progress=self._attack_progress,
                current_health=health_before,
                incident_type=scenario.incident_type,
            )

        # ── Upgrade #2: Infinite Loop Detection ──────────────────────────────
        try:
            from .infinite_loop_detector import get_loop_breaker
            loop_breaker = get_loop_breaker()
            loop_detection = loop_breaker.detect_loop(
                action.action_type, action.parameters or {}, 0
            )
            if loop_detection.loop_detected and loop_detection.recommendation in ("BREAK_LOOP", "CIRCUIT_OPEN"):
                loop_action = loop_breaker.break_loop(
                    loop_detection, self._state.steps_taken, self.MAX_STEPS
                )
                self._logs_buffer.append(
                    f"[LOOP_BREAKER] {loop_action.reason}: {loop_action.suggestion[:80]}"
                )
                # Track in observability
                try:
                    from .observability import get_observability_dashboard
                    get_observability_dashboard().track_loop_detection(
                        loop_detection.pattern, loop_action.estimated_tokens_saved
                    )
                    get_observability_dashboard().track_arl_event("circuit_break")
                except Exception:
                    pass
            loop_breaker.record_action(action.action_type, action.parameters or {})
        except Exception:
            pass

        # ── Upgrade #6: Context Poisoning Scan ───────────────────────────────
        if action.reasoning:
            try:
                from .context_poisoning_detector import get_context_defender
                ctx_defender = get_context_defender()
                poison_detection = ctx_defender.scan_action_reasoning(
                    action.reasoning, action.action_type, action.target
                )
                if poison_detection and poison_detection.severity == "CRITICAL":
                    self._logs_buffer.append(
                        f"[CONTEXT_POISON] CRITICAL: {poison_detection.attack_type} detected in reasoning"
                    )
                    try:
                        from .observability import get_observability_dashboard
                        get_observability_dashboard().track_context_poisoning(
                            poison_detection.attack_type, True
                        )
                    except Exception:
                        pass
            except Exception:
                pass
            if not arl_decision.proceed:                # Circuit breaker blocked — return forced message without executing
                self._arl.record_blocked(
                    step=self._state.steps_taken,
                    action_type=action.action_type,
                    target=action.target,
                    health=health_before,
                )
                # Compute a small penalty reward for the blocked action
                from .reward import compute_reward as _cr
                rb_blocked = _cr(
                    action_type=action.action_type, target=action.target,
                    action_success=False, health_before=health_before,
                    health_after=health_before, step_number=self._state.steps_taken,
                    max_steps=self.MAX_STEPS, correct_targets=scenario.correct_targets,
                    correct_actions=scenario.correct_actions, incident_resolved=False,
                    total_downtime_steps=self._state.total_downtime_steps,
                    wrong_actions_so_far=self._state.wrong_actions,
                )
                self._state.wrong_actions += 1
                self._state.cumulative_reward += rb_blocked.total
                # Sync new tracking fields
                self._state.wrong_fixes_applied = self._wrong_fixes_applied
                self._state.recovery_attempts = self._recovery_attempts
                self._state.hypotheses_tested = len(self._hypotheses_tested)
                return AIRENObservation(
                    incident_id=scenario.incident_id,
                    incident_type=scenario.incident_type,
                    severity=scenario.severity,
                    step_number=self._state.steps_taken,
                    max_steps=self.MAX_STEPS,
                    services=deepcopy(self._services),
                    metrics=deepcopy(self._metrics),
                    logs=self._logs_buffer[-5:] + [arl_decision.forced_message],
                    alerts=self._alerts,
                    system_health=health_before,
                    threat_level=round(self._threat_level, 3),
                    attack_progress=round(self._attack_progress, 3),
                    action_result=arl_decision.forced_message,
                    action_success=False,
                    recovery_score=0.0, diagnosis_score=0.0,
                    efficiency_score=0.0, threat_mitigation=0.0,
                    hallucination_penalty=0.1, security_violation_penalty=0.0,
                    cost_penalty=0.0, downtime_penalty=0.0,
                    resolve_bonus=0.0, exploration_bonus=0.0, recovery_bonus=0.0,
                    reward_explanation=f"ARL blocked: {arl_decision.blocked_by}",
                    metadata={
                        "arl_blocked": True,
                        "arl_reason": arl_decision.blocked_by,
                        "ledger_context": arl_decision.ledger_context,
                        "arl_stats": self._arl.stats,
                        "wrong_fixes_applied": self._wrong_fixes_applied,
                        "recovery_attempts": self._recovery_attempts,
                        "hypotheses_tested": len(self._hypotheses_tested),
                        "multi_agent_active": self._multi_agent,
                        "monitoring_signals": 0,
                        "autoscaler_actions": 0,
                        "topology": self._topology_info,
                        "k8s_pods": self._k8s_pods,
                        "digital_twin_enabled": self._digital_twin_enabled,
                    },
                    done=False,
                    reward=rb_blocked.total,
                )

        # ── 1. Attacker acts first ────────────────────────────────────────────
        attack_log: Optional[str] = None
        if self._attacker:
            attack = self._attacker.act(
                incident_type=scenario.incident_type,
                services=self._services,
                step=self._state.steps_taken,
                scenario_logs=scenario.initial_logs,
            )
            self._services, new_logs, self._metrics = self._attacker.apply(
                attack, self._services, [], self._metrics
            )
            if new_logs:
                attack_log = new_logs[0]
                self._logs_buffer.append(attack_log)

        # ── 1b. Monitoring agent surfaces signals ─────────────────────────────
        if self._monitor:
            signal = self._monitor.act(
                incident_type=scenario.incident_type,
                threat_level=self._threat_level,
                step=self._state.steps_taken,
                scenario_metrics=self._metrics,
                scenario_root_cause=scenario.root_cause,
                scenario_correct_targets=scenario.correct_targets,
            )
            new_logs, self._metrics = self._monitor.apply(signal, [], self._metrics)
            for log_line in new_logs:
                self._logs_buffer.append(log_line)

        # ── 1c. AutoScaler acts independently ────────────────────────────────
        if self._autoscaler:
            scaling = self._autoscaler.act(
                services=self._services,
                incident_type=scenario.incident_type,
                correct_targets=scenario.correct_targets,
            )
            if scaling:
                self._services, new_logs, self._metrics = self._autoscaler.apply(
                    scaling, self._services, [], self._metrics
                )
                for log_line in new_logs:
                    self._logs_buffer.append(log_line)

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
        # Sync new tracking fields to state (exposed via /state endpoint)
        self._state.wrong_fixes_applied = self._wrong_fixes_applied
        self._state.recovery_attempts = self._recovery_attempts
        self._state.hypotheses_tested = len(self._hypotheses_tested)

        # Track correct/wrong
        if action.target in scenario.correct_targets and action.action_type in scenario.correct_actions:
            self._state.correct_actions += 1
        else:
            self._state.wrong_actions += 1
            if action.action_type in scenario.wrong_action_effects:
                self._wrong_fixes_applied += 1
                self._logs_buffer.append(
                    f"[SYSTEM] Wrong fix applied: {scenario.wrong_fix_description or 'Situation worsened'}"
                )

        if action.action_type in ("inspect_logs", "inspect_metrics", "run_diagnostic"):
            if action.target not in self._hypotheses_tested:
                self._hypotheses_tested.append(action.target)

        if self._wrong_fixes_applied > 0 and action.action_type in scenario.correct_actions:
            self._recovery_attempts += 1

        # ── ARL Post-Action: Rollback check + Ledger record ───────────────────
        arl_rolled_back = False
        if self._arl_enabled and arl_decision is not None:
            rollback_result, _ledger_entry = self._arl.post_action(
                decision=arl_decision,
                step=self._state.steps_taken,
                action_type=action.action_type,
                target=action.target,
                action_result=result_msg,
                action_success=success,
                health_before=health_before,
                health_after=health_after,
                clues_revealed=self._diagnostic_clues_revealed[-1:] if self._diagnostic_clues_revealed else None,
            )
            if rollback_result.rolled_back:
                arl_rolled_back = True
                restored = self._arl.rollback_engine.restore_snapshot(arl_decision.snapshot)
                self._services = restored["services"]
                self._metrics = restored["metrics"]
                self._alerts = restored["alerts"]
                self._logs_buffer = restored["logs"]
                self._threat_level = restored["threat_level"]
                self._attack_progress = restored["attack_progress"]
                health_after = restored["health"]
                self._logs_buffer.append(rollback_result.forced_message)
                result_msg = rollback_result.forced_message
                success = False

        # ── 5. Dense multi-objective reward ───────────────────────────────────
        # Counterfactual: estimate what a random action would have done.
        # Random action = diagnostic on a random service (no health change expected).
        # If agent's health_delta > random_expected_delta → positive counterfactual.
        _random_expected_delta = -scenario.degradation_rate  # random action = world degrades
        _counterfactual_delta = (health_after - health_before) - _random_expected_delta

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
            hypotheses_tested=len(self._hypotheses_tested),
            recovery_after_wrong_fix=self._recovery_attempts > 0 and resolved,
            counterfactual_health_delta=round(_counterfactual_delta, 3),
        )
        self._state.cumulative_reward += rb.total

        done = self._state.incident_resolved or self._state.steps_taken >= self.MAX_STEPS

        # ── Per-step: Reward Hacking Detection ───────────────────────────────
        hack_penalty_explanation = ""
        try:
            hack_detector = get_hack_detector()
            hack_detection = hack_detector.check_step(
                action_type=action.action_type,
                target=action.target,
                reasoning=action.reasoning or "",
                health_before=health_before,
                health_after=health_after,
                reward_before_hack_check=rb.total,
                incident_resolved=resolved,
                step_number=self._state.steps_taken,
                max_steps=self.MAX_STEPS,
                correct_targets=scenario.correct_targets,
                correct_actions=scenario.correct_actions,
                exploration_bonus=rb.exploration_bonus,
                resolve_bonus=rb.resolve_bonus,
            )
            if hack_detection:
                adjusted, hack_penalty_explanation = hack_detector.apply_penalty(
                    rb.total, hack_detection
                )
                # Adjust cumulative reward
                self._state.cumulative_reward -= rb.total
                self._state.cumulative_reward += adjusted
        except Exception:
            pass

        # ── Per-step: Persistent Memory recording ────────────────────────────
        try:
            get_agent_memory().record_step(
                action_type=action.action_type,
                target=action.target,
                reasoning=action.reasoning or "",
                reward=rb.total,
                health=health_after,
            )
        except Exception:
            pass

        # ── Per-step: Self-Reward augmentation ───────────────────────────────
        self_reward_info = {}
        try:
            self_judge = get_self_reward_judge()
            self_result = self_judge.evaluate(
                incident_type=scenario.incident_type,
                action_type=action.action_type,
                target=action.target,
                reasoning=action.reasoning or "",
                health_before=health_before,
                health_after=health_after,
                env_reward=rb.total,
                correct_targets=scenario.correct_targets,
                correct_actions=scenario.correct_actions,
                step_number=self._state.steps_taken,
            )
            self_reward_info = {
                "self_score": self_result.self_score,
                "self_calibrated": self_result.calibrated_score,
                "self_critique": self_result.self_critique,
                "self_judge_used": self_result.judge_used,
            }
        except Exception:
            pass

        # ── Multi-agent: Deadlock detection ──────────────────────────────────
        if self._multi_agent:
            try:
                deadlock_detector = get_deadlock_detector()
                deadlock_detector.record_agent_action(
                    "defender", action.action_type, action.target
                )
                deadlock_event = deadlock_detector.detect_deadlock(
                    self._state.steps_taken, ["defender", "attacker", "monitor"]
                )
                if deadlock_event:
                    resolution = deadlock_detector.resolve_deadlock(
                        deadlock_event, ["defender", "attacker", "monitor"]
                    )
                    self._logs_buffer.append(
                        f"[DEADLOCK] {deadlock_event.deadlock_type}: "
                        f"{resolution.get('inject_message', '')[:80]}"
                    )
            except Exception:
                pass

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

            # ── Observability tracking ────────────────────────────────────────
            try:
                from .observability import get_observability_dashboard
                obs_dash = get_observability_dashboard()
                obs_dash.track_episode_end(
                    incident_type=scenario.incident_type,
                    resolved=self._state.incident_resolved,
                    steps_taken=self._state.steps_taken,
                    cumulative_reward=self._state.cumulative_reward,
                    diagnosis_score=judge_result.final_score if judge_result else rb.diagnosis_score,
                    api_calls=judge_tokens // 100 if judge_tokens else 0,
                )
            except Exception:
                pass

            # ── Incident Replay Forensics recording ───────────────────────────
            try:
                from .incident_replay import get_replay_forensics
                forensics = get_replay_forensics()
                action_records = []
                for i, act_str in enumerate(self._state.actions_taken):
                    parts = act_str.split(":")
                    action_records.append({
                        "action_type": parts[0] if parts else act_str,
                        "target": parts[1] if len(parts) > 1 else "",
                        "reasoning": "",
                        "step": i + 1,
                    })
                forensics.record_episode(
                    episode_id=self._state.episode_id or f"ep_{int(time.time())}",
                    incident_type=scenario.incident_type,
                    seed=getattr(self, "_last_seed", 0),
                    actions=action_records,
                    rewards=[],
                    health_trajectory=list(self._state.system_health_history),
                    resolved=self._state.incident_resolved,
                    final_health=health_after,
                    root_cause=scenario.root_cause,
                    correct_actions=scenario.correct_actions,
                    correct_targets=scenario.correct_targets,
                    attacker_seed=getattr(self, "_last_attacker_seed", 0),
                )
            except Exception:
                pass

            # ── Self-Evolving Curriculum: record episode outcome ───────────────
            try:
                get_curriculum().record_episode(
                    incident_type=scenario.incident_type,
                    reward=self._state.cumulative_reward,
                    resolved=self._state.incident_resolved,
                    steps_taken=self._state.steps_taken,
                )
            except Exception:
                pass

            # ── Persistent Memory: consolidate episode ────────────────────────
            try:
                get_agent_memory().consolidate_episode(
                    episode_id=self._state.episode_id or f"ep_{int(time.time())}",
                    incident_type=scenario.incident_type,
                    seed=getattr(self, "_last_seed", 0),
                    resolved=self._state.incident_resolved,
                    steps_taken=self._state.steps_taken,
                    cumulative_reward=self._state.cumulative_reward,
                    final_health=health_after,
                    root_cause=scenario.root_cause,
                    correct_actions=scenario.correct_actions,
                    correct_targets=scenario.correct_targets,
                )
            except Exception:
                pass

            # ── State Drift Monitor: record episode ───────────────────────────
            try:
                action_types = [a.split(":")[0] for a in self._state.actions_taken]
                get_drift_monitor().record_episode(
                    reward=self._state.cumulative_reward,
                    resolved=self._state.incident_resolved,
                    action_types=action_types,
                    incident_type=scenario.incident_type,
                )
            except Exception:
                pass

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
                exploration_bonus=rb.exploration_bonus,
                recovery_bonus=rb.recovery_bonus,
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
            "hypotheses_tested": len(self._hypotheses_tested),
            "wrong_fixes_applied": self._wrong_fixes_applied,
            "recovery_attempts": self._recovery_attempts,
            "multi_agent_active": self._multi_agent,
            "monitoring_signals": len(self._monitor.signal_history) if self._monitor else 0,
            "autoscaler_actions": len(self._autoscaler.scaling_history) if self._autoscaler else 0,
            "topology": self._topology_info,
            "k8s_pods": self._k8s_pods,
            "digital_twin_enabled": self._digital_twin_enabled,
            "arl_enabled": self._arl_enabled,
            "arl_stats": self._arl.stats if self._arl_enabled else {},
            "arl_rolled_back": arl_rolled_back,
            "ledger_context": self._arl.get_ledger_context(health_after, scenario.incident_type) if self._arl_enabled else "",
            "compliance_enabled": self._compliance_enabled,
            "compliance_stats": self._compliance.stats if self._compliance_enabled else {},
            "compliance_violations": [
                {"framework": v.framework, "article": v.article, "severity": v.severity}
                for v in (compliance_decision.violations if compliance_decision else [])
                if not v.blocked
            ],
            # Counterfactual reward signal
            "counterfactual_bonus": rb.counterfactual_bonus,
            "counterfactual_delta": round(_counterfactual_delta, 3),
            "random_expected_delta": round(_random_expected_delta, 3),
            # Self-evolving upgrades
            "reward_hacking_detected": bool(hack_penalty_explanation),
            "reward_hacking_explanation": hack_penalty_explanation,
            "self_reward": self_reward_info,
            "curriculum_stats": {},  # populated lazily
        }

        final_reward = judge_result.final_score if judge_result else rb.total
        final_reward = _clamp_score(final_reward)

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
            exploration_bonus=rb.exploration_bonus,
            recovery_bonus=rb.recovery_bonus,
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
            reward=final_reward,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def compliance_audit(self) -> dict:
        return self._compliance.get_audit_report() if self._compliance_enabled else {}

    @property
    def arl_stats(self) -> dict:
        return self._arl.stats if self._arl_enabled else {}

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

        # ── Inspect actions: return real clues from THIS scenario ─────────────
        if atype in ("inspect_logs", "inspect_metrics", "run_diagnostic"):
            clue_map = _build_inspect_clues(scenario)
            clues = clue_map.get(scenario.incident_type, {}).get(atype, [])
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
