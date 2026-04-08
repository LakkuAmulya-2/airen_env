# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Multi-Agent System for AIREN — 3 concurrent agents running alongside the defender.

1. AttackerAgent    — adversarial: injects misleading logs, escalates incidents
2. MonitoringAgent  — cooperative: surfaces real signals, reduces noise
3. AutoScalerAgent  — autonomous: scales services based on load thresholds

This creates a true multi-agent RL environment:
  - Defender must outpace an active attacker
  - Monitoring agent provides partial help (but can be wrong)
  - AutoScaler acts independently, sometimes helping, sometimes interfering
  - Defender must coordinate with cooperative agents while resisting adversarial ones
"""

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AttackAction:
    attack_type: str          # "inject_log" | "escalate_service" | "trigger_cascade"
    target: str               # which service to attack
    payload: str              # log line to inject or escalation description
    severity_boost: float     # how much to increase risk (0.0-1.0)


# ── Attack catalogue ──────────────────────────────────────────────────────────
# No static lists — all attack content generated at runtime from scenario data

# Escalation targets per incident type — service + metric changes
# Values are relative (multipliers/additions), not absolute hardcoded numbers
_ESCALATION_TARGETS = {
    "db_overload":       ("db",       "cpu_pct"),
    "memory_leak":       ("worker",   "memory_pct"),
    "bad_deployment":    ("payment",  "error_rate"),
    "network_partition": ("api",      "error_rate"),
    "cache_stampede":    ("cache",    "error_rate"),
    "api_timeout":       ("upstream", "error_rate"),
    "disk_full":         ("infra",    "error_rate"),
    "ssl_cert_expired":  ("tls",      "error_rate"),
    "ddos_attack":       ("network",  "error_rate"),
}


def _build_injection_pool(
    scenario_logs: Optional[List[str]],
    services: Dict[str, Dict],
) -> List[str]:
    """
    Build misleading log injection pool entirely from live scenario data.
    Uses actual service names and current metric values — no static strings.
    """
    pool = []
    # Inject "healthy" looking logs for each service using its real name
    for svc_name, svc_data in services.items():
        cpu = svc_data.get("cpu_pct", 30)
        lat = svc_data.get("latency_ms", 50)
        err = svc_data.get("error_rate", 0.0)
        # Only inject misleading "OK" logs for services that look degraded
        # (to confuse the agent about which service is actually the problem)
        pool.append(f"[{svc_name.upper()}] Health check: OK ({lat}ms response)")
        pool.append(f"[{svc_name.upper()}] CPU: {cpu}% — within normal range")
        if err > 0:
            pool.append(f"[{svc_name.upper()}] Error rate {err:.1%} — transient, auto-recovering")
    # Also inject red-herring logs from the scenario's own initial logs
    if scenario_logs:
        for log in scenario_logs:
            low = log.lower()
            if any(kw in low for kw in ["nominal", "healthy", "ok", "normal",
                                         "completed", "stable", "resolved"]):
                pool.append(log)
    return pool if pool else [f"[SYSTEM] All services nominal — no anomalies detected"]


def _build_false_positive_pool(
    scenario_metrics: Dict[str, float],
    services: Dict[str, Dict],
) -> List[tuple]:
    """
    Build false-positive monitoring signals from actual scenario metrics.
    Returns list of (service, message) tuples — no static strings.
    """
    pool = []
    for svc_name, svc_data in services.items():
        cpu = svc_data.get("cpu_pct", 30)
        lat = svc_data.get("latency_ms", 50)
        mem = svc_data.get("memory_pct", 40)
        # Generate plausible-but-wrong signals
        pool.append((svc_name,
                     f"MONITORING: {svc_name} CPU {cpu}% — spike detected, auto-resolving"))
        pool.append((svc_name,
                     f"MONITORING: {svc_name} latency {lat}ms — within SLA bounds"))
        pool.append((svc_name,
                     f"MONITORING: {svc_name} memory {mem}% — no action needed"))
    # Add metric-based false positives
    for metric_key, val in list(scenario_metrics.items())[:3]:
        svc = metric_key.split("_")[0] if "_" in metric_key else "api"
        pool.append((svc,
                     f"MONITORING: {metric_key}={val} — within expected range, monitoring"))
    return pool if pool else [("api", "MONITORING: All metrics nominal — no action needed")]


class AttackerAgent:
    """
    Adversarial agent that actively worsens the incident.

    Strategy:
      - Step 1-2: Inject misleading logs to confuse defender
      - Step 3+:  Escalate the primary failing service
      - Step 5+:  Trigger cascading failures on secondary services
    """

    def __init__(self, aggression: float = 0.5) -> None:
        """
        Args:
            aggression: 0.0 (passive) to 1.0 (maximum aggression).
                        Controls how aggressively the attacker escalates.
        """
        self.aggression = max(0.0, min(1.0, aggression))
        self._step = 0
        self._injected: List[str] = []

    def reset(self) -> None:
        self._step = 0
        self._injected = []

    def act(
        self,
        incident_type: str,
        services: Dict[str, Dict],
        step: int,
        scenario_logs: Optional[List[str]] = None,
    ) -> AttackAction:
        """Choose an attack action based on current game state."""
        self._step = step

        # Phase 1 (steps 1-2): inject misleading logs built from live scenario data
        if step <= 2:
            pool = _build_injection_pool(scenario_logs, services)
            payload = random.choice(pool)
            self._injected.append(payload)
            return AttackAction(
                attack_type="inject_log",
                target="logs",
                payload=payload,
                severity_boost=0.0,
            )

        # Phase 2 (steps 3-4): escalate the primary failing service
        # Use current metric values to compute realistic escalation
        if step <= 4:
            target_svc, metric_key = _ESCALATION_TARGETS.get(
                incident_type, ("api", "error_rate")
            )
            svc_data = services.get(target_svc, {})
            # Escalate relative to current value — not a hardcoded absolute
            changes: Dict[str, Any] = {}
            if metric_key == "cpu_pct":
                current = svc_data.get("cpu_pct", 50)
                changes["cpu_pct"] = min(100, int(current + 15 * self.aggression))
                changes["status"] = "degraded"
            elif metric_key == "memory_pct":
                current = svc_data.get("memory_pct", 50)
                changes["memory_pct"] = min(99, int(current + 10 * self.aggression))
                changes["status"] = "degraded"
            else:  # error_rate
                current = svc_data.get("error_rate", 0.1)
                changes["error_rate"] = min(1.0, round(current + 0.2 * self.aggression, 3))
                if changes["error_rate"] > 0.7:
                    changes["status"] = "down"
                elif changes["error_rate"] > 0.4:
                    changes["status"] = "degraded"
            return AttackAction(
                attack_type="escalate_service",
                target=target_svc,
                payload=f"Attacker escalating {target_svc}: {changes}",
                severity_boost=self.aggression * 0.3,
            )

        # Phase 3 (step 5+): cascade to a healthy service
        healthy = [n for n, s in services.items() if s.get("status") == "healthy"]
        if healthy and random.random() < self.aggression:
            cascade_target = random.choice(healthy)
            return AttackAction(
                attack_type="trigger_cascade",
                target=cascade_target,
                payload=f"Cascading failure injected into {cascade_target}",
                severity_boost=self.aggression * 0.5,
            )

        # Fallback: another log injection
        pool = _build_injection_pool(scenario_logs, services)
        payload = random.choice(pool)
        return AttackAction(
            attack_type="inject_log",
            target="logs",
            payload=payload,
            severity_boost=0.0,
        )

    def apply(
        self,
        attack: AttackAction,
        services: Dict[str, Dict],
        logs: List[str],
        metrics: Dict[str, float],
    ) -> Tuple[Dict[str, Dict], List[str], Dict[str, float]]:
        """
        Apply the attack to the environment state.
        Returns updated (services, logs, metrics).
        """
        if attack.attack_type == "inject_log":
            logs = logs + [attack.payload]

        elif attack.attack_type == "escalate_service":
            target = attack.target
            if target in services:
                svc = services[target]
                svc["error_rate"] = min(1.0, svc.get("error_rate", 0.0) + 0.15 * self.aggression)
                svc["latency_ms"] = int(svc.get("latency_ms", 100) * (1 + 0.3 * self.aggression))
                if svc["error_rate"] > 0.5:
                    svc["status"] = "degraded"
                if svc["error_rate"] > 0.8:
                    svc["status"] = "down"
            # Boost risk metric
            metrics["attacker_escalation_count"] = metrics.get("attacker_escalation_count", 0) + 1

        elif attack.attack_type == "trigger_cascade":
            target = attack.target
            if target in services:
                services[target]["status"] = "degraded"
                services[target]["error_rate"] = min(1.0, 0.2 + 0.3 * self.aggression)
                services[target]["latency_ms"] = int(services[target].get("latency_ms", 50) * 2)
            logs = logs + [f"[{target.upper()}] Cascading failure detected — service degrading"]
            metrics["cascade_events"] = metrics.get("cascade_events", 0) + 1

        return services, logs, metrics

    @property
    def injected_logs(self) -> List[str]:
        return list(self._injected)


# ── MonitoringAgent ───────────────────────────────────────────────────────────

@dataclass
class MonitoringSignal:
    signal_type: str
    service: str
    message: str
    is_real: bool
    confidence: float


def _correct_service_for(incident_type: str, scenario_correct_targets: Optional[List[str]] = None) -> str:
    """Derive the correct service to monitor from the scenario's correct_targets — no static map."""
    if scenario_correct_targets:
        return scenario_correct_targets[0]
    # Fallback: derive from incident type name
    if "db" in incident_type or "overload" in incident_type:
        return "db"
    if "memory" in incident_type or "leak" in incident_type:
        return "worker"
    if "network" in incident_type or "partition" in incident_type:
        return "network"
    if "cache" in incident_type:
        return "cache"
    if "deploy" in incident_type:
        return "payment"
    if "timeout" in incident_type:
        return "upstream"
    if "disk" in incident_type:
        return "infra"
    if "ssl" in incident_type or "cert" in incident_type:
        return "tls"
    if "ddos" in incident_type or "attack" in incident_type:
        return "network"
    return "api"


class MonitoringAgent:
    """
    Cooperative monitoring agent that surfaces diagnostic signals.

    Runs every step and either:
    - Surfaces a real diagnostic signal (helps the defender)
    - Injects a false positive (adds noise)

    The defender must learn to distinguish real signals from noise.
    Accuracy degrades under high threat (attacker interference).
    """

    def __init__(self, accuracy: float = 0.75) -> None:
        """
        Args:
            accuracy: 0.0-1.0 — probability of surfacing a real signal vs false positive.
                      Degrades under high threat level.
        """
        self.base_accuracy = max(0.0, min(1.0, accuracy))
        self._step = 0
        self._signals_emitted: List[MonitoringSignal] = []

    def reset(self) -> None:
        self._step = 0
        self._signals_emitted = []

    def act(
        self,
        incident_type: str,
        threat_level: float,
        step: int,
        scenario_metrics: Optional[Dict[str, float]] = None,
        scenario_root_cause: str = "",
        scenario_correct_targets: Optional[List[str]] = None,
    ) -> MonitoringSignal:
        """Emit a monitoring signal — real or false positive based on accuracy."""
        self._step = step
        effective_accuracy = self.base_accuracy * (1.0 - threat_level * 0.3)
        service = _correct_service_for(incident_type, scenario_correct_targets)

        if random.random() < effective_accuracy:
            # Real signal — built from actual scenario data
            # Use scenario_correct_targets if available, else derive from incident type
            service = _correct_service_for(incident_type, scenario_correct_targets)
            if scenario_metrics:
                metric_hints = {k: v for k, v in scenario_metrics.items()
                                if any(kw in k for kw in
                                       ["cpu","error","latency","memory","miss",
                                        "timeout","disk","tls","requests","connections"])}
                if metric_hints:
                    key, val = max(metric_hints.items(), key=lambda x: float(x[1]))
                    message = f"MONITORING: {key}={val} on {service} — investigate {service} service"
                else:
                    message = f"MONITORING: Anomaly on {service} — {scenario_root_cause[:60]}"
            else:
                message = f"MONITORING: Anomaly detected on {service} — investigate immediately"
            signal = MonitoringSignal(
                signal_type="metric_spike",
                service=service,
                message=message,
                is_real=True,
                confidence=round(effective_accuracy, 2),
            )
        else:
            # False positive — built from live scenario metrics, not static strings
            fp_pool = _build_false_positive_pool(
                scenario_metrics or {}, {}
            )
            service, message = random.choice(fp_pool)
            signal = MonitoringSignal(
                signal_type="false_positive",
                service=service,
                message=message,
                is_real=False,
                confidence=round(1.0 - effective_accuracy, 2),
            )

        self._signals_emitted.append(signal)
        return signal

    def apply(
        self,
        signal: MonitoringSignal,
        logs: List[str],
        metrics: Dict[str, float],
    ) -> Tuple[List[str], Dict[str, float]]:
        """Inject the monitoring signal into the environment."""
        logs = logs + [signal.message]
        if signal.is_real:
            metrics["monitoring_real_signals"] = metrics.get("monitoring_real_signals", 0) + 1
        else:
            metrics["monitoring_false_positives"] = metrics.get("monitoring_false_positives", 0) + 1
        return logs, metrics

    @property
    def signal_history(self) -> List[MonitoringSignal]:
        return list(self._signals_emitted)


# ── AutoScalerAgent ───────────────────────────────────────────────────────────

@dataclass
class ScalingAction:
    action_type: str      # "scale_up" | "scale_down" | "no_op"
    service: str
    reason: str
    replicas_delta: int   # +N or -N
    helpful: bool         # True if this scaling actually helps the incident


# Thresholds that trigger auto-scaling
_SCALE_UP_THRESHOLD_CPU = 80    # CPU > 80% → scale up
_SCALE_DOWN_THRESHOLD_CPU = 20  # CPU < 20% → scale down (cost optimization)


class AutoScalerAgent:
    """
    Autonomous auto-scaler that acts independently based on CPU/load thresholds.

    Sometimes helpful (scales up the right service under load),
    sometimes interfering (scales down a service the defender needs,
    or scales up the wrong service wasting resources).

    The defender must account for auto-scaler actions in their strategy.
    """

    def __init__(self, aggressiveness: float = 0.6) -> None:
        """
        Args:
            aggressiveness: 0.0-1.0 — how quickly the scaler reacts.
                            High aggressiveness = more frequent scaling actions.
        """
        self.aggressiveness = max(0.0, min(1.0, aggressiveness))
        self._step = 0
        self._scaling_history: List[ScalingAction] = []

    def reset(self) -> None:
        self._step = 0
        self._scaling_history = []

    def act(
        self,
        services: Dict[str, Dict],
        incident_type: str,
        correct_targets: List[str],
    ) -> Optional[ScalingAction]:
        """
        Decide whether to scale any service based on current metrics.
        Returns None if no scaling action is needed.
        """
        self._step += 1

        # Only act every 2-3 steps (not every step)
        if random.random() > self.aggressiveness * 0.5:
            return None

        # Find services that need scaling
        for name, svc in services.items():
            cpu = svc.get("cpu_pct", 0)
            status = svc.get("status", "healthy")

            if cpu > _SCALE_UP_THRESHOLD_CPU and status != "down":
                helpful = name in correct_targets
                action = ScalingAction(
                    action_type="scale_up",
                    service=name,
                    reason=f"CPU {cpu}% > {_SCALE_UP_THRESHOLD_CPU}% threshold",
                    replicas_delta=+2,
                    helpful=helpful,
                )
                self._scaling_history.append(action)
                return action

            if cpu < _SCALE_DOWN_THRESHOLD_CPU and status == "healthy" and self._step > 3:
                # Scale down healthy services to save cost — may interfere with recovery
                helpful = False
                action = ScalingAction(
                    action_type="scale_down",
                    service=name,
                    reason=f"CPU {cpu}% < {_SCALE_DOWN_THRESHOLD_CPU}% — cost optimization",
                    replicas_delta=-1,
                    helpful=False,
                )
                self._scaling_history.append(action)
                return action

        return None

    def apply(
        self,
        scaling: ScalingAction,
        services: Dict[str, Dict],
        logs: List[str],
        metrics: Dict[str, float],
    ) -> Tuple[Dict[str, Dict], List[str], Dict[str, float]]:
        """Apply the scaling action to the environment."""
        svc = services.get(scaling.service, {})

        if scaling.action_type == "scale_up":
            # Scale up reduces CPU load but adds brief latency spike
            svc["cpu_pct"] = max(20, svc.get("cpu_pct", 50) - 15)
            if scaling.helpful:
                # Scaling the right service helps recovery
                svc["error_rate"] = max(0.0, svc.get("error_rate", 0.0) - 0.05)
            logs = logs + [
                f"[AUTOSCALER] Scaled up {scaling.service}: +{scaling.replicas_delta} replicas ({scaling.reason})"
            ]
            metrics["autoscaler_scale_up_events"] = metrics.get("autoscaler_scale_up_events", 0) + 1

        elif scaling.action_type == "scale_down":
            # Scale down increases CPU load slightly
            svc["cpu_pct"] = min(100, svc.get("cpu_pct", 30) + 10)
            logs = logs + [
                f"[AUTOSCALER] Scaled down {scaling.service}: {scaling.replicas_delta} replicas ({scaling.reason})"
            ]
            metrics["autoscaler_scale_down_events"] = metrics.get("autoscaler_scale_down_events", 0) + 1

        return services, logs, metrics

    @property
    def scaling_history(self) -> List[ScalingAction]:
        return list(self._scaling_history)
