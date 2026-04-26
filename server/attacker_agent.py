# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Attacker Agent — adversarial counterpart in multi-agent mode.

The attacker runs concurrently with the defender and tries to:
  - Inject misleading log entries (prompt injection into the environment)
  - Escalate the incident faster than the defender can fix it
  - Trigger cascading failures via targeted service attacks

This creates a true adversarial RL environment — the defender must
outpace an active attacker, not just a static scenario.
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

_LOG_INJECTIONS = [
    # Misleading red herrings designed to confuse the defender
    "[CACHE] Cache eviction rate normal — no issues detected",
    "[DB] Replication lag: 0ms — all replicas healthy",
    "[API] Health check: OK (200ms)",
    "[WORKER] Job queue nominal — 0 pending",
    "[INFRA] Network latency: 1ms — all routes healthy",
    "[DEPLOY] No deployments in last 24h",
    "[DB] Index statistics up to date",
    "[CACHE] Memory usage: 45% — well within limits",
    # Panic-inducing false positives
    "[SECURITY] Unusual login from 192.168.1.1 — investigating",
    "[INFRA] Disk I/O spike on /dev/sda — monitoring",
    "[NETWORK] BGP route flap detected — auto-recovering",
]

_ESCALATION_TARGETS = {
    "db_overload":       ("db",      {"cpu_pct": 99, "error_rate": 0.35, "status": "degraded"}),
    "memory_leak":       ("worker",  {"memory_pct": 97, "status": "degraded"}),
    "bad_deployment":    ("payment", {"error_rate": 0.55, "latency_ms": 2000}),
    "network_partition": ("api",     {"error_rate": 1.0, "status": "down"}),
    "cache_stampede":    ("cache",   {"status": "down", "error_rate": 0.9}),
}


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
    ) -> AttackAction:
        """Choose an attack action based on current game state."""
        self._step = step

        # Phase 1 (steps 1-2): confusion via log injection
        if step <= 2:
            payload = random.choice(_LOG_INJECTIONS)
            self._injected.append(payload)
            return AttackAction(
                attack_type="inject_log",
                target="logs",
                payload=payload,
                severity_boost=0.0,
            )

        # Phase 2 (steps 3-4): escalate primary service
        if step <= 4:
            target, changes = _ESCALATION_TARGETS.get(
                incident_type,
                ("api", {"error_rate": 0.3, "status": "degraded"}),
            )
            return AttackAction(
                attack_type="escalate_service",
                target=target,
                payload=f"Attacker escalating {target}: {changes}",
                severity_boost=self.aggression * 0.3,
            )

        # Phase 3 (step 5+): cascade to healthy services
        healthy = [
            name for name, svc in services.items()
            if svc.get("status") == "healthy"
        ]
        if healthy and random.random() < self.aggression:
            cascade_target = random.choice(healthy)
            return AttackAction(
                attack_type="trigger_cascade",
                target=cascade_target,
                payload=f"Cascading failure injected into {cascade_target}",
                severity_boost=self.aggression * 0.5,
            )

        # Fallback: another log injection
        payload = random.choice(_LOG_INJECTIONS)
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
