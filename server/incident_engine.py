# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Incident Engine — generates realistic production incidents with cascading failures.

5 incident types (easy → hard):
  1. bad_deployment    (easy)   — new version causes error spike → rollback
  2. db_overload       (medium) — unindexed query → DB CPU 95% → API latency
  3. memory_leak       (medium) — worker leaks memory → OOM → crash
  4. network_partition (hard)   — firewall blocks API→DB → 100% errors
  5. cache_stampede    (hard)   — TTL expiry storm → DB cascade

Each incident has:
  - Realistic initial state (symptoms visible to agent)
  - Hidden root cause (not shown to agent)
  - Cascading failure progression (wrong actions make it worse)
  - Noise logs (red herrings to test diagnosis quality)
  - Recovery path (correct actions to fix)
"""

import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IncidentScenario:
    incident_id: str
    incident_type: str
    severity: str
    root_cause: str          # hidden from agent
    description: str
    initial_services: Dict[str, Dict[str, Any]]
    initial_metrics: Dict[str, float]
    initial_alerts: List[Dict[str, Any]]
    initial_logs: List[str]
    correct_actions: List[str]   # action_types that fix it
    correct_targets: List[str]   # which services to target
    wrong_action_effects: Dict[str, Dict[str, Any]]
    recovery_trajectory: List[float]
    # RL-grade additions
    threat_level: float = 0.5          # initial threat 0.0-1.0
    degradation_rate: float = 0.05     # health lost per step if untreated
    cascade_threshold: float = 3.0     # attack_progress that triggers cascade


def _svc(status="healthy", latency=50, error_rate=0.0, cpu=30, memory=40):
    return {"status": status, "latency_ms": latency, "error_rate": error_rate,
            "cpu_pct": cpu, "memory_pct": memory, "uptime_s": random.randint(3600, 86400)}


def _bad_deployment(seed: int) -> IncidentScenario:
    random.seed(seed)
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-DEP",
        incident_type="bad_deployment",
        severity="high",
        root_cause="v2.4.1 introduced null pointer exception in payment handler",
        description="Error rate spiked after deployment 20 minutes ago",
        initial_services={
            "api":     _svc("degraded", 180, 0.18, 45, 55),
            "db":      _svc("healthy",  40,  0.0,  38, 52),
            "cache":   _svc("healthy",  2,   0.0,  12, 22),
            "payment": _svc("degraded", 950, 0.35, 60, 65),
        },
        initial_metrics={
            "api_error_rate": 0.18, "payment_error_rate": 0.35,
            "deployment_age_minutes": 20.0, "previous_error_rate": 0.01,
        },
        initial_alerts=[
            {"service": "payment", "severity": "high",
             "message": "Error rate 35% (was 1% before deploy)", "triggered_at": time.time()-1200},
            {"service": "api", "severity": "high",
             "message": "Error rate 18% after v2.4.1 deploy", "triggered_at": time.time()-1180},
        ],
        initial_logs=[
            "[PAYMENT] NullPointerException in PaymentHandler.process() line 247",
            "[PAYMENT] NullPointerException in PaymentHandler.process() line 247",
            "[API] 500 Error: payment service exception",
            "[DEPLOY] v2.4.1 deployed at 14:32 UTC",
            "[PAYMENT] Stack trace: PaymentHandler -> CardValidator -> null",
            "[DB] Query performance nominal",           # red herring
            "[DEPLOY] Previous version: v2.4.0 (stable)",
            "[PAYMENT] NullPointerException in PaymentHandler.process() line 247",
        ],
        correct_actions=["rollback_deployment"],
        correct_targets=["payment"],
        wrong_action_effects={
            "restart_service": {"payment": {"error_rate": 0.35}},
            "scale_service":   {"payment": {"error_rate": 0.35}},
        },
        recovery_trajectory=[0.4, 0.65, 0.85, 0.95, 1.0],
    )


def _db_overload(seed: int) -> IncidentScenario:
    random.seed(seed)
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-DB",
        incident_type="db_overload",
        severity="critical",
        root_cause="Unindexed query causing full table scan on orders table",
        description="DB CPU at 95%, causing API latency spike and user errors",
        initial_services={
            "api":    _svc("degraded", 2400, 0.12, 65, 55),
            "db":     _svc("degraded", 8500, 0.08, 95, 88),
            "cache":  _svc("healthy",  2,    0.0,  15, 30),
            "worker": _svc("healthy",  120,  0.02, 40, 45),
        },
        initial_metrics={
            "db_cpu_pct": 95.0, "db_connections": 487.0,
            "api_p99_latency_ms": 2400.0, "api_error_rate": 0.12,
            "cache_hit_rate": 0.82,
        },
        initial_alerts=[
            {"service": "db",  "severity": "critical",
             "message": "DB CPU > 90% for 5 minutes", "triggered_at": time.time()-300},
            {"service": "api", "severity": "high",
             "message": "API p99 latency > 2000ms",   "triggered_at": time.time()-240},
        ],
        initial_logs=[
            "[DB] SLOW QUERY: SELECT * FROM orders WHERE user_id=? (8.4s)",
            "[DB] SLOW QUERY: SELECT * FROM orders WHERE user_id=? (9.1s)",
            "[API] 500 Internal Server Error: upstream timeout",
            "[DB] Connection pool exhausted: 487/500 connections used",
            "[CACHE] Cache hit rate nominal: 82%",      # red herring
            "[WORKER] Background job completed",        # red herring
            "[DB] SLOW QUERY: SELECT * FROM orders WHERE user_id=? (11.2s)",
            "[DB] Index scan on orders.user_id: 2.3M rows scanned",
        ],
        correct_actions=["run_diagnostic", "apply_fix"],
        correct_targets=["db", "db"],
        wrong_action_effects={
            "restart_service": {"api": {"status": "down", "error_rate": 1.0}},
            "ignore_alert":    {"db":  {"cpu_pct": 99, "error_rate": 0.25}},
        },
        recovery_trajectory=[0.3, 0.5, 0.7, 0.85, 0.95, 1.0],
    )


def _memory_leak(seed: int) -> IncidentScenario:
    random.seed(seed)
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-MEM",
        incident_type="memory_leak",
        severity="high",
        root_cause="Worker service not releasing buffers after request completion",
        description="Worker memory growing 2MB/min, approaching OOM threshold",
        initial_services={
            "api":    _svc("healthy",  80,  0.01, 35, 50),
            "db":     _svc("healthy",  45,  0.0,  40, 60),
            "cache":  _svc("healthy",  2,   0.0,  10, 25),
            "worker": _svc("degraded", 450, 0.05, 55, 91),
        },
        initial_metrics={
            "worker_memory_pct": 91.0,
            "worker_memory_growth_mb_per_min": 2.1,
            "worker_latency_ms": 450.0,
        },
        initial_alerts=[
            {"service": "worker", "severity": "high",
             "message": "Memory usage > 90%", "triggered_at": time.time()-600},
            {"service": "worker", "severity": "medium",
             "message": "Memory growing continuously", "triggered_at": time.time()-1200},
        ],
        initial_logs=[
            "[WORKER] Memory usage: 91% (threshold: 95%)",
            "[WORKER] Memory usage: 91.3% (+0.3% in 10s)",
            "[API] All requests nominal",               # red herring
            "[WORKER] Buffer pool size: 2.1GB (expected: 512MB)",
            "[WORKER] GC pause: 450ms (high)",
            "[WORKER] WARNING: Approaching OOM threshold",
            "[WORKER] Job handler: buffer not released after completion",
            "[DB] Query performance nominal",           # red herring
        ],
        correct_actions=["inspect_logs", "restart_service"],
        correct_targets=["worker", "worker"],
        wrong_action_effects={
            "scale_service":       {"worker": {"memory_pct": 91}},
            "rollback_deployment": {"worker": {"memory_pct": 85}},
            "ignore_alert":        {"worker": {"memory_pct": 96, "status": "down"}},
        },
        recovery_trajectory=[0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
    )


def _network_partition(seed: int) -> IncidentScenario:
    random.seed(seed)
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-NET",
        incident_type="network_partition",
        severity="critical",
        root_cause="Firewall rule change blocked API→DB traffic on port 5432",
        description="API cannot reach DB — all DB-dependent requests failing",
        initial_services={
            "api":     _svc("down",    0,   1.0,  20, 40),
            "db":      _svc("healthy", 45,  0.0,  35, 55),
            "cache":   _svc("healthy", 2,   0.0,  10, 20),
            "worker":  _svc("down",    0,   1.0,  15, 35),
            "network": _svc("degraded", 0,  1.0,  5,  10),  # virtual — represents firewall layer
        },
        initial_metrics={
            "api_error_rate": 1.0, "api_db_connection_failures": 847.0,
            "db_cpu_pct": 35.0, "db_connections": 0.0,
            "network_packet_loss_pct": 100.0,
        },
        initial_alerts=[
            {"service": "api", "severity": "critical",
             "message": "100% error rate — all requests failing", "triggered_at": time.time()-120},
            {"service": "api", "severity": "critical",
             "message": "DB connection refused: port 5432", "triggered_at": time.time()-110},
        ],
        initial_logs=[
            "[API] ERROR: connect ECONNREFUSED db:5432",
            "[API] ERROR: connect ECONNREFUSED db:5432",
            "[WORKER] ERROR: DB connection failed after 3 retries",
            "[DB] No incoming connections (firewall?)",
            "[CACHE] Cache operational",                # red herring
            "[INFRA] Firewall rules updated 8 minutes ago",
            "[API] ERROR: connect ECONNREFUSED db:5432",
            "[DB] Listening on port 5432 — no clients connected",
        ],
        correct_actions=["run_diagnostic", "apply_fix"],
        correct_targets=["network", "network"],
        wrong_action_effects={
            "restart_service":     {"api": {"status": "down", "error_rate": 1.0}},
            "rollback_deployment": {"api": {"error_rate": 1.0}},
        },
        recovery_trajectory=[0.2, 0.5, 0.8, 0.95, 1.0],
    )


def _cache_stampede(seed: int) -> IncidentScenario:
    random.seed(seed)
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-CACHE",
        incident_type="cache_stampede",
        severity="high",
        root_cause="Cache TTL expired simultaneously for 50K keys — thundering herd to DB",
        description="Cache miss storm causing DB overload — cascading latency",
        initial_services={
            "api":    _svc("degraded", 1800, 0.08, 70, 60),
            "db":     _svc("degraded", 4200, 0.05, 88, 75),
            "cache":  _svc("degraded", 5,    0.0,  20, 15),
            "worker": _svc("degraded", 800,  0.03, 55, 50),
        },
        initial_metrics={
            "cache_hit_rate": 0.12, "cache_miss_rate": 0.88,
            "db_cpu_pct": 88.0, "db_queries_per_sec": 12400.0,
            "api_latency_ms": 1800.0,
        },
        initial_alerts=[
            {"service": "cache", "severity": "high",
             "message": "Cache hit rate dropped to 12% (was 85%)", "triggered_at": time.time()-180},
            {"service": "db", "severity": "high",
             "message": "DB query rate 10x normal", "triggered_at": time.time()-175},
        ],
        initial_logs=[
            "[CACHE] Mass expiry: 52,847 keys expired simultaneously",
            "[DB] Query rate: 12,400/s (normal: 1,200/s)",
            "[CACHE] Cache miss: user_profile:* (thundering herd)",
            "[DB] Connection pool: 490/500 used",
            "[CACHE] TTL batch expiry — all keys set at 09:00 UTC expired",
            "[DB] CPU: 88% — query backlog growing",
            "[WORKER] Job queue backing up: 4,200 pending",
            "[CACHE] Cache miss: session_data:* (thundering herd)",
        ],
        correct_actions=["apply_fix", "scale_service"],
        correct_targets=["cache", "db"],
        wrong_action_effects={
            "restart_service": {"cache": {"status": "down"}},
            "ignore_alert":    {"db": {"cpu_pct": 95, "error_rate": 0.15}},
        },
        recovery_trajectory=[0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0],
    )


SCENARIO_GENERATORS = {
    "bad_deployment":    _bad_deployment,
    "db_overload":       _db_overload,
    "memory_leak":       _memory_leak,
    "network_partition": _network_partition,
    "cache_stampede":    _cache_stampede,
}

ALL_INCIDENT_TYPES = list(SCENARIO_GENERATORS.keys())


def generate_incident(
    incident_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> IncidentScenario:
    if seed is None:
        seed = random.randint(0, 9999)
    if incident_type is None:
        incident_type = random.choice(ALL_INCIDENT_TYPES)
    return SCENARIO_GENERATORS[incident_type](seed)


# ── Composite incident mixing (analogous to reasoning_gym composite datasets) ─

@dataclass
class IncidentSpec:
    """Single entry in a CompositeIncidentConfig."""
    incident_type: str
    weight: float = 1.0


@dataclass
class CompositeIncidentConfig:
    """
    Weighted mixture of incident types for curriculum learning.

    Mirrors the reasoning_gym_env composite dataset pattern — lets you
    control the distribution of incident types during training without
    changing the environment code.

    Example — focus on hard incidents:
        config = CompositeIncidentConfig(specs=[
            IncidentSpec("network_partition", weight=3.0),
            IncidentSpec("cache_stampede",    weight=3.0),
            IncidentSpec("db_overload",       weight=1.0),
        ])
        incident_type = config.sample(seed=42)
        scenario = generate_incident(incident_type, seed=42)

    Example — uniform over all types (default behaviour):
        config = CompositeIncidentConfig.uniform()
        incident_type = config.sample()
    """

    specs: list  # List[IncidentSpec]

    @classmethod
    def uniform(cls) -> "CompositeIncidentConfig":
        """Equal weight across all 5 incident types."""
        return cls(specs=[IncidentSpec(t, 1.0) for t in ALL_INCIDENT_TYPES])

    @classmethod
    def hard_only(cls) -> "CompositeIncidentConfig":
        """Only hard incidents (network_partition, cache_stampede)."""
        return cls(specs=[
            IncidentSpec("network_partition", 1.0),
            IncidentSpec("cache_stampede",    1.0),
        ])

    @classmethod
    def curriculum(cls, stage: int) -> "CompositeIncidentConfig":
        """
        Progressive curriculum — start easy, add harder types each stage.

        stage=0: easy only   (bad_deployment)
        stage=1: easy+medium (bad_deployment, db_overload, memory_leak)
        stage=2: all types   (uniform)
        """
        if stage == 0:
            return cls(specs=[IncidentSpec("bad_deployment", 1.0)])
        if stage == 1:
            return cls(specs=[
                IncidentSpec("bad_deployment", 1.0),
                IncidentSpec("db_overload",    1.0),
                IncidentSpec("memory_leak",    1.0),
            ])
        return cls.uniform()

    def sample(self, seed: Optional[int] = None) -> str:
        """Sample an incident type according to weights."""
        rng = random.Random(seed)
        types   = [s.incident_type for s in self.specs]
        weights = [s.weight for s in self.specs]
        return rng.choices(types, weights=weights, k=1)[0]

    def sample_and_generate(self, seed: Optional[int] = None) -> IncidentScenario:
        """Sample an incident type and generate a scenario in one call."""
        if seed is None:
            seed = random.randint(0, 9999)
        incident_type = self.sample(seed=seed)
        return generate_incident(incident_type, seed=seed)
