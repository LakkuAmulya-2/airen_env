# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Incident Engine — fully dynamic production incident generation.

9 incident types (easy -> hard). Zero hardcoded values — every metric,
log line, root cause, company, and recovery trajectory is generated at
runtime from the seed. Same incident type with different seeds produces
a genuinely different scenario every time.

Types:
  1. bad_deployment    (easy)
  2. db_overload       (medium)
  3. memory_leak       (medium)
  4. network_partition (hard)
  5. cache_stampede    (hard)
  6. api_timeout       (medium)
  7. disk_full         (medium)
  8. ssl_cert_expired  (easy)
  9. ddos_attack       (hard)
"""

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class IncidentScenario:
    incident_id: str
    incident_type: str
    severity: str
    root_cause: str
    description: str
    initial_services: Dict[str, Dict[str, Any]]
    initial_metrics: Dict[str, float]
    initial_alerts: List[Dict[str, Any]]
    initial_logs: List[str]
    correct_actions: List[str]
    correct_targets: List[str]
    wrong_action_effects: Dict[str, Dict[str, Any]]
    recovery_trajectory: List[float]
    threat_level: float = 0.5
    degradation_rate: float = 0.05
    cascade_threshold: float = 3.0
    failure_recovery_path: Optional[List[str]] = None
    wrong_fix_description: str = ""
    company_context: str = ""
    difficulty: str = "medium"


# ── Pure runtime generators — no module-level constants ──────────────────────

def _svc(rng: random.Random, status="healthy", latency=50, error_rate=0.0, cpu=30, memory=40):
    return {
        "status": status,
        "latency_ms": latency,
        "error_rate": error_rate,
        "cpu_pct": cpu,
        "memory_pct": memory,
        "uptime_s": rng.randint(3600, 86400),
    }


def _vi(base: int, pct: float, rng: random.Random) -> int:
    d = max(1, int(base * pct))
    return max(1, base + rng.randint(-d, d))


def _vf(base: float, pct: float, rng: random.Random) -> float:
    d = base * pct
    return round(max(0.0, min(1.0, base + rng.uniform(-d, d))), 3)


def _traj(rng: random.Random, start_lo: float, start_hi: float,
          step_lo: float, step_hi: float, n: int) -> List[float]:
    """Generate a monotonically increasing recovery trajectory."""
    start = rng.uniform(start_lo, start_hi)
    traj = [round(start, 2)]
    for _ in range(n - 1):
        traj.append(round(min(1.0, traj[-1] + rng.uniform(step_lo, step_hi)), 2))
    return traj


def _company(rng: random.Random) -> str:
    stacks = [
        ("FinTech", ["PostgreSQL", "Redis", "Node.js"]),
        ("E-commerce", ["MySQL", "Memcached", "Python/Django"]),
        ("SaaS analytics", ["Cassandra", "Kafka", "Go"]),
        ("Healthcare portal", ["PostgreSQL", "Redis", "Java Spring"]),
        ("Gaming backend", ["MongoDB", "Redis", "Node.js"]),
        ("Logistics platform", ["PostgreSQL", "RabbitMQ", "Python"]),
        ("EdTech platform", ["MySQL", "Redis", "Ruby on Rails"]),
        ("Media streaming", ["Cassandra", "Kafka", "Scala"]),
        ("IoT platform", ["TimescaleDB", "MQTT", "Rust"]),
        ("Fintech API", ["CockroachDB", "Redis", "Go"]),
        ("Marketplace", ["PostgreSQL", "Elasticsearch", "Python/FastAPI"]),
        ("DevOps SaaS", ["PostgreSQL", "Redis", "TypeScript/Next.js"]),
    ]
    name, tech = rng.choice(stacks)
    rng.shuffle(tech)
    return f"{name} ({' + '.join(tech)})"


def _red_herring_logs(rng: random.Random, services: List[str]) -> List[str]:
    """Generate red-herring log lines from the actual service names."""
    templates = [
        lambda s: f"[{s.upper()}] Health check: OK ({rng.randint(10,50)}ms)",
        lambda s: f"[{s.upper()}] Metrics nominal — no anomalies",
        lambda s: f"[{s.upper()}] Scheduled job completed successfully",
        lambda s: f"[{s.upper()}] Connection pool: {rng.randint(5,20)}/100 used",
        lambda s: f"[{s.upper()}] Cache hit rate: {rng.randint(80,95)}%",
        lambda s: f"[{s.upper()}] Replication lag: {rng.randint(0,5)}ms",
    ]
    chosen = rng.sample(services, min(2, len(services)))
    return [rng.choice(templates)(s) for s in chosen]


def _bad_deployment(seed: int) -> IncidentScenario:
    rng = random.Random(seed)
    # Version numbers — generated from seed
    major = rng.randint(1, 6)
    minor = rng.randint(0, 9)
    patch = rng.randint(1, 9)
    ver_new = f"v{major}.{minor}.{patch}"
    ver_old = f"v{major}.{minor}.{patch - 1}"
    # Handler — generated from seed
    prefixes = ["Payment", "Order", "User", "Auth", "Checkout", "Billing", "Shipping", "Inventory"]
    handler = rng.choice(prefixes) + "Handler"
    line_no = rng.randint(150, 450)
    deploy_min = rng.randint(8, 55)
    api_err = _vf(0.18, 0.35, rng)
    pay_err = _vf(0.35, 0.25, rng)
    pay_lat = _vi(950, 0.35, rng)
    api_lat = _vi(180, 0.25, rng)
    api_cpu = _vi(45, 0.2, rng)
    pay_cpu = _vi(60, 0.2, rng)
    deploy_h = rng.randint(8, 22)
    deploy_m = rng.randint(0, 59)
    # Exception type varies
    exceptions = ["NullPointerException", "IllegalArgumentException",
                  "ArrayIndexOutOfBoundsException", "ClassCastException"]
    exc = rng.choice(exceptions)
    # Wrong fix description uses actual version
    wrong_desc = (f"Restarting {handler.replace('Handler','')} service does NOT fix "
                  f"the {exc} introduced in {ver_new} — errors persist after restart")
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-DEP",
        incident_type="bad_deployment",
        severity="high",
        root_cause=f"{ver_new} introduced {exc} in {handler}.process() at line {line_no}",
        description=f"Error rate spiked {deploy_min} minutes after deploying {ver_new}",
        company_context=_company(rng),
        difficulty="easy",
        initial_services={
            "api":     _svc(rng, "degraded", api_lat, api_err, api_cpu, 55),
            "db":      _svc(rng, "healthy",  _vi(40,0.1,rng), 0.0, _vi(38,0.1,rng), 52),
            "cache":   _svc(rng, "healthy",  _vi(2,0.2,rng),  0.0, _vi(12,0.2,rng), 22),
            "payment": _svc(rng, "degraded", pay_lat, pay_err, pay_cpu, 65),
        },
        initial_metrics={
            "api_error_rate": api_err,
            "payment_error_rate": pay_err,
            "deployment_age_minutes": float(deploy_min),
            "previous_error_rate": round(rng.uniform(0.005, 0.02), 3),
            "payment_latency_ms": float(pay_lat),
        },
        initial_alerts=[
            {"service": "payment", "severity": "high",
             "message": f"Error rate {pay_err:.0%} (was <2% before {ver_new})",
             "triggered_at": time.time() - deploy_min * 60},
            {"service": "api", "severity": "high",
             "message": f"Error rate {api_err:.0%} — correlated with {ver_new} deploy",
             "triggered_at": time.time() - deploy_min * 60 + rng.randint(10, 30)},
        ],
        initial_logs=[
            f"[PAYMENT] {exc} in {handler}.process() line {line_no}",
            f"[PAYMENT] {exc} in {handler}.process() line {line_no}",
            f"[API] 500 Error: {handler.replace('Handler','').lower()} service exception",
            f"[DEPLOY] {ver_new} deployed at {deploy_h:02d}:{deploy_m:02d} UTC",
            f"[PAYMENT] Stack trace: {handler} -> Validator -> null",
        ] + _red_herring_logs(rng, ["db", "cache"]) + [
            f"[DEPLOY] Previous stable version: {ver_old}",
        ],
        correct_actions=["rollback_deployment"],
        correct_targets=["payment"],
        wrong_action_effects={
            "restart_service": {"payment": {"error_rate": pay_err, "latency_ms": pay_lat}},
            "scale_service":   {"payment": {"error_rate": pay_err}},
        },
        wrong_fix_description=wrong_desc,
        failure_recovery_path=["rollback_deployment"],
        recovery_trajectory=_traj(rng, 0.30, 0.45, 0.12, 0.20, 5),
    )


def _db_overload(seed: int) -> IncidentScenario:
    rng = random.Random(seed)
    # Table and column — generated from seed
    tables = ["orders", "users", "transactions", "events", "sessions",
              "products", "payments", "audit_logs", "notifications", "inventory"]
    table = rng.choice(tables)
    cols = ["user_id", "created_at", "status", "customer_id", "account_id", "tenant_id"]
    col = rng.choice(cols)
    db_cpu = _vi(95, 0.06, rng)
    db_conns = _vi(487, 0.12, rng)
    db_conns_max = rng.choice([500, 1000, 200])
    api_lat = _vi(2400, 0.22, rng)
    api_err = _vf(0.12, 0.25, rng)
    api_cpu = _vi(65, 0.18, rng)
    db_lat = _vi(8500, 0.25, rng)
    rows = rng.randint(800_000, 5_000_000)
    slow1 = round(rng.uniform(6.0, 14.0), 1)
    slow2 = round(rng.uniform(slow1, slow1 + 4.0), 1)
    slow3 = round(rng.uniform(slow2, slow2 + 3.0), 1)
    cache_hit = _vf(0.82, 0.06, rng)
    conn_wait_ms = _vi(45, 0.3, rng)
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-DB",
        incident_type="db_overload",
        severity="critical",
        root_cause=f"Missing index on {table}.{col} — full table scan on {rows:,} rows per query",
        description=f"DB CPU at {db_cpu}% — unindexed {table} queries causing API latency spike",
        company_context=_company(rng),
        difficulty="medium",
        initial_services={
            "api":    _svc(rng, "degraded", api_lat, api_err, api_cpu, 55),
            "db":     _svc(rng, "degraded", db_lat, _vf(0.08,0.2,rng), db_cpu, _vi(88,0.05,rng)),
            "cache":  _svc(rng, "healthy",  _vi(2,0.3,rng), 0.0, _vi(15,0.2,rng), 30),
            "worker": _svc(rng, "healthy",  _vi(120,0.2,rng), _vf(0.02,0.3,rng), _vi(40,0.2,rng), 45),
        },
        initial_metrics={
            "db_cpu_pct": float(db_cpu),
            "db_connections": float(db_conns),
            "db_connections_max": float(db_conns_max),
            "api_p99_latency_ms": float(api_lat),
            "api_error_rate": api_err,
            "cache_hit_rate": cache_hit,
            "db_connection_wait_ms": float(conn_wait_ms),
        },
        initial_alerts=[
            {"service": "db", "severity": "critical",
             "message": f"DB CPU {db_cpu}% for {rng.randint(3,10)} minutes",
             "triggered_at": time.time() - rng.randint(240, 420)},
            {"service": "api", "severity": "high",
             "message": f"API p99 latency {api_lat}ms — DB upstream slow",
             "triggered_at": time.time() - rng.randint(180, 300)},
        ],
        initial_logs=[
            f"[DB] SLOW QUERY: SELECT * FROM {table} WHERE {col}=? ({slow1}s) — seq scan",
            f"[DB] SLOW QUERY: SELECT * FROM {table} WHERE {col}=? ({slow2}s) — seq scan",
            f"[API] 500 Internal Server Error: DB upstream timeout after {conn_wait_ms}ms",
            f"[DB] Connection pool: {db_conns}/{db_conns_max} used — {db_conns_max - db_conns} remaining",
            f"[DB] SLOW QUERY: SELECT * FROM {table} WHERE {col}=? ({slow3}s) — seq scan",
            f"[DB] Sequential scan on {table}.{col}: {rows:,} rows — no index found",
        ] + _red_herring_logs(rng, ["cache", "worker"]),
        correct_actions=["run_diagnostic", "apply_fix"],
        correct_targets=["db", "db"],
        wrong_action_effects={
            "restart_service": {"api": {"status": "down", "error_rate": 1.0}},
            "ignore_alert":    {"db":  {"cpu_pct": min(100, db_cpu + 4), "error_rate": _vf(0.25,0.1,rng)}},
        },
        wrong_fix_description=(f"Restarting API does NOT add the missing index on {table}.{col} "
                               f"— DB stays at {db_cpu}% CPU, queries still scan {rows:,} rows"),
        failure_recovery_path=["run_diagnostic", "apply_fix"],
        recovery_trajectory=_traj(rng, 0.22, 0.35, 0.10, 0.16, 6),
    )


def _memory_leak(seed: int) -> IncidentScenario:
    rng = random.Random(seed)
    mem_pct = _vi(91, 0.06, rng)
    growth = round(rng.uniform(1.2, 4.0), 1)
    buf_gb = round(rng.uniform(1.5, 3.8), 1)
    gc_ms = _vi(450, 0.35, rng)
    worker_lat = _vi(450, 0.25, rng)
    worker_cpu = _vi(55, 0.2, rng)
    # Root cause — generated from seed, not a static string
    leak_types = [
        ("Job handler", "request buffers not released after completion"),
        ("HTTP client", "connection pool not closing idle connections"),
        ("Event emitter", "listeners accumulating without cleanup per request"),
        ("Image processor", "decoded frames retained in heap after processing"),
        ("DB cursor", "ResultSet not closed after query execution"),
        ("WebSocket handler", "message buffers not freed on disconnect"),
        ("Cache client", "response objects pinned in memory after TTL"),
    ]
    component, detail = rng.choice(leak_types)
    root_cause = f"{component}: {detail} — heap growing {growth}MB/min"
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-MEM",
        incident_type="memory_leak",
        severity="high",
        root_cause=root_cause,
        description=f"Worker memory {mem_pct}% — growing {growth}MB/min, OOM imminent",
        company_context=_company(rng),
        difficulty="medium",
        initial_services={
            "api":    _svc(rng, "healthy",  _vi(80,0.15,rng),  _vf(0.01,0.3,rng), _vi(35,0.15,rng), 50),
            "db":     _svc(rng, "healthy",  _vi(45,0.1,rng),   0.0, _vi(40,0.1,rng), 60),
            "cache":  _svc(rng, "healthy",  _vi(2,0.3,rng),    0.0, _vi(10,0.2,rng), 25),
            "worker": _svc(rng, "degraded", worker_lat, _vf(0.05,0.3,rng), worker_cpu, mem_pct),
        },
        initial_metrics={
            "worker_memory_pct": float(mem_pct),
            "worker_memory_growth_mb_per_min": growth,
            "worker_latency_ms": float(worker_lat),
            "worker_gc_pause_ms": float(gc_ms),
            "worker_heap_used_gb": buf_gb,
        },
        initial_alerts=[
            {"service": "worker", "severity": "high",
             "message": f"Worker memory {mem_pct}% — threshold {mem_pct-5}% exceeded",
             "triggered_at": time.time() - rng.randint(400, 800)},
            {"service": "worker", "severity": "medium",
             "message": f"Memory growing {growth}MB/min — continuous increase",
             "triggered_at": time.time() - rng.randint(800, 1400)},
        ],
        initial_logs=[
            f"[WORKER] Memory: {mem_pct}% (threshold: 95%)",
            f"[WORKER] Heap: {buf_gb}GB used (expected: {round(buf_gb*0.25,1)}GB)",
            f"[WORKER] GC pause: {gc_ms}ms — high pressure",
            f"[WORKER] WARNING: {root_cause}",
            f"[WORKER] Memory: {mem_pct + 1}% (+{round(growth/6,1)}MB in 10s)",
        ] + _red_herring_logs(rng, ["api", "db"]),
        correct_actions=["inspect_logs", "restart_service"],
        correct_targets=["worker", "worker"],
        wrong_action_effects={
            "scale_service":       {"worker": {"memory_pct": min(99, mem_pct + 3)}},
            "rollback_deployment": {"worker": {"memory_pct": max(70, mem_pct - 5)}},
            "ignore_alert":        {"worker": {"memory_pct": 96, "status": "down"}},
        },
        wrong_fix_description=(f"Scaling worker adds instances but each leaks via {component} "
                               f"— total heap grows {growth * 2:.1f}MB/min instead of {growth}MB/min"),
        failure_recovery_path=["inspect_logs", "restart_service"],
        recovery_trajectory=_traj(rng, 0.42, 0.58, 0.08, 0.13, 6),
    )


def _network_partition(seed: int) -> IncidentScenario:
    rng = random.Random(seed)
    fw_min = rng.randint(4, 18)
    conn_failures = _vi(847, 0.25, rng)
    # DB type and port — generated from seed
    db_options = [
        (5432, "PostgreSQL"), (3306, "MySQL"), (27017, "MongoDB"),
        (6379, "Redis"), (9200, "Elasticsearch"), (5433, "PostgreSQL replica"),
    ]
    port, db_name = rng.choice(db_options)
    # Network failure reason — generated from seed
    fw_reasons = [
        f"Firewall rule change blocked API to {db_name} on port {port}",
        f"Security group update removed inbound rule for port {port}",
        f"Network ACL misconfiguration dropped packets to {db_name} subnet",
        f"VPC peering route removed — API cannot reach {db_name} subnet",
        f"iptables rule added by automation script blocked port {port}",
    ]
    root_cause = rng.choice(fw_reasons)
    conn_failures_2 = _vi(conn_failures, 0.1, rng)
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-NET",
        incident_type="network_partition",
        severity="critical",
        root_cause=root_cause,
        description=f"API cannot reach {db_name}:{port} — 100% requests failing",
        company_context=_company(rng),
        difficulty="hard",
        initial_services={
            "api":     _svc(rng, "down",     0,   1.0,  _vi(20,0.2,rng), 40),
            "db":      _svc(rng, "healthy",  _vi(45,0.1,rng), 0.0, _vi(35,0.1,rng), 55),
            "cache":   _svc(rng, "healthy",  _vi(2,0.2,rng),  0.0, _vi(10,0.2,rng), 20),
            "worker":  _svc(rng, "down",     0,   1.0,  _vi(15,0.2,rng), 35),
            "network": _svc(rng, "degraded", 0,   1.0,  _vi(5,0.3,rng),  10),
        },
        initial_metrics={
            "api_error_rate": 1.0,
            "api_db_connection_failures": float(conn_failures),
            "db_cpu_pct": float(_vi(35, 0.15, rng)),
            "db_connections": 0.0,
            "network_packet_loss_pct": 100.0,
            "firewall_rule_changes_last_hour": float(rng.randint(1, 5)),
        },
        initial_alerts=[
            {"service": "api", "severity": "critical",
             "message": f"100% error rate — {conn_failures:,} {db_name} connection failures",
             "triggered_at": time.time() - rng.randint(90, 150)},
            {"service": "api", "severity": "critical",
             "message": f"{db_name} connection refused: port {port}",
             "triggered_at": time.time() - rng.randint(80, 130)},
        ],
        initial_logs=[
            f"[API] ERROR: connect ECONNREFUSED db:{port}",
            f"[API] ERROR: connect ECONNREFUSED db:{port}",
            f"[WORKER] ERROR: {db_name} connection failed after {rng.randint(2,5)} retries",
            f"[DB] No incoming connections on port {port} (firewall?)",
            f"[INFRA] Firewall rules updated {fw_min} minutes ago",
            f"[API] {conn_failures_2:,} connection failures in last 60s",
            f"[DB] Listening on port {port} — 0 active clients",
        ] + _red_herring_logs(rng, ["cache"]),
        correct_actions=["run_diagnostic", "apply_fix"],
        correct_targets=["network", "network"],
        wrong_action_effects={
            "restart_service":     {"api": {"status": "down", "error_rate": 1.0}},
            "rollback_deployment": {"api": {"error_rate": 1.0}},
        },
        wrong_fix_description=(f"Restarting API does NOT fix the network rule — "
                               f"port {port} still blocked, {db_name} still unreachable"),
        failure_recovery_path=["run_diagnostic", "apply_fix"],
        recovery_trajectory=_traj(rng, 0.14, 0.24, 0.15, 0.24, 5),
    )


def _cache_stampede(seed: int) -> IncidentScenario:
    rng = random.Random(seed)
    keys_expired = _vi(52847, 0.35, rng)
    db_qps = _vi(12400, 0.25, rng)
    normal_qps = _vi(1200, 0.2, rng)
    db_cpu = _vi(88, 0.10, rng)
    cache_hit = _vf(0.12, 0.35, rng)
    api_lat = _vi(1800, 0.25, rng)
    worker_backlog = _vi(4200, 0.3, rng)
    # Key namespaces — generated from seed
    ns_options = ["user_profile", "session_data", "product_catalog",
                  "cart_items", "auth_tokens", "feed_cache", "search_results"]
    ns1, ns2 = rng.sample(ns_options, 2)
    ttl_hour = rng.randint(6, 23)
    ttl_min = rng.randint(0, 59)
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-CACHE",
        incident_type="cache_stampede",
        severity="high",
        root_cause=(f"Cache TTL batch expiry: {keys_expired:,} {ns1} and {ns2} keys "
                    f"set at {ttl_hour:02d}:{ttl_min:02d} UTC expired simultaneously"),
        description=f"Cache hit rate {cache_hit:.0%} — thundering herd driving DB to {db_cpu}% CPU",
        company_context=_company(rng),
        difficulty="hard",
        initial_services={
            "api":    _svc(rng, "degraded", api_lat, _vf(0.08,0.2,rng), _vi(70,0.1,rng), 60),
            "db":     _svc(rng, "degraded", _vi(4200,0.25,rng), _vf(0.05,0.2,rng), db_cpu, 75),
            "cache":  _svc(rng, "degraded", _vi(5,0.3,rng), 0.0, _vi(20,0.2,rng), 15),
            "worker": _svc(rng, "degraded", _vi(800,0.25,rng), _vf(0.03,0.2,rng), _vi(55,0.15,rng), 50),
        },
        initial_metrics={
            "cache_hit_rate": cache_hit,
            "cache_miss_rate": round(1.0 - cache_hit, 3),
            "db_cpu_pct": float(db_cpu),
            "db_queries_per_sec": float(db_qps),
            "normal_db_qps_baseline": float(normal_qps),
            "api_latency_ms": float(api_lat),
            "worker_job_backlog": float(worker_backlog),
        },
        initial_alerts=[
            {"service": "cache", "severity": "high",
             "message": f"Cache hit rate {cache_hit:.0%} (was 85%) — mass expiry detected",
             "triggered_at": time.time() - rng.randint(150, 240)},
            {"service": "db", "severity": "high",
             "message": f"DB query rate {db_qps:,}/s ({db_qps//normal_qps}x normal baseline)",
             "triggered_at": time.time() - rng.randint(140, 220)},
        ],
        initial_logs=[
            f"[CACHE] Mass expiry: {keys_expired:,} keys expired simultaneously at {ttl_hour:02d}:{ttl_min:02d} UTC",
            f"[DB] Query rate: {db_qps:,}/s (normal: {normal_qps:,}/s)",
            f"[CACHE] Cache miss: {ns1}:* — thundering herd to DB",
            f"[DB] Connection pool: {_vi(490,0.02,rng)}/500 used",
            f"[CACHE] Cache miss: {ns2}:* — thundering herd to DB",
            f"[DB] CPU: {db_cpu}% — query backlog growing",
            f"[WORKER] Job queue backing up: {worker_backlog:,} pending",
        ] + _red_herring_logs(rng, ["api", "worker"]),
        correct_actions=["apply_fix", "scale_service"],
        correct_targets=["cache", "db"],
        wrong_action_effects={
            "restart_service": {"cache": {"status": "down", "error_rate": 0.5}},
            "ignore_alert":    {"db": {"cpu_pct": min(100, db_cpu + 7), "error_rate": _vf(0.15,0.1,rng)}},
        },
        wrong_fix_description=(f"Restarting cache evicts all remaining warm {ns1}/{ns2} entries "
                               f"— stampede intensifies, DB CPU spikes further"),
        failure_recovery_path=["apply_fix", "scale_service"],
        recovery_trajectory=_traj(rng, 0.28, 0.38, 0.08, 0.13, 7),
    )


# ── NEW INCIDENT TYPES ────────────────────────────────────────────────────────

def _api_timeout(seed: int) -> IncidentScenario:
    rng = random.Random(seed)
    # Upstream service — generated from seed
    upstream_names = ["payment-gateway", "auth-service", "email-service",
                      "sms-provider", "fraud-detection", "kyc-service",
                      "shipping-api", "tax-calculator", "identity-provider"]
    upstream = rng.choice(upstream_names)
    timeout_ms = _vi(30000, 0.25, rng)
    api_err = _vf(0.45, 0.22, rng)
    api_lat = _vi(31000, 0.18, rng)
    api_cpu = _vi(75, 0.18, rng)
    worker_lat = _vi(28000, 0.25, rng)
    pool_waiting = _vi(48, 0.15, rng)
    pool_max = rng.choice([50, 100, 200])
    retry_count = rng.randint(2, 5)
    # Timeout cause varies
    timeout_causes = [
        f"{upstream} database connection pool exhausted",
        f"{upstream} CPU at 100% — not processing requests",
        f"{upstream} network route flapping — intermittent packet loss",
        f"{upstream} memory OOM — process unresponsive",
        f"{upstream} deployment in progress — health checks failing",
    ]
    root_cause = rng.choice(timeout_causes)
    upstream_metric_key = upstream.replace("-", "_") + "_timeout_rate"
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-TOUT",
        incident_type="api_timeout",
        severity="high",
        root_cause=f"{root_cause} — connection pool exhausted after {timeout_ms}ms timeout",
        description=f"API requests timing out waiting for {upstream} ({timeout_ms//1000}s timeout)",
        company_context=_company(rng),
        difficulty="medium",
        initial_services={
            "api":      _svc(rng, "degraded", api_lat, api_err, api_cpu, 60),
            "db":       _svc(rng, "healthy",  _vi(45,0.1,rng), 0.0, _vi(35,0.1,rng), 50),
            "cache":    _svc(rng, "healthy",  _vi(2,0.2,rng),  0.0, _vi(12,0.2,rng), 22),
            "worker":   _svc(rng, "degraded", worker_lat, _vf(0.3,0.15,rng), _vi(55,0.15,rng), 45),
            "upstream": _svc(rng, "down",     0, 1.0, _vi(5,0.3,rng), 10),
        },
        initial_metrics={
            "api_error_rate": api_err,
            "api_p99_latency_ms": float(api_lat),
            upstream_metric_key: 1.0,
            "connection_pool_waiting": float(pool_waiting),
            "connection_pool_max": float(pool_max),
            "upstream_timeout_ms": float(timeout_ms),
        },
        initial_alerts=[
            {"service": "api", "severity": "high",
             "message": f"API error rate {api_err:.0%} — {upstream} timeouts",
             "triggered_at": time.time() - rng.randint(240, 400)},
            {"service": "upstream", "severity": "critical",
             "message": f"{upstream} not responding (timeout: {timeout_ms//1000}s)",
             "triggered_at": time.time() - rng.randint(260, 420)},
        ],
        initial_logs=[
            f"[API] Timeout waiting for {upstream}: {timeout_ms}ms exceeded",
            f"[API] Timeout waiting for {upstream}: {timeout_ms}ms exceeded",
            f"[WORKER] {upstream} call failed: connection timeout after {timeout_ms}ms",
            f"[API] Connection pool: {pool_waiting}/{pool_max} threads waiting for {upstream}",
            f"[API] Circuit breaker OPEN for {upstream} after {retry_count} failures",
            f"[API] Retry attempt {retry_count}/{retry_count} failed: {upstream} unreachable",
        ] + _red_herring_logs(rng, ["db", "cache"]),
        correct_actions=["run_diagnostic", "apply_fix"],
        correct_targets=["upstream", "upstream"],
        wrong_action_effects={
            "restart_service": {"api": {"status": "degraded", "error_rate": api_err}},
            "scale_service":   {"api": {"error_rate": api_err}},
        },
        wrong_fix_description=(f"Scaling API adds more threads but they all timeout on {upstream} "
                               f"— pool exhaustion worsens from {pool_waiting} to {pool_max} waiting"),
        failure_recovery_path=["run_diagnostic", "apply_fix"],
        recovery_trajectory=_traj(rng, 0.22, 0.32, 0.13, 0.20, 5),
        threat_level=_vf(0.6, 0.12, rng),
        degradation_rate=round(rng.uniform(0.04, 0.08), 3),
    )


def _disk_full(seed: int) -> IncidentScenario:
    rng = random.Random(seed)
    disk_pct = _vi(97, 0.03, rng)
    disk_gb = rng.choice([200, 250, 500, 1000])
    log_gb = round(rng.uniform(disk_gb * 0.75, disk_gb * 0.92), 1)
    log_dir_gb = round(log_gb * rng.uniform(0.75, 0.90), 1)
    hours_since_rotation = rng.randint(18, 96)
    # Log path varies
    log_paths = ["/var/log/nginx/access.log", "/var/log/app/requests.log",
                 "/var/log/api/combined.log", "/var/log/service/output.log"]
    log_path = rng.choice(log_paths)
    # Cron failure reason varies
    cron_failures = [
        "logrotate cron job failed — permission denied on log directory",
        "logrotate cron job not running — systemd timer disabled",
        "log rotation script exited with error — disk quota exceeded",
        "logrotate misconfigured — rotate count set to 0",
    ]
    cron_fail = rng.choice(cron_failures)
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-DISK",
        incident_type="disk_full",
        severity="critical",
        root_cause=f"{cron_fail} — {log_path} grew to {log_dir_gb}GB unrotated",
        description=f"Disk {disk_pct}% full ({log_gb}GB/{disk_gb}GB) — services failing to write",
        company_context=_company(rng),
        difficulty="medium",
        initial_services={
            "api":    _svc(rng, "degraded", _vi(200,0.25,rng), _vf(0.08,0.25,rng), _vi(40,0.2,rng), 55),
            "db":     _svc(rng, "degraded", _vi(150,0.2,rng),  _vf(0.05,0.2,rng),  _vi(45,0.2,rng), 60),
            "cache":  _svc(rng, "healthy",  _vi(2,0.3,rng),    0.0, _vi(12,0.2,rng), 22),
            "worker": _svc(rng, "degraded", _vi(500,0.25,rng), _vf(0.12,0.25,rng), _vi(50,0.2,rng), 55),
            "infra":  _svc(rng, "degraded", 0, _vf(0.3,0.2,rng), _vi(15,0.2,rng), 20),
        },
        initial_metrics={
            "disk_usage_pct": float(disk_pct),
            "disk_used_gb": log_gb,
            "disk_total_gb": float(disk_gb),
            "log_dir_size_gb": log_dir_gb,
            "log_rotation_last_success_hours_ago": float(hours_since_rotation),
            "disk_write_errors_per_min": float(_vi(45, 0.3, rng)),
        },
        initial_alerts=[
            {"service": "infra", "severity": "critical",
             "message": f"Disk {disk_pct}% full ({log_gb}GB/{disk_gb}GB) — critical",
             "triggered_at": time.time() - rng.randint(400, 800)},
            {"service": "db", "severity": "high",
             "message": "DB write errors: No space left on device",
             "triggered_at": time.time() - rng.randint(200, 400)},
        ],
        initial_logs=[
            f"[INFRA] Disk usage: {disk_pct}% ({log_gb}GB / {disk_gb}GB)",
            "[DB] ERROR: could not write to file: No space left on device",
            f"[LOGROTATE] ERROR: {cron_fail}",
            f"[API] WARNING: Cannot write to {log_path} — disk full",
            f"[INFRA] {log_path}: {log_dir_gb}GB (unbounded growth)",
            f"[LOGROTATE] Last successful rotation: {hours_since_rotation}h ago",
            "[WORKER] Job failed: cannot write output — disk full",
        ] + _red_herring_logs(rng, ["cache"]),
        correct_actions=["run_diagnostic", "apply_fix"],
        correct_targets=["infra", "infra"],
        wrong_action_effects={
            "restart_service": {"db": {"error_rate": _vf(0.15,0.1,rng), "status": "degraded"}},
            "scale_service":   {"worker": {"error_rate": _vf(0.2,0.1,rng)}},
        },
        wrong_fix_description=(f"Restarting services does NOT free disk space — "
                               f"{log_path} still {log_dir_gb}GB, errors resume immediately"),
        failure_recovery_path=["run_diagnostic", "apply_fix"],
        recovery_trajectory=_traj(rng, 0.32, 0.42, 0.13, 0.20, 5),
        threat_level=_vf(0.7, 0.1, rng),
        degradation_rate=round(rng.uniform(0.05, 0.09), 3),
        cascade_threshold=round(rng.uniform(2.0, 3.0), 1),
    )


def _ssl_cert_expired(seed: int) -> IncidentScenario:
    rng = random.Random(seed)
    # Domain generated from seed — not a static list
    subdomains = ["api", "app", "service", "gateway", "auth", "payments", "data", "admin"]
    tlds = ["com", "io", "net", "co", "dev", "cloud"]
    domain = f"{rng.choice(subdomains)}.{rng.randint(1000,9999)}-company.{rng.choice(tlds)}"
    expired_hours = rng.randint(1, 8)
    tls_failures = _vi(2400, 0.25, rng)
    days_since_renewal = rng.randint(25, 120)
    # Renewal failure reason varies
    renewal_failures = [
        "permission denied on /etc/letsencrypt — certbot running as wrong user",
        "DNS challenge failed — CNAME record missing for _acme-challenge",
        "HTTP-01 challenge failed — port 80 blocked by firewall",
        "certbot cron job disabled after system migration",
        "renewal script exited non-zero — disk full prevented cert write",
    ]
    renewal_fail = rng.choice(renewal_failures)
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-SSL",
        incident_type="ssl_cert_expired",
        severity="critical",
        root_cause=f"TLS cert for {domain} expired {expired_hours}h ago — {renewal_fail}",
        description=f"100% HTTPS requests to {domain} failing with TLS handshake error",
        company_context=_company(rng),
        difficulty="easy",
        initial_services={
            "api":    _svc(rng, "down",    0,   1.0,  _vi(20,0.2,rng), 40),
            "db":     _svc(rng, "healthy", _vi(45,0.1,rng), 0.0, _vi(35,0.1,rng), 55),
            "cache":  _svc(rng, "healthy", _vi(2,0.2,rng),  0.0, _vi(10,0.2,rng), 20),
            "worker": _svc(rng, "healthy", _vi(120,0.1,rng), _vf(0.01,0.3,rng), _vi(30,0.1,rng), 40),
            "tls":    _svc(rng, "down",    0,   1.0,  5,  5),
        },
        initial_metrics={
            "api_error_rate": 1.0,
            "tls_handshake_failures": float(tls_failures),
            "cert_expiry_hours_ago": float(expired_hours),
            "cert_renewal_last_attempt_days_ago": float(days_since_renewal),
            "https_requests_blocked": float(_vi(tls_failures, 0.1, rng)),
        },
        initial_alerts=[
            {"service": "api", "severity": "critical",
             "message": f"100% HTTPS requests failing — TLS handshake error on {domain}",
             "triggered_at": time.time() - expired_hours * 3600},
            {"service": "tls", "severity": "critical",
             "message": f"Certificate expired: {domain} ({expired_hours}h ago)",
             "triggered_at": time.time() - expired_hours * 3600},
        ],
        initial_logs=[
            f"[API] SSL_ERROR_RX_RECORD_TOO_LONG: {domain}",
            f"[NGINX] SSL_CTX_use_certificate_file: certificate expired for {domain}",
            f"[CLIENT] TLS handshake failed: certificate has expired ({domain})",
            f"[CERTBOT] ERROR: {renewal_fail}",
            f"[NGINX] Certificate expiry: {domain} expired {expired_hours}h ago",
            f"[CERTBOT] Last renewal attempt: {days_since_renewal} days ago",
        ] + _red_herring_logs(rng, ["db", "worker"]),
        correct_actions=["run_diagnostic", "apply_fix"],
        correct_targets=["tls", "tls"],
        wrong_action_effects={
            "restart_service":     {"api": {"status": "down", "error_rate": 1.0}},
            "rollback_deployment": {"api": {"error_rate": 1.0}},
        },
        wrong_fix_description=(f"Restarting nginx/API does NOT renew the expired cert for {domain} "
                               f"— TLS handshake errors persist until cert is renewed"),
        failure_recovery_path=["run_diagnostic", "apply_fix"],
        recovery_trajectory=_traj(rng, 0.12, 0.22, 0.22, 0.32, 4),
        threat_level=_vf(0.8, 0.1, rng),
        degradation_rate=round(rng.uniform(0.01, 0.03), 3),
        cascade_threshold=round(rng.uniform(3.5, 5.0), 1),
    )


def _ddos_attack(seed: int) -> IncidentScenario:
    rng = random.Random(seed)
    rps_normal = _vi(1200, 0.25, rng)
    rps_attack = _vi(85000, 0.35, rng)
    attacker_ips = rng.randint(2000, 20000)
    bw_gbps = round(rng.uniform(7.5, 14.0), 1)
    bw_limit = rng.choice([10, 20, 40])
    req_per_ip = rng.randint(30, 300)
    active_conns = _vi(65000, 0.15, rng)
    db_lat = _vi(5000, 0.25, rng)
    db_cpu = _vi(88, 0.12, rng)
    # Attack vector varies
    vectors = [
        f"Layer 7 HTTP GET flood from {attacker_ips:,} IPs — {rps_attack:,} req/s",
        f"HTTP POST flood targeting /api/search from {attacker_ips:,} IPs",
        f"Slowloris attack from {attacker_ips:,} IPs — connection exhaustion",
        f"Amplification attack via {attacker_ips:,} open resolvers — {bw_gbps}Gbps",
    ]
    attack_vector = rng.choice(vectors)
    return IncidentScenario(
        incident_id=f"INC-{seed:04d}-DDOS",
        incident_type="ddos_attack",
        severity="critical",
        root_cause=f"{attack_vector} overwhelming API ({rps_attack//rps_normal}x normal traffic)",
        description=f"Traffic {rps_attack//rps_normal}x normal — API saturated, cascading to DB",
        company_context=_company(rng),
        difficulty="hard",
        initial_services={
            "api":     _svc(rng, "down",     0,   1.0,  99, 80),
            "db":      _svc(rng, "degraded", db_lat, _vf(0.15,0.15,rng), db_cpu, 75),
            "cache":   _svc(rng, "degraded", _vi(50,0.3,rng), _vf(0.05,0.2,rng), _vi(70,0.1,rng), 60),
            "worker":  _svc(rng, "degraded", _vi(2000,0.25,rng), _vf(0.2,0.15,rng), _vi(80,0.1,rng), 65),
            "network": _svc(rng, "degraded", 0, _vf(0.8,0.1,rng), 95, 20),
        },
        initial_metrics={
            "api_requests_per_sec": float(rps_attack),
            "normal_rps_baseline": float(rps_normal),
            "api_cpu_pct": 99.0,
            "unique_attacker_ips": float(attacker_ips),
            "api_error_rate": 1.0,
            "network_bandwidth_gbps": bw_gbps,
            "network_bandwidth_limit_gbps": float(bw_limit),
            "active_connections": float(active_conns),
        },
        initial_alerts=[
            {"service": "api", "severity": "critical",
             "message": f"Traffic {rps_attack//rps_normal}x normal ({rps_attack:,} req/s) — DDoS suspected",
             "triggered_at": time.time() - rng.randint(120, 240)},
            {"service": "network", "severity": "critical",
             "message": f"Bandwidth {bw_gbps}Gbps — approaching {bw_limit}Gbps limit",
             "triggered_at": time.time() - rng.randint(110, 220)},
        ],
        initial_logs=[
            f"[API] Request rate: {rps_attack:,}/s (normal: {rps_normal:,}/s)",
            f"[NGINX] Too many connections: {active_conns:,} active",
            f"[SECURITY] Suspicious traffic from {attacker_ips:,} unique IPs",
            f"[API] CPU: 99% — request queue overflowing",
            f"[DB] Connection pool exhausted — cascading from API overload",
            f"[SECURITY] {req_per_ip} req/s per attacker IP — flood pattern",
            f"[NETWORK] Bandwidth: {bw_gbps}Gbps (limit: {bw_limit}Gbps)",
        ] + _red_herring_logs(rng, ["cache", "worker"]),
        correct_actions=["run_diagnostic", "apply_fix", "scale_service"],
        correct_targets=["network", "network", "api"],
        wrong_action_effects={
            "restart_service":     {"api": {"status": "down", "error_rate": 1.0}},
            "rollback_deployment": {"api": {"error_rate": 1.0}},
        },
        wrong_fix_description=(f"Restarting API under active {attack_vector[:40]} "
                               f"— immediately overwhelmed again, no improvement"),
        failure_recovery_path=["run_diagnostic", "apply_fix", "scale_service"],
        recovery_trajectory=_traj(rng, 0.15, 0.25, 0.10, 0.16, 6),
        threat_level=_vf(0.9, 0.08, rng),
        degradation_rate=round(rng.uniform(0.06, 0.10), 3),
        cascade_threshold=round(rng.uniform(1.5, 2.5), 1),
    )


SCENARIO_GENERATORS = {
    "bad_deployment":    _bad_deployment,
    "db_overload":       _db_overload,
    "memory_leak":       _memory_leak,
    "network_partition": _network_partition,
    "cache_stampede":    _cache_stampede,
    "api_timeout":       _api_timeout,
    "disk_full":         _disk_full,
    "ssl_cert_expired":  _ssl_cert_expired,
    "ddos_attack":       _ddos_attack,
}

ALL_INCIDENT_TYPES = list(SCENARIO_GENERATORS.keys())

# Difficulty groupings for curriculum learning
EASY_INCIDENTS   = ["bad_deployment", "ssl_cert_expired"]
MEDIUM_INCIDENTS = ["db_overload", "memory_leak", "api_timeout", "disk_full"]
HARD_INCIDENTS   = ["network_partition", "cache_stampede", "ddos_attack"]


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
        """Only hard incidents (network_partition, cache_stampede, ddos_attack)."""
        return cls(specs=[IncidentSpec(t, 1.0) for t in HARD_INCIDENTS])

    @classmethod
    def curriculum(cls, stage: int) -> "CompositeIncidentConfig":
        """
        Progressive curriculum — start easy, add harder types each stage.

        stage=0: easy only   (bad_deployment, ssl_cert_expired)
        stage=1: easy+medium (+ db_overload, memory_leak, api_timeout, disk_full)
        stage=2: all types   (uniform across all 9)
        """
        if stage == 0:
            return cls(specs=[IncidentSpec(t, 1.0) for t in EASY_INCIDENTS])
        if stage == 1:
            return cls(specs=[IncidentSpec(t, 1.0) for t in EASY_INCIDENTS + MEDIUM_INCIDENTS])
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
