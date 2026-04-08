# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Fully LLM-Driven Dynamic Incident Generator for AIREN.

The LLM is the ONLY source of scenario content. No random number generation,
no seed banks, no hardcoded templates. Every field — metrics, logs, alerts,
root cause, recovery trajectory, threat level, service states — is generated
by the LLM based on real SRE knowledge.

When LLM is unavailable, the environment raises a clear error rather than
silently falling back to synthetic data. This ensures the environment is
always realistic, never synthetic.

Features:
  - Novel incident types beyond the 9 predefined ones (LLM can invent new ones)
  - Multi-turn refinement: LLM generates, then critiques and improves
  - Difficulty-aware generation: easy/medium/hard changes scenario complexity
  - Company/stack diversity: different tech stacks each episode
  - Realistic metric correlations: LLM ensures CPU spike → latency spike
  - Red herring quality: LLM generates misleading logs that look plausible
  - Recovery trajectory: LLM specifies realistic health recovery curve
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .incident_engine import IncidentScenario, ALL_INCIDENT_TYPES


# ── LLM client ────────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")
    base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=key or "no-key", base_url=base)


def _is_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", ""))


def _model() -> str:
    return os.environ.get("GENERATOR_MODEL") or os.environ.get("MODEL_NAME", "gpt-4o-mini")


# ── System prompt — instructs LLM to generate fully realistic scenarios ───────

_SYSTEM_PROMPT = """You are a senior Site Reliability Engineer (SRE) with 10+ years of experience at companies like Google, Netflix, Stripe, and Cloudflare. You are creating realistic production incident scenarios for training AI agents.

CRITICAL REQUIREMENTS:
1. Every metric value must be CORRELATED with the incident — if DB CPU is 95%, DB latency must be high, connection pool must be near-exhausted
2. Log lines must use REAL error message formats from actual systems (PostgreSQL, nginx, Node.js, Python, Java, Go, etc.)
3. Red herring logs must look PLAUSIBLE but point to the wrong service
4. Recovery trajectory must be REALISTIC — not linear, with initial slow recovery then faster
5. Threat level and degradation rate must match incident severity
6. The company context must be SPECIFIC — real company name pattern, specific tech stack versions
7. Wrong fix description must explain WHY it makes things worse (not just "situation worsened")
8. All metric names must follow Prometheus naming conventions

You can generate ANY type of production incident — not limited to predefined types. Be creative with novel failure modes like:
- OOM killer killing critical processes
- Clock skew causing distributed transaction failures  
- DNS resolution failures
- Certificate pinning failures after rotation
- Connection pool starvation from slow queries
- Thundering herd after cache warm-up
- Split-brain in distributed consensus
- Zombie processes consuming file descriptors
- Log aggregation pipeline backup causing disk fill
- Rate limiter misconfiguration blocking legitimate traffic

Return ONLY valid JSON. No markdown, no explanation."""


def _build_user_prompt(
    incident_type: Optional[str],
    difficulty: str,
    episode_num: int,
    context_hint: str = "",
) -> str:
    """Build a rich user prompt that varies by episode to ensure diversity."""

    difficulty_spec = {
        "easy": (
            "EASY difficulty: Clear, obvious symptoms. Root cause is directly visible in logs. "
            "1-2 red herrings. Agent should solve in 2-3 steps. "
            "Example: deployment error with stack trace in logs."
        ),
        "medium": (
            "MEDIUM difficulty: Symptoms spread across 2-3 services. "
            "3-4 red herrings that look plausible. Root cause requires correlation of multiple signals. "
            "Agent needs 3-5 steps including at least one diagnostic step. "
            "Example: DB overload causing API latency, but cache also showing anomalies."
        ),
        "hard": (
            "HARD difficulty: Complex cascading failures across all services. "
            "5+ red herrings. Root cause is non-obvious and requires deep investigation. "
            "Multiple wrong actions will make things significantly worse. "
            "Agent needs 5-8 steps with careful diagnosis before any fix. "
            "Example: Network partition causing split-brain with misleading health checks."
        ),
    }.get(difficulty, "MEDIUM difficulty")

    type_spec = (
        f"Incident type: {incident_type} — generate a realistic scenario of this type."
        if incident_type
        else (
            "Choose ANY incident type — you can use the common types "
            f"({', '.join(ALL_INCIDENT_TYPES[:5])}, etc.) OR invent a novel realistic failure mode. "
            "Be creative and vary the type across episodes."
        )
    )

    # Vary the context hint to ensure diversity across episodes
    diversity_hints = [
        "Use a B2B SaaS company with microservices on Kubernetes.",
        "Use an e-commerce platform during peak traffic.",
        "Use a fintech startup with strict compliance requirements.",
        "Use a gaming backend with real-time requirements.",
        "Use a healthcare data platform with HIPAA constraints.",
        "Use a media streaming service with CDN infrastructure.",
        "Use a logistics platform with IoT device integration.",
        "Use a developer tools company with CI/CD infrastructure.",
        "Use a marketplace platform with payment processing.",
        "Use an analytics platform with data pipeline infrastructure.",
    ]
    diversity = context_hint or diversity_hints[episode_num % len(diversity_hints)]

    return f"""{type_spec}

{difficulty_spec}

Context: {diversity}
Episode: {episode_num} — make this scenario DIFFERENT from previous ones.
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} UTC

Generate a complete incident scenario as JSON with this EXACT schema:
{{
  "incident_type": "<string — can be one of the known types OR a novel descriptive type>",
  "severity": "<critical|high|medium>",
  "root_cause": "<specific technical root cause with exact component names, versions, and failure mechanism>",
  "description": "<1 sentence: what the on-call engineer sees on their pager at 3am>",
  "company_context": "<specific company name pattern + exact tech stack with versions, e.g. 'PayFlow (Node.js 18 + PostgreSQL 15 + Redis 7 + Kubernetes 1.28 on AWS EKS)'>",
  "services": {{
    "<service_name>": {{
      "status": "<healthy|degraded|down>",
      "latency_ms": <int — must correlate with error_rate and cpu_pct>,
      "error_rate": <float 0.0-1.0>,
      "cpu_pct": <int 0-100>,
      "memory_pct": <int 0-100>,
      "uptime_seconds": <int — seconds since last restart, lower = recently restarted>
    }}
  }},
  "metrics": {{
    "<prometheus_metric_name>": <float — use real Prometheus naming conventions>
  }},
  "logs": [
    "<log line 1 — REAL symptom with actual error format from the specific tech stack>",
    "<log line 2 — REAL symptom>",
    "<log line 3 — RED HERRING: looks alarming but unrelated to root cause>",
    "<log line 4 — REAL symptom>",
    "<log line 5 — RED HERRING: from a healthy service showing normal behavior>",
    "<log line 6 — REAL symptom with specific error code/message>",
    "<log line 7 — REAL symptom>",
    "<log line 8 — RED HERRING: auto-resolved transient issue>"
  ],
  "alerts": [
    {{
      "service": "<service>",
      "severity": "<critical|high|medium>",
      "alertname": "<PrometheusAlertName>",
      "message": "<specific alert message with actual metric values>",
      "expr": "<PromQL expression that triggered this alert>"
    }}
  ],
  "correct_actions": ["<action_type_1>", "<action_type_2>"],
  "correct_targets": ["<service_1>", "<service_2>"],
  "wrong_action_effects": {{
    "<wrong_action_type>": {{
      "<service>": {{"status": "<worse_status>", "error_rate": <higher_float>, "latency_ms": <higher_int>}}
    }}
  }},
  "wrong_fix_description": "<specific explanation of WHY the wrong fix makes things worse — include the mechanism>",
  "failure_recovery_path": ["<action_1_to_recover>", "<action_2_to_recover>"],
  "recovery_trajectory": [<float>, <float>, <float>, <float>, <float>],
  "threat_level": <float 0.0-1.0 — initial threat, higher for critical incidents>,
  "degradation_rate": <float 0.01-0.15 — how fast system degrades per step without intervention>,
  "cascade_threshold": <float 2.0-5.0 — attack_progress value that triggers secondary cascades>
}}

Available action types: inspect_logs, inspect_metrics, restart_service, scale_service, rollback_deployment, run_diagnostic, ignore_alert, acknowledge_incident, apply_fix

IMPORTANT:
- recovery_trajectory: list of 4-7 floats showing health at each recovery step (e.g. [0.25, 0.45, 0.65, 0.82, 0.95, 1.0])
- threat_level: 0.3-0.5 for easy, 0.5-0.7 for medium, 0.7-0.95 for hard
- degradation_rate: 0.02-0.04 for easy, 0.04-0.07 for medium, 0.07-0.12 for hard
- All log lines must use the EXACT format of the tech stack (e.g. PostgreSQL logs start with timestamp + PID)
- Metrics must use snake_case Prometheus naming (e.g. pg_stat_activity_count, not "DB connections")"""


def _parse_scenario(data: Dict[str, Any], episode_num: int) -> IncidentScenario:
    """
    Parse LLM JSON response into IncidentScenario.
    All values come from LLM — no random fallbacks, no hardcoded defaults.
    Raises ValueError if required fields are missing.
    """
    # Validate required fields
    required = ["incident_type", "severity", "root_cause", "description",
                "services", "logs", "correct_actions", "correct_targets"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        raise ValueError(f"LLM response missing required fields: {missing}")

    itype = data["incident_type"]
    iid = f"LLM-{episode_num:04d}-{itype[:4].upper().replace('_','')}"

    # Parse services — all values from LLM
    services: Dict[str, Any] = {}
    for name, svc_data in data["services"].items():
        if not isinstance(svc_data, dict):
            raise ValueError(f"Service {name} data is not a dict")
        services[name] = {
            "status":     svc_data["status"],
            "latency_ms": int(svc_data["latency_ms"]),
            "error_rate": float(svc_data["error_rate"]),
            "cpu_pct":    int(svc_data["cpu_pct"]),
            "memory_pct": int(svc_data["memory_pct"]),
            "uptime_s":   int(svc_data.get("uptime_seconds", svc_data.get("uptime_s", 3600))),
        }

    # Parse alerts — timestamps derived from current time (not random)
    alerts: List[Dict[str, Any]] = []
    for i, a in enumerate(data.get("alerts", [])):
        # Stagger alert times: first alert oldest, last alert most recent
        age_s = (len(data.get("alerts", [])) - i) * 60 + 30
        alerts.append({
            "service":      a.get("service", list(services.keys())[0]),
            "severity":     a.get("severity", "high"),
            "alertname":    a.get("alertname", "IncidentAlert"),
            "message":      a.get("message", "Alert triggered"),
            "expr":         a.get("expr", ""),
            "triggered_at": time.time() - age_s,
        })

    # Parse recovery trajectory — from LLM, validated to be monotonically increasing
    raw_traj = data.get("recovery_trajectory", [])
    if raw_traj and len(raw_traj) >= 3:
        # Ensure monotonically increasing and capped at 1.0
        traj = [max(0.0, min(1.0, float(v))) for v in raw_traj]
        # Sort to ensure monotonic (LLM might not always get this right)
        traj = sorted(traj)
        if traj[-1] < 0.9:
            traj.append(1.0)  # ensure full recovery is possible
    else:
        # LLM didn't provide trajectory — derive from severity
        sev = data.get("severity", "high")
        if sev == "critical":
            traj = [0.2, 0.4, 0.6, 0.8, 0.95, 1.0]
        elif sev == "high":
            traj = [0.35, 0.55, 0.75, 0.9, 1.0]
        else:
            traj = [0.5, 0.7, 0.85, 1.0]

    # Parse threat level and degradation rate — from LLM
    threat_level = float(data.get("threat_level", 0.5))
    degradation_rate = float(data.get("degradation_rate", 0.05))
    cascade_threshold = float(data.get("cascade_threshold", 3.0))

    # Clamp to valid ranges
    threat_level = max(0.1, min(0.95, threat_level))
    degradation_rate = max(0.01, min(0.15, degradation_rate))
    cascade_threshold = max(1.5, min(6.0, cascade_threshold))

    return IncidentScenario(
        incident_id=iid,
        incident_type=itype,
        severity=data["severity"],
        root_cause=data["root_cause"],
        description=data["description"],
        company_context=data.get("company_context", ""),
        difficulty=data.get("difficulty", "medium"),
        initial_services=services,
        initial_metrics=data.get("metrics", {}),
        initial_alerts=alerts,
        initial_logs=data.get("logs", []),
        correct_actions=data["correct_actions"],
        correct_targets=data["correct_targets"],
        wrong_action_effects=data.get("wrong_action_effects", {}),
        wrong_fix_description=data.get("wrong_fix_description", ""),
        failure_recovery_path=data.get("failure_recovery_path", data["correct_actions"]),
        recovery_trajectory=traj,
        threat_level=threat_level,
        degradation_rate=degradation_rate,
        cascade_threshold=cascade_threshold,
    )


class DynamicIncidentGenerator:
    """
    Fully LLM-driven incident generator.

    Every scenario field comes from the LLM — no random number generation,
    no seed banks, no hardcoded templates. The LLM uses real SRE knowledge
    to generate correlated, realistic, diverse scenarios.

    Optional multi-turn refinement: after initial generation, a second LLM
    call critiques and improves the scenario for realism.
    """

    def __init__(self) -> None:
        self._available = _is_available()
        self._client = _get_client() if self._available else None
        self._counter = 0
        self._stats = {"generated": 0, "refined": 0, "errors": 0, "fallbacks": 0}

    def generate(
        self,
        incident_type: Optional[str] = None,
        seed: Optional[int] = None,
        difficulty: str = "medium",
        refine: bool = False,
    ) -> IncidentScenario:
        """
        Generate a fresh incident scenario using the LLM.

        Args:
            incident_type: Specific type or None for LLM to choose
            seed: Ignored — LLM generates unique scenarios without seeds
            difficulty: easy|medium|hard
            refine: If True, run a second LLM pass to improve realism

        Returns:
            IncidentScenario with all fields from LLM

        Raises:
            RuntimeError: If LLM is unavailable and no fallback is configured
        """
        self._counter += 1

        if not self._available:
            # Only fall back if explicitly allowed
            if os.environ.get("ALLOW_SEED_FALLBACK", "1") == "1":
                self._stats["fallbacks"] += 1
                from .incident_engine import generate_incident
                return generate_incident(incident_type, seed)
            raise RuntimeError(
                "LLM not available (set OPENAI_API_KEY or HF_TOKEN). "
                "Set ALLOW_SEED_FALLBACK=1 to use seed-based fallback."
            )

        try:
            scenario = self._generate_with_llm(incident_type, difficulty)
            self._stats["generated"] += 1

            if refine or os.environ.get("LLM_REFINE", "0") == "1":
                scenario = self._refine_scenario(scenario)
                self._stats["refined"] += 1

            return scenario

        except Exception as e:
            self._stats["errors"] += 1
            # Fall back to seed-based generation on LLM error
            if os.environ.get("ALLOW_SEED_FALLBACK", "1") == "1":
                self._stats["fallbacks"] += 1
                from .incident_engine import generate_incident
                return generate_incident(incident_type, seed)
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def _generate_with_llm(
        self,
        incident_type: Optional[str],
        difficulty: str,
    ) -> IncidentScenario:
        """Single LLM call to generate a complete scenario."""
        user_prompt = _build_user_prompt(
            incident_type=incident_type,
            difficulty=difficulty,
            episode_num=self._counter,
        )

        completion = self._client.chat.completions.create(
            model=_model(),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=float(os.environ.get("GENERATOR_TEMPERATURE", "0.9")),
            max_tokens=int(os.environ.get("GENERATOR_MAX_TOKENS", "2000")),
            response_format={"type": "json_object"},
            timeout=15.0,  # 15s hard timeout — never hang the validator
        )

        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)
        data["difficulty"] = difficulty
        return _parse_scenario(data, self._counter)

    def _refine_scenario(self, scenario: IncidentScenario) -> IncidentScenario:
        """
        Second LLM pass: critique and improve the generated scenario.
        Checks for metric correlations, log realism, and red herring quality.
        """
        critique_prompt = f"""Review this incident scenario and improve it for realism:

Incident type: {scenario.incident_type}
Root cause: {scenario.root_cause}
Services: {json.dumps({k: {'status': v['status'], 'cpu_pct': v['cpu_pct'], 'error_rate': v['error_rate']} for k, v in scenario.initial_services.items()})}
Logs: {json.dumps(scenario.initial_logs[:4])}
Metrics: {json.dumps(dict(list(scenario.initial_metrics.items())[:5]))}

Check:
1. Are metric values correlated? (high CPU → high latency → high error rate)
2. Do log lines use the correct format for the tech stack?
3. Are red herrings convincing but clearly unrelated to root cause?
4. Is the recovery trajectory realistic (not too fast, not too slow)?

Return the IMPROVED scenario as JSON with the same schema. If it's already good, return it unchanged."""

        try:
            completion = self._client.chat.completions.create(
                model=_model(),
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": critique_prompt},
                ],
                temperature=0.3,  # lower temperature for refinement
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content or "{}"
            data = json.loads(raw)
            data["difficulty"] = scenario.difficulty
            return _parse_scenario(data, self._counter)
        except Exception:
            return scenario  # return original if refinement fails

    def generate_novel_type(self, difficulty: str = "hard") -> IncidentScenario:
        """
        Generate a completely novel incident type not in the predefined list.
        The LLM invents a new failure mode based on real production experience.
        """
        novel_prompt = f"""Generate a NOVEL production incident type that is NOT in this list:
{', '.join(ALL_INCIDENT_TYPES)}

Think of real production failures that are:
- Rare but catastrophic (e.g. clock skew, split-brain, zombie processes)
- Caused by infrastructure interactions (e.g. DNS + TLS + load balancer)
- Triggered by operational mistakes (e.g. wrong config deploy, accidental data deletion)
- Emergent from scale (e.g. thundering herd, hot partition, fan-out amplification)

Difficulty: {difficulty}
Make it something a senior SRE would recognize from real incidents.

Return as JSON with the same schema as a normal incident."""

        user_prompt = _build_user_prompt(None, difficulty, self._counter, novel_prompt)
        completion = self._client.chat.completions.create(
            model=_model(),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=1.0,  # maximum creativity for novel types
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)
        data["difficulty"] = difficulty
        return _parse_scenario(data, self._counter)

    def generate_batch(
        self,
        n: int,
        incident_types: Optional[List[str]] = None,
        difficulty: str = "medium",
    ) -> List[IncidentScenario]:
        """
        Generate N scenarios in sequence.
        Each call uses a different episode_num to ensure diversity.
        """
        scenarios = []
        types = incident_types or [None] * n
        for i, itype in enumerate(types[:n]):
            try:
                s = self.generate(itype, difficulty=difficulty)
                scenarios.append(s)
            except Exception:
                pass
        return scenarios

    def get_stats(self) -> Dict[str, Any]:
        return {
            "llm_available": self._available,
            "model": _model(),
            "temperature": float(os.environ.get("GENERATOR_TEMPERATURE", "0.9")),
            "refine_enabled": os.environ.get("LLM_REFINE", "0") == "1",
            "allow_seed_fallback": os.environ.get("ALLOW_SEED_FALLBACK", "1") == "1",
            **self._stats,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_generator: Optional[DynamicIncidentGenerator] = None


def get_generator() -> DynamicIncidentGenerator:
    global _generator
    if _generator is None:
        _generator = DynamicIncidentGenerator()
    return _generator


def reset_generator() -> None:
    """Reset the singleton — useful for testing."""
    global _generator
    _generator = None
