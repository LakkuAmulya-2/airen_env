# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
LLM-Driven Dynamic Incident Generator for AIREN.

Instead of hardcoded scenarios, GPT-4o-mini generates fresh, realistic
production incidents every episode. This makes the environment:
  - Open-ended: infinite unique scenarios
  - Realistic: based on real SRE knowledge
  - Varied: different companies, stack sizes, failure modes
  - Unpredictable: agent cannot memorize scenarios

Falls back to seed bank if LLM unavailable.
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .incident_engine import (
    IncidentScenario, ALL_INCIDENT_TYPES,
    _svc, generate_incident
)


_SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) creating realistic production incident scenarios for AI agent training.

Generate a realistic production incident scenario. The scenario must:
1. Be based on real-world distributed system failures
2. Have clear symptoms visible in logs/metrics (but NOT the root cause)
3. Include red herring log lines that could mislead the agent
4. Have cascading effects across multiple services
5. Be solvable with specific SRE actions

Return ONLY valid JSON matching this exact schema:
{
  "incident_type": "<one of: db_overload|memory_leak|network_partition|bad_deployment|cache_stampede>",
  "severity": "<critical|high|medium>",
  "root_cause": "<specific technical root cause, 1 sentence>",
  "description": "<what an on-call engineer would see, 1 sentence>",
  "company_context": "<fictional company name and stack, e.g. 'FinTech startup using PostgreSQL + Redis + Node.js'>",
  "services": {
    "api":    {"status": "<healthy|degraded|down>", "latency_ms": <int>, "error_rate": <0.0-1.0>, "cpu_pct": <int>, "memory_pct": <int>},
    "db":     {"status": "<healthy|degraded|down>", "latency_ms": <int>, "error_rate": <0.0-1.0>, "cpu_pct": <int>, "memory_pct": <int>},
    "cache":  {"status": "<healthy|degraded|down>", "latency_ms": <int>, "error_rate": <0.0-1.0>, "cpu_pct": <int>, "memory_pct": <int>},
    "worker": {"status": "<healthy|degraded|down>", "latency_ms": <int>, "error_rate": <0.0-1.0>, "cpu_pct": <int>, "memory_pct": <int>}
  },
  "metrics": {
    "<metric_name>": <float>
  },
  "logs": [
    "<log line 1 — real symptom>",
    "<log line 2 — real symptom>",
    "<log line 3 — red herring>",
    "<log line 4 — real symptom>",
    "<log line 5 — red herring>",
    "<log line 6 — real symptom>",
    "<log line 7 — real symptom>",
    "<log line 8 — red herring>"
  ],
  "alerts": [
    {"service": "<service>", "severity": "<critical|high|medium>", "message": "<alert message>"},
    {"service": "<service>", "severity": "<high|medium>", "message": "<alert message>"}
  ],
  "correct_actions": ["<action_type_1>", "<action_type_2>"],
  "correct_targets": ["<service_1>", "<service_2>"],
  "wrong_action_effects": {
    "<wrong_action_type>": {"<service>": {"status": "<worse_status>", "error_rate": <higher_float>}}
  }
}

Available action types: inspect_logs, inspect_metrics, restart_service, scale_service, rollback_deployment, run_diagnostic, ignore_alert, acknowledge_incident, apply_fix

Make it realistic — use actual error messages, real metric names, plausible log formats."""


class DynamicIncidentGenerator:
    """
    Generates fresh production incidents using GPT-4o-mini.

    Each call produces a unique, realistic scenario that an agent
    cannot have memorized. The LLM draws from real SRE knowledge
    to create varied failure modes, stack traces, and cascading effects.
    """

    def __init__(self):
        _key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")
        _base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
        self._model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        self._available = bool(_key)
        self._client = OpenAI(api_key=_key or "no-key", base_url=_base) if self._available else None
        self._counter = 0
        self._cache: List[IncidentScenario] = []  # pre-generated pool

    def generate(
        self,
        incident_type: Optional[str] = None,
        seed: Optional[int] = None,
        difficulty: str = "medium",
    ) -> IncidentScenario:
        """
        Generate a fresh incident scenario.

        Args:
            incident_type: Specific type or None for random
            seed: For reproducibility (used in fallback)
            difficulty: easy|medium|hard — affects complexity of scenario
        """
        self._counter += 1

        if self._available:
            try:
                return self._generate_with_llm(incident_type, difficulty)
            except Exception as e:
                pass  # fallback to seed bank

        # Fallback: seed bank
        return generate_incident(incident_type, seed)

    def _generate_with_llm(
        self, incident_type: Optional[str], difficulty: str
    ) -> IncidentScenario:
        """Use GPT-4o-mini to generate a novel incident scenario."""

        difficulty_guidance = {
            "easy":   "Make it obvious — clear symptoms pointing directly to root cause",
            "medium": "Add some ambiguity — 2-3 red herring logs, symptoms spread across services",
            "hard":   "Make it complex — multiple cascading failures, many red herrings, non-obvious root cause",
        }.get(difficulty, "medium difficulty")

        type_guidance = f"Incident type: {incident_type}" if incident_type else f"Choose randomly from: {ALL_INCIDENT_TYPES}"

        user_prompt = f"""{type_guidance}
Difficulty: {difficulty} — {difficulty_guidance}
Make it unique — use a different company, stack, and specific failure details each time.
Current timestamp context: {time.strftime('%Y-%m-%d %H:%M')} UTC"""

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,   # high diversity
            max_tokens=1200,
            response_format={"type": "json_object"},
        )

        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)

        # Parse into IncidentScenario
        itype = data.get("incident_type", incident_type or random.choice(ALL_INCIDENT_TYPES))
        iid = f"DYN-{self._counter:04d}-{itype[:3].upper()}"

        # Build services with uptime
        services = {}
        for name, svc_data in data.get("services", {}).items():
            services[name] = {
                "status":     svc_data.get("status", "healthy"),
                "latency_ms": svc_data.get("latency_ms", 50),
                "error_rate": svc_data.get("error_rate", 0.0),
                "cpu_pct":    svc_data.get("cpu_pct", 30),
                "memory_pct": svc_data.get("memory_pct", 40),
                "uptime_s":   random.randint(3600, 86400),
            }

        # Build alerts with timestamps
        alerts = []
        for a in data.get("alerts", []):
            alerts.append({
                "service":      a.get("service", "api"),
                "severity":     a.get("severity", "high"),
                "message":      a.get("message", "Alert triggered"),
                "triggered_at": time.time() - random.randint(60, 600),
            })

        return IncidentScenario(
            incident_id=iid,
            incident_type=itype,
            severity=data.get("severity", "high"),
            root_cause=data.get("root_cause", "Unknown root cause"),
            description=data.get("description", "Production incident detected"),
            initial_services=services,
            initial_metrics=data.get("metrics", {}),
            initial_alerts=alerts,
            initial_logs=data.get("logs", ["[SYSTEM] Incident detected"]),
            correct_actions=data.get("correct_actions", ["run_diagnostic", "apply_fix"]),
            correct_targets=data.get("correct_targets", ["api", "api"]),
            wrong_action_effects=data.get("wrong_action_effects", {}),
            recovery_trajectory=[0.4, 0.6, 0.8, 0.95, 1.0],
        )

    def prefetch(self, n: int = 3) -> None:
        """Pre-generate N scenarios in background for low-latency episodes."""
        for _ in range(n):
            try:
                s = self._generate_with_llm(None, "medium")
                self._cache.append(s)
            except Exception:
                break

    def get_stats(self) -> Dict[str, Any]:
        return {
            "llm_available": self._available,
            "scenarios_generated": self._counter,
            "model": self._model,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_generator: Optional[DynamicIncidentGenerator] = None

def get_generator() -> DynamicIncidentGenerator:
    global _generator
    if _generator is None:
        _generator = DynamicIncidentGenerator()
    return _generator
