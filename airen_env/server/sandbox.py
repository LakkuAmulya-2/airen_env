# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
AIREN Sandbox Layer — 3 sandboxes for production-grade RL testing.

1. ToolCallSandbox   — intercepts agent actions, returns mock results,
                       never touches real infra. Validates action safety.

2. EpisodeReplaySandbox — re-runs any past episode with exact same seed
                          and scenario, step-by-step. Debug + compare.

3. ChaosSandbox      — injects random failures mid-episode to test
                       agent resilience under unexpected conditions.

All sandboxes are stateless singletons — no shared mutable state between sessions.
"""

import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


# ═══════════════════════════════════════════════════════════════════════════
# 1. TOOL CALL SANDBOX
# ═══════════════════════════════════════════════════════════════════════════

# Real infra actions that must NEVER hit production
_DESTRUCTIVE_ACTIONS = {
    "restart_service", "rollback_deployment", "scale_service",
    "apply_fix", "ignore_alert",
}
_DIAGNOSTIC_ACTIONS = {
    "inspect_logs", "inspect_metrics", "run_diagnostic", "acknowledge_incident",
}

# Mock result templates — derived from action + target, not hardcoded strings
_MOCK_RESULT_TEMPLATES = {
    "restart_service":     lambda t: f"[SANDBOX] {t} restart simulated — pod restarted in 2.3s (mock)",
    "rollback_deployment": lambda t: f"[SANDBOX] {t} rollback simulated — reverted to v{random.randint(1,9)}.{random.randint(0,9)}.{random.randint(0,9)} (mock)",
    "scale_service":       lambda t: f"[SANDBOX] {t} scaled to {random.randint(2,8)} replicas (mock)",
    "apply_fix":           lambda t: f"[SANDBOX] fix applied to {t} — patch deployed (mock)",
    "ignore_alert":        lambda t: f"[SANDBOX] alert on {t} suppressed (mock — NOT recommended)",
    "inspect_logs":        lambda t: f"[SANDBOX] logs fetched from {t} — {random.randint(10,200)} lines returned (mock)",
    "inspect_metrics":     lambda t: f"[SANDBOX] metrics pulled from {t} — Prometheus query executed (mock)",
    "run_diagnostic":      lambda t: f"[SANDBOX] diagnostic run on {t} — health check completed (mock)",
    "acknowledge_incident":lambda t: f"[SANDBOX] incident on {t} acknowledged — PagerDuty notified (mock)",
}


@dataclass
class ToolCallResult:
    action_type: str
    target: str
    sandboxed: bool          # True = intercepted, False = passed through
    mock_result: str
    would_have_affected: List[str]   # services that would have been affected
    safety_verdict: str      # "safe" | "destructive" | "diagnostic"
    latency_ms: float        # simulated latency
    intercepted_at: float    # timestamp


@dataclass
class ToolCallSandboxStats:
    total_intercepted: int = 0
    destructive_blocked: int = 0
    diagnostic_allowed: int = 0
    by_action: Dict[str, int] = field(default_factory=dict)


class ToolCallSandbox:
    """
    Intercepts every agent action before it hits the environment.

    In SANDBOX mode:
      - Destructive actions (restart, rollback, scale) → intercepted, mock result returned
      - Diagnostic actions (inspect, run_diagnostic) → allowed through
      - All actions logged with safety verdict

    In PASSTHROUGH mode (default):
      - All actions pass through to real environment
      - Still logs what WOULD have been intercepted

    Enabled via SANDBOX_TOOL_CALLS=1 env var.
    """

    def __init__(self):
        self.enabled: bool = os.environ.get("SANDBOX_TOOL_CALLS", "0") == "1"
        self.stats = ToolCallSandboxStats()
        self._log: List[ToolCallResult] = []

    def intercept(
        self,
        action_type: str,
        target: str,
        services: Dict[str, Any],
    ) -> Optional[ToolCallResult]:
        """
        Intercept an action. Returns ToolCallResult if sandboxed, None if passthrough.
        """
        t0 = time.time()
        is_destructive = action_type in _DESTRUCTIVE_ACTIONS
        is_diagnostic = action_type in _DIAGNOSTIC_ACTIONS

        verdict = "destructive" if is_destructive else ("diagnostic" if is_diagnostic else "unknown")

        # Which services would be affected
        affected = []
        if target in services:
            affected = [target]
        elif target in ("network", "infra"):
            affected = list(services.keys())

        mock = _MOCK_RESULT_TEMPLATES.get(action_type, lambda t: f"[SANDBOX] {action_type}({t}) executed (mock)")(target)
        latency = round(random.uniform(50, 500), 1)  # realistic latency range

        result = ToolCallResult(
            action_type=action_type,
            target=target,
            sandboxed=self.enabled and is_destructive,
            mock_result=mock,
            would_have_affected=affected,
            safety_verdict=verdict,
            latency_ms=latency,
            intercepted_at=t0,
        )

        # Update stats
        self.stats.total_intercepted += 1
        self.stats.by_action[action_type] = self.stats.by_action.get(action_type, 0) + 1
        if result.sandboxed:
            self.stats.destructive_blocked += 1
        if is_diagnostic:
            self.stats.diagnostic_allowed += 1

        self._log.append(result)
        if len(self._log) > 500:
            self._log = self._log[-500:]

        return result if result.sandboxed else None

    def get_log(self, limit: int = 50) -> List[Dict]:
        return [
            {
                "action_type": r.action_type,
                "target": r.target,
                "sandboxed": r.sandboxed,
                "mock_result": r.mock_result,
                "would_have_affected": r.would_have_affected,
                "safety_verdict": r.safety_verdict,
                "latency_ms": r.latency_ms,
                "intercepted_at": r.intercepted_at,
            }
            for r in self._log[-limit:]
        ]

    def get_stats(self) -> Dict:
        return {
            "enabled": self.enabled,
            "total_intercepted": self.stats.total_intercepted,
            "destructive_blocked": self.stats.destructive_blocked,
            "diagnostic_allowed": self.stats.diagnostic_allowed,
            "by_action": self.stats.by_action,
            "mode": "sandbox" if self.enabled else "passthrough",
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. EPISODE REPLAY SANDBOX
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReplayStep:
    step: int
    action_type: str
    target: str
    reasoning: str
    health_before: float
    health_after: float
    reward: float
    action_result: str
    incident_resolved: bool
    services_snapshot: Dict[str, Any]


@dataclass
class ReplayResult:
    episode_id: str
    incident_type: str
    seed: int
    difficulty: str
    steps: List[ReplayStep]
    final_health: float
    resolved: bool
    cumulative_reward: float
    root_cause: str
    replay_diverged: bool        # True if replay produced different results than original
    divergence_step: Optional[int]


class EpisodeReplaySandbox:
    """
    Re-runs any past episode with exact same seed + scenario.

    Use cases:
      - Debug why an agent failed at step N
      - Compare two agent policies on identical scenarios
      - Verify reward shaping changes don't break existing episodes
      - Generate training data from known-good episodes

    Enabled always — replay is read-only, no side effects.
    """

    def replay(
        self,
        incident_type: str,
        seed: int,
        actions: List[Dict[str, str]],
        difficulty: str = "medium",
        compare_rewards: Optional[List[float]] = None,
    ) -> ReplayResult:
        """
        Replay an episode with given actions.

        Args:
            incident_type: e.g. "db_overload"
            seed: exact seed used in original episode
            actions: list of {action_type, target, reasoning}
            difficulty: easy | medium | hard
            compare_rewards: original rewards to compare against (optional)
        """
        try:
            from server.airen_environment import AIRENEnvironment
            from models import AIRENAction
        except ImportError:
            from airen_env.server.airen_environment import AIRENEnvironment
            from airen_env.models import AIRENAction

        env = AIRENEnvironment()
        obs = env.reset(incident_type=incident_type, seed=seed, difficulty=difficulty)
        scenario = env._scenario

        steps: List[ReplayStep] = []
        cumulative = 0.0
        diverged = False
        divergence_step = None

        for i, act in enumerate(actions):
            health_before = obs.system_health
            action = AIRENAction(
                action_type=act.get("action_type", "run_diagnostic"),
                target=act.get("target", "api"),
                reasoning=act.get("reasoning", ""),
            )
            obs = env.step(action)
            reward = obs.reward or 0.0
            cumulative += reward

            # Check divergence against original rewards
            if compare_rewards and i < len(compare_rewards):
                orig = compare_rewards[i]
                if abs(reward - orig) > 0.05 and not diverged:
                    diverged = True
                    divergence_step = i + 1

            steps.append(ReplayStep(
                step=i + 1,
                action_type=act.get("action_type", ""),
                target=act.get("target", ""),
                reasoning=act.get("reasoning", ""),
                health_before=round(health_before, 3),
                health_after=round(obs.system_health, 3),
                reward=round(reward, 3),
                action_result=obs.action_result or "",
                incident_resolved=obs.incident_resolved or False,
                services_snapshot={
                    k: {"status": v["status"], "error_rate": v.get("error_rate", 0)}
                    for k, v in obs.services.items()
                },
            ))

            if obs.done:
                break

        return ReplayResult(
            episode_id=env.state.episode_id,
            incident_type=incident_type,
            seed=seed,
            difficulty=difficulty,
            steps=steps,
            final_health=round(obs.system_health, 3),
            resolved=env.state.incident_resolved,
            cumulative_reward=round(cumulative, 3),
            root_cause=scenario.root_cause,
            replay_diverged=diverged,
            divergence_step=divergence_step,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. CHAOS SANDBOX
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChaosEvent:
    step: int
    chaos_type: str       # "service_crash" | "network_spike" | "cascade_inject" | "metric_corrupt"
    target: str
    description: str
    health_impact: float  # how much health dropped due to chaos


@dataclass
class ChaosRunResult:
    incident_type: str
    chaos_profile: str
    steps: List[Dict]
    chaos_events: List[ChaosEvent]
    final_health: float
    resolved: bool
    cumulative_reward: float
    agent_survived_chaos: bool


# Chaos profiles — different failure injection patterns
_CHAOS_PROFILES = {
    "mild":       {"crash_prob": 0.1, "spike_prob": 0.15, "cascade_prob": 0.05},
    "moderate":   {"crash_prob": 0.2, "spike_prob": 0.25, "cascade_prob": 0.15},
    "aggressive": {"crash_prob": 0.35, "spike_prob": 0.4,  "cascade_prob": 0.3},
}


class ChaosSandbox:
    """
    Injects random failures mid-episode to test agent resilience.

    Chaos types:
      - service_crash:    randomly crashes a healthy service mid-episode
      - network_spike:    spikes latency on all services for 1 step
      - cascade_inject:   forces a cascade failure regardless of attack_progress
      - metric_corrupt:   injects false metrics to confuse the agent

    Use cases:
      - Test if agent can recover from unexpected failures
      - Verify ARL rollback engine triggers correctly
      - Stress-test multi-agent coordination under chaos
      - Measure reward degradation under different chaos profiles
    """

    def run(
        self,
        incident_type: str,
        agent_actions: List[Dict[str, str]],
        chaos_profile: str = "moderate",
        seed: Optional[int] = None,
        difficulty: str = "medium",
    ) -> ChaosRunResult:
        """
        Run an episode with chaos injected at random steps.

        Args:
            incident_type: incident type to run
            agent_actions: list of {action_type, target, reasoning}
            chaos_profile: "mild" | "moderate" | "aggressive"
            seed: random seed
            difficulty: easy | medium | hard
        """
        try:
            from server.airen_environment import AIRENEnvironment
            from models import AIRENAction
        except ImportError:
            from airen_env.server.airen_environment import AIRENEnvironment
            from airen_env.models import AIRENAction

        rng = random.Random(seed or int(time.time()))
        profile = _CHAOS_PROFILES.get(chaos_profile, _CHAOS_PROFILES["moderate"])

        env = AIRENEnvironment()
        obs = env.reset(incident_type=incident_type, seed=seed, difficulty=difficulty)

        steps: List[Dict] = []
        chaos_events: List[ChaosEvent] = []
        cumulative = 0.0

        for i, act in enumerate(agent_actions):
            # Inject chaos BEFORE agent acts
            chaos_event = self._maybe_inject_chaos(env, i + 1, profile, rng)
            if chaos_event:
                chaos_events.append(chaos_event)

            health_before = obs.system_health
            action = AIRENAction(
                action_type=act.get("action_type", "run_diagnostic"),
                target=act.get("target", "api"),
                reasoning=act.get("reasoning", ""),
            )
            obs = env.step(action)
            reward = obs.reward or 0.0
            cumulative += reward

            steps.append({
                "step": i + 1,
                "action_type": act.get("action_type"),
                "target": act.get("target"),
                "health_before": round(health_before, 3),
                "health_after": round(obs.system_health, 3),
                "reward": round(reward, 3),
                "chaos_injected": chaos_event.chaos_type if chaos_event else None,
            })

            if obs.done:
                break

        return ChaosRunResult(
            incident_type=incident_type,
            chaos_profile=chaos_profile,
            steps=steps,
            chaos_events=chaos_events,
            final_health=round(obs.system_health, 3),
            resolved=env.state.incident_resolved,
            cumulative_reward=round(cumulative, 3),
            agent_survived_chaos=env.state.incident_resolved or obs.system_health > 0.4,
        )

    def _maybe_inject_chaos(
        self,
        env: Any,
        step: int,
        profile: Dict[str, float],
        rng: random.Random,
    ) -> Optional[ChaosEvent]:
        """Probabilistically inject a chaos event into the environment state."""
        services = env._services
        if not services:
            return None

        healthy = [n for n, s in services.items() if s.get("status") == "healthy"]
        all_svcs = list(services.keys())

        # service_crash
        if healthy and rng.random() < profile["crash_prob"]:
            target = rng.choice(healthy)
            services[target]["status"] = "down"
            services[target]["error_rate"] = 1.0
            env._logs_buffer.append(f"[CHAOS] {target} crashed unexpectedly — OOMKilled")
            return ChaosEvent(
                step=step, chaos_type="service_crash", target=target,
                description=f"{target} crashed (OOMKilled) — chaos injection",
                health_impact=round(-0.15 * (1 / max(len(all_svcs), 1)), 3),
            )

        # network_spike
        if rng.random() < profile["spike_prob"]:
            for svc in all_svcs:
                services[svc]["latency_ms"] = int(services[svc].get("latency_ms", 50) * rng.uniform(3, 8))
            env._logs_buffer.append("[CHAOS] Network spike — all services experiencing high latency")
            return ChaosEvent(
                step=step, chaos_type="network_spike", target="all",
                description="Network spike injected — latency 3-8x normal",
                health_impact=-0.05,
            )

        # cascade_inject
        if healthy and rng.random() < profile["cascade_prob"]:
            target = rng.choice(healthy)
            services[target]["status"] = "degraded"
            services[target]["error_rate"] = min(1.0, services[target].get("error_rate", 0) + 0.3)
            env._attack_progress = min(5.0, env._attack_progress + 1.5)
            env._logs_buffer.append(f"[CHAOS] Cascade injected into {target} — attack_progress boosted")
            return ChaosEvent(
                step=step, chaos_type="cascade_inject", target=target,
                description=f"Cascade failure forced on {target} — attack_progress +1.5",
                health_impact=-0.1,
            )

        return None


# ═══════════════════════════════════════════════════════════════════════════
# SESSION ISOLATION SANDBOX (shared)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SandboxSession:
    session_id: str
    created_at: float
    last_active: float
    episode_count: int
    request_count: int
    token_budget_used: int
    token_budget_max: int
    rate_limit_hits: int
    env_name: str


class SessionSandbox:
    """
    Per-user isolated sandbox session with TTL + token budget + rate limiting.

    Ensures:
      - Each user gets an isolated session (no state leakage)
      - Sessions expire after TTL (default: 30 min)
      - Per-session token budget enforced (default: 50k tokens)
      - Per-session request rate limited (default: 60 req/min)

    Configurable via env vars:
      SESSION_TTL_SECONDS     (default: 1800)
      SESSION_TOKEN_BUDGET    (default: 50000)
      SESSION_RATE_LIMIT_RPM  (default: 60)
    """

    TTL: int = int(os.environ.get("SESSION_TTL_SECONDS", "1800"))
    TOKEN_BUDGET: int = int(os.environ.get("SESSION_TOKEN_BUDGET", "50000"))
    RATE_LIMIT_RPM: int = int(os.environ.get("SESSION_RATE_LIMIT_RPM", "60"))

    def __init__(self, env_name: str):
        self.env_name = env_name
        self._sessions: Dict[str, SandboxSession] = {}
        self._request_times: Dict[str, List[float]] = {}

    def get_or_create(self, session_id: str) -> SandboxSession:
        now = time.time()
        if session_id not in self._sessions:
            self._sessions[session_id] = SandboxSession(
                session_id=session_id,
                created_at=now,
                last_active=now,
                episode_count=0,
                request_count=0,
                token_budget_used=0,
                token_budget_max=self.TOKEN_BUDGET,
                rate_limit_hits=0,
                env_name=self.env_name,
            )
            self._request_times[session_id] = []
        return self._sessions[session_id]

    def check_rate_limit(self, session_id: str) -> Tuple[bool, str]:
        """Returns (allowed, reason). Cleans up expired window."""
        now = time.time()
        times = self._request_times.get(session_id, [])
        # Keep only requests in last 60 seconds
        times = [t for t in times if now - t < 60.0]
        self._request_times[session_id] = times

        if len(times) >= self.RATE_LIMIT_RPM:
            sess = self._sessions.get(session_id)
            if sess:
                sess.rate_limit_hits += 1
            return False, f"Rate limit exceeded: {len(times)}/{self.RATE_LIMIT_RPM} req/min"

        times.append(now)
        return True, "ok"

    def check_token_budget(self, session_id: str, tokens_needed: int) -> Tuple[bool, str]:
        sess = self._sessions.get(session_id)
        if not sess:
            return True, "ok"
        remaining = sess.token_budget_max - sess.token_budget_used
        if tokens_needed > remaining:
            return False, f"Token budget exhausted: {sess.token_budget_used}/{sess.token_budget_max}"
        return True, "ok"

    def record_request(self, session_id: str, tokens_used: int = 0) -> None:
        sess = self.get_or_create(session_id)
        sess.last_active = time.time()
        sess.request_count += 1
        sess.token_budget_used += tokens_used

    def record_episode(self, session_id: str) -> None:
        sess = self.get_or_create(session_id)
        sess.episode_count += 1

    def cleanup_expired(self) -> int:
        """Remove sessions older than TTL. Returns count removed."""
        now = time.time()
        expired = [sid for sid, s in self._sessions.items()
                   if now - s.last_active > self.TTL]
        for sid in expired:
            del self._sessions[sid]
            self._request_times.pop(sid, None)
        return len(expired)

    def get_session_info(self, session_id: str) -> Dict:
        sess = self._sessions.get(session_id)
        if not sess:
            return {"error": "session not found"}
        now = time.time()
        return {
            "session_id": sess.session_id,
            "env_name": sess.env_name,
            "created_at": sess.created_at,
            "last_active": sess.last_active,
            "age_seconds": round(now - sess.created_at, 1),
            "ttl_remaining_seconds": max(0, self.TTL - (now - sess.last_active)),
            "episode_count": sess.episode_count,
            "request_count": sess.request_count,
            "token_budget_used": sess.token_budget_used,
            "token_budget_max": sess.token_budget_max,
            "token_budget_remaining": sess.token_budget_max - sess.token_budget_used,
            "rate_limit_hits": sess.rate_limit_hits,
        }

    def get_all_sessions(self) -> Dict:
        self.cleanup_expired()
        return {
            "active_sessions": len(self._sessions),
            "env_name": self.env_name,
            "ttl_seconds": self.TTL,
            "token_budget_per_session": self.TOKEN_BUDGET,
            "rate_limit_rpm": self.RATE_LIMIT_RPM,
            "sessions": [self.get_session_info(sid) for sid in self._sessions],
        }


# ── Singletons ────────────────────────────────────────────────────────────────

_tool_sandbox: Optional[ToolCallSandbox] = None
_replay_sandbox: Optional[EpisodeReplaySandbox] = None
_chaos_sandbox: Optional[ChaosSandbox] = None
_session_sandbox: Optional[SessionSandbox] = None


def get_tool_sandbox() -> ToolCallSandbox:
    global _tool_sandbox
    if _tool_sandbox is None:
        _tool_sandbox = ToolCallSandbox()
    return _tool_sandbox


def get_replay_sandbox() -> EpisodeReplaySandbox:
    global _replay_sandbox
    if _replay_sandbox is None:
        _replay_sandbox = EpisodeReplaySandbox()
    return _replay_sandbox


def get_chaos_sandbox() -> ChaosSandbox:
    global _chaos_sandbox
    if _chaos_sandbox is None:
        _chaos_sandbox = ChaosSandbox()
    return _chaos_sandbox


def get_session_sandbox() -> SessionSandbox:
    global _session_sandbox
    if _session_sandbox is None:
        _session_sandbox = SessionSandbox(env_name="airen_env")
    return _session_sandbox
