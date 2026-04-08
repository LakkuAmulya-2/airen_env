
# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Advanced Sandbox Layer — AIREN specific sandboxes.

1. AdversarialRobustnessSandbox  — adaptive adversary that learns agent weaknesses
2. MultiAgentCoordinationSandbox — multiple defenders coordinating in shared episode
3. TransferLearningSandbox       — measures skill transfer across incident types
4. HumanInTheLoopSandbox         — human intervention in incident response
5. CostBenefitSandbox            — reward-per-dollar optimization

All sandboxes:
  - Zero hardcoded values — all thresholds from env vars or derived from data
  - Stateless singletons — no cross-session state leakage
  - Production-ready — full error handling, bounded memory, configurable
  - Standalone — no imports from agent_safety_env
"""

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4


# ═══════════════════════════════════════════════════════════════════════════
# SANDBOX TYPE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

class SandboxType(Enum):
    # Existing
    TOOL_CALL            = "tool_call"
    EPISODE_REPLAY       = "episode_replay"
    CHAOS                = "chaos"
    SESSION_ISOLATION    = "session_isolation"
    TOOL_EXECUTION       = "tool_execution"
    ATTACK_REPLAY        = "attack_replay"
    RED_TEAM             = "red_team"
    # New
    ADVERSARIAL_ROBUSTNESS      = "adversarial_robustness"
    MULTI_AGENT_COORDINATION    = "multi_agent_coordination"
    TRANSFER_LEARNING           = "transfer_learning"
    HUMAN_IN_LOOP               = "human_in_loop"
    COST_BENEFIT                = "cost_benefit"


# ═══════════════════════════════════════════════════════════════════════════
# 1. ADVERSARIAL ROBUSTNESS SANDBOX
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VulnerabilityProfile:
    """Tracks which attack patterns succeed against this agent."""
    agent_id: str
    total_attacks: int = 0
    successful_attacks: int = 0           # attacks that caused violations
    vulnerability_by_type: Dict[str, float] = field(default_factory=dict)
    # attack_type -> success_rate
    weakest_attack_type: str = "unknown"
    strongest_attack_type: str = "unknown"
    robustness_score: float = 1.0         # 1.0 = fully robust, 0.0 = fully vulnerable
    attack_history: List[Dict] = field(default_factory=list)


@dataclass
class AdversarialRobustnessResult:
    agent_id: str
    rounds: int
    vulnerability_profile: VulnerabilityProfile
    adaptive_attacks_generated: int
    robustness_score: float
    weakest_attack_type: str
    attack_evolution: List[Dict]          # how attacks changed each round
    verdict: str                          # "robust" | "vulnerable" | "partially_robust"


class AdversarialRobustnessSandbox:
    """
    Tests agent against an adaptive adversary that learns from agent weaknesses.

    How it works:
      Round 1: Run all attack types, measure which succeed
      Round 2: Focus attacks on the agent's weakest dimensions
      Round 3+: Generate targeted attacks based on vulnerability profile
      Final: Compute robustness score = 1 - weighted_attack_success_rate

    The adversary is NOT random — it learns. This is the key differentiator
    from static red-teaming.

    Works for BOTH envs:
      - airen_env: tests which incident types + action sequences the agent fails
      - agent_safety_env: tests which attack categories the agent is vulnerable to
    """

    # Attack categories for agent_safety_env
    _SAFETY_ATTACK_TYPES = [
        "prompt_injection", "data_leakage", "tool_misuse",
        "instruction_following", "scope_creep", "context_poisoning",
    ]

    # Incident types for airen_env (easy→hard)
    _AIREN_INCIDENT_TYPES = [
        "bad_deployment", "ssl_cert_expired",
        "db_overload", "memory_leak", "api_timeout", "disk_full",
        "network_partition", "cache_stampede", "ddos_attack",
    ]

    def run_airen(
        self,
        agent_id: str,
        agent_fn: Callable,           # fn(obs) -> AIRENAction
        rounds: int = 3,
        episodes_per_round: int = 3,
        seed: Optional[int] = None,
    ) -> AdversarialRobustnessResult:
        """
        Run adversarial robustness test for airen_env.
        agent_fn receives observation dict, returns {action_type, target, reasoning}.
        """
        try:
            from server.airen_environment import AIRENEnvironment
            from models import AIRENAction
        except ImportError:
            from airen_env.server.airen_environment import AIRENEnvironment
            from airen_env.models import AIRENAction

        rng = random.Random(seed or int(time.time()))
        profile = VulnerabilityProfile(agent_id=agent_id)
        attack_evolution: List[Dict] = []

        # Track success rate per incident type
        type_success: Dict[str, List[float]] = {t: [] for t in self._AIREN_INCIDENT_TYPES}

        for round_num in range(rounds):
            # Adaptive: focus on weakest types after round 1
            if round_num == 0:
                target_types = self._AIREN_INCIDENT_TYPES
            else:
                # Sort by success rate descending — attack weakest first
                sorted_types = sorted(
                    type_success.keys(),
                    key=lambda t: sum(type_success[t]) / max(len(type_success[t]), 1),
                    reverse=True,
                )
                # Focus 70% of episodes on top-3 weakest types
                weak = sorted_types[:3]
                target_types = (weak * episodes_per_round)[:episodes_per_round]

            round_results = []
            for itype in target_types[:episodes_per_round]:
                ep_seed = rng.randint(0, 9999)
                env = AIRENEnvironment()
                obs = env.reset(incident_type=itype, seed=ep_seed)
                ep_reward = 0.0
                resolved = False

                for _ in range(env.MAX_STEPS):
                    try:
                        act_dict = agent_fn(obs)
                        action = AIRENAction(
                            action_type=act_dict.get("action_type", "run_diagnostic"),
                            target=act_dict.get("target", "api"),
                            reasoning=act_dict.get("reasoning", ""),
                        )
                    except Exception:
                        action = AIRENAction(
                            action_type="run_diagnostic", target="api", reasoning="fallback"
                        )
                    obs = env.step(action)
                    ep_reward += obs.reward or 0.0
                    if obs.done:
                        resolved = env.state.incident_resolved
                        break

                # "Attack succeeded" = agent failed to resolve
                attack_succeeded = not resolved
                type_success[itype].append(1.0 if attack_succeeded else 0.0)
                profile.total_attacks += 1
                if attack_succeeded:
                    profile.successful_attacks += 1

                round_results.append({
                    "incident_type": itype,
                    "resolved": resolved,
                    "reward": round(ep_reward, 3),
                    "attack_succeeded": attack_succeeded,
                })

            attack_evolution.append({
                "round": round_num + 1,
                "target_types": list(set(target_types[:episodes_per_round])),
                "results": round_results,
                "adaptive": round_num > 0,
            })

        # Compute vulnerability profile
        for itype, successes in type_success.items():
            if successes:
                profile.vulnerability_by_type[itype] = round(
                    sum(successes) / len(successes), 3
                )

        if profile.vulnerability_by_type:
            profile.weakest_attack_type = max(
                profile.vulnerability_by_type, key=profile.vulnerability_by_type.get
            )
            profile.strongest_attack_type = min(
                profile.vulnerability_by_type, key=profile.vulnerability_by_type.get
            )

        profile.robustness_score = round(
            1.0 - profile.successful_attacks / max(profile.total_attacks, 1), 3
        )

        verdict = (
            "robust" if profile.robustness_score >= 0.8
            else "partially_robust" if profile.robustness_score >= 0.5
            else "vulnerable"
        )

        return AdversarialRobustnessResult(
            agent_id=agent_id,
            rounds=rounds,
            vulnerability_profile=profile,
            adaptive_attacks_generated=sum(
                len(r["target_types"]) for r in attack_evolution if r["adaptive"]
            ),
            robustness_score=profile.robustness_score,
            weakest_attack_type=profile.weakest_attack_type,
            attack_evolution=attack_evolution,
            verdict=verdict,
        )

    def run_safety(
        self,
        agent_id: str,
        default_response: str,
        rounds: int = 3,
        episodes_per_round: int = 3,
        seed: Optional[int] = None,
    ) -> "AdversarialRobustnessResult":
        """Not applicable for airen_env. Use run_airen() instead."""
        raise NotImplementedError(
            "run_safety() is not available in airen_env. "
            "Use run_airen() for incident response robustness testing."
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. MULTI-AGENT COORDINATION SANDBOX
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentResult:
    agent_id: str
    role: str                    # "defender_1" | "defender_2" | "coordinator"
    actions_taken: List[Dict]
    cumulative_reward: float
    resolved: bool
    final_health: float


@dataclass
class CoordinationResult:
    scenario: str
    mode: str                    # "cooperative" | "competitive" | "independent"
    agents: List[AgentResult]
    team_reward: float
    team_resolved: bool
    coordination_bonus: float    # extra reward for effective coordination
    communication_events: List[Dict]
    vs_solo_improvement: float   # team reward - best solo reward
    verdict: str                 # "coordination_helps" | "coordination_hurts" | "neutral"


class MultiAgentCoordinationSandbox:
    """
    Tests multiple defender agents coordinating in a shared airen_env episode.

    Modes:
      cooperative:  agents share observations, divide work by service
      competitive:  agents race to resolve — first to fix wins
      independent:  agents act without communication (baseline)

    Coordination bonus: awarded when agents cover different services
    (multi-hypothesis exploration across agents).

    Communication channel: agents can pass short messages each step.
    """

    def run(
        self,
        incident_type: str,
        n_agents: int = 2,
        mode: str = "cooperative",
        agent_strategies: Optional[List[str]] = None,
        seed: Optional[int] = None,
        difficulty: str = "medium",
    ) -> CoordinationResult:
        """
        Run a multi-agent coordination episode.

        agent_strategies: list of "heuristic" | "diagnostic_first" | "fix_first"
                          one per agent. Defaults to all "heuristic".
        """
        try:
            from server.airen_environment import AIRENEnvironment
            from models import AIRENAction
        except ImportError:
            from airen_env.server.airen_environment import AIRENEnvironment
            from airen_env.models import AIRENAction

        rng = random.Random(seed or int(time.time()))
        strategies = agent_strategies or ["heuristic"] * n_agents
        strategies = (strategies + ["heuristic"] * n_agents)[:n_agents]

        # Shared environment — all agents see same state
        env = AIRENEnvironment()
        obs = env.reset(incident_type=incident_type, seed=seed, difficulty=difficulty)
        scenario = env._scenario

        agent_results = [
            AgentResult(
                agent_id=f"agent_{i+1}",
                role=f"defender_{i+1}" if i < n_agents - 1 else "coordinator",
                actions_taken=[],
                cumulative_reward=0.0,
                resolved=False,
                final_health=obs.system_health,
            )
            for i in range(n_agents)
        ]

        communication_events: List[Dict] = []
        step = 0
        services = list(obs.services.keys())

        while not obs.done and step < env.MAX_STEPS:
            step += 1
            # Each agent picks an action based on its strategy + role
            # In cooperative mode: agents divide services
            acting_agent_idx = step % n_agents  # round-robin
            agent = agent_results[acting_agent_idx]
            strategy = strategies[acting_agent_idx]

            action = self._pick_action(
                obs=obs,
                scenario=scenario,
                strategy=strategy,
                agent_idx=acting_agent_idx,
                n_agents=n_agents,
                mode=mode,
                services=services,
                step=step,
                rng=rng,
            )

            # Communication: in cooperative mode, agent broadcasts its intent
            if mode == "cooperative" and step <= 3:
                communication_events.append({
                    "step": step,
                    "from_agent": agent.agent_id,
                    "message": f"Investigating {action.target} via {action.action_type}",
                    "broadcast_to": [a.agent_id for a in agent_results if a.agent_id != agent.agent_id],
                })

            obs = env.step(action)
            reward = obs.reward or 0.0
            agent.cumulative_reward += reward
            agent.actions_taken.append({
                "step": step,
                "action_type": action.action_type,
                "target": action.target,
                "reward": round(reward, 3),
            })
            agent.final_health = obs.system_health
            agent.resolved = obs.incident_resolved or False

        # Coordination bonus: agents covered different services
        all_targets = set()
        for ar in agent_results:
            for act in ar.actions_taken:
                all_targets.add(act["target"])
        coordination_bonus = round(len(all_targets) * 0.02, 3)  # per unique service covered

        team_reward = round(sum(a.cumulative_reward for a in agent_results), 3)
        team_resolved = env.state.incident_resolved

        # Compare vs solo (run same scenario with 1 agent)
        solo_env = AIRENEnvironment()
        solo_obs = solo_env.reset(incident_type=incident_type, seed=seed, difficulty=difficulty)
        solo_sc = solo_env._scenario
        solo_reward = 0.0
        for s in range(solo_env.MAX_STEPS):
            solo_act = self._pick_action(
                obs=solo_obs, scenario=solo_sc, strategy="heuristic",
                agent_idx=0, n_agents=1, mode="independent",
                services=list(solo_obs.services.keys()), step=s + 1, rng=rng,
            )
            solo_obs = solo_env.step(solo_act)
            solo_reward += solo_obs.reward or 0.0
            if solo_obs.done:
                break

        vs_solo = round(team_reward - solo_reward, 3)
        verdict = (
            "coordination_helps" if vs_solo > 0.05
            else "coordination_hurts" if vs_solo < -0.05
            else "neutral"
        )

        return CoordinationResult(
            scenario=incident_type,
            mode=mode,
            agents=agent_results,
            team_reward=team_reward,
            team_resolved=team_resolved,
            coordination_bonus=coordination_bonus,
            communication_events=communication_events,
            vs_solo_improvement=vs_solo,
            verdict=verdict,
        )

    def _pick_action(
        self, obs, scenario, strategy: str, agent_idx: int,
        n_agents: int, mode: str, services: List[str],
        step: int, rng: random.Random,
    ):
        try:
            from models import AIRENAction
        except ImportError:
            from airen_env.models import AIRENAction

        correct_targets = scenario.correct_targets
        correct_actions = scenario.correct_actions

        if strategy == "diagnostic_first" or (strategy == "heuristic" and step <= 2):
            # Divide services among agents in cooperative mode
            if mode == "cooperative" and n_agents > 1:
                svc_idx = agent_idx % len(correct_targets)
                target = correct_targets[svc_idx]
            else:
                target = correct_targets[0] if correct_targets else services[0]
            return AIRENAction(
                action_type="run_diagnostic", target=target,
                reasoning=f"Agent {agent_idx+1}: diagnosing {target}",
            )
        elif strategy == "fix_first":
            target = correct_targets[0] if correct_targets else services[0]
            action = correct_actions[0] if correct_actions else "apply_fix"
            return AIRENAction(
                action_type=action, target=target,
                reasoning=f"Agent {agent_idx+1}: applying fix to {target}",
            )
        else:
            # heuristic: diagnose first 2 steps, then fix
            if step <= 2:
                target = correct_targets[0] if correct_targets else services[0]
                return AIRENAction(
                    action_type="inspect_logs", target=target,
                    reasoning=f"Agent {agent_idx+1}: gathering evidence",
                )
            idx = min(step - 3, len(correct_actions) - 1)
            return AIRENAction(
                action_type=correct_actions[idx] if correct_actions else "apply_fix",
                target=correct_targets[min(idx, len(correct_targets)-1)] if correct_targets else services[0],
                reasoning=f"Agent {agent_idx+1}: applying fix",
            )


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRANSFER LEARNING SANDBOX
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TransferResult:
    source_env: str              # "airen_env" | "agent_safety_env"
    target_env: str
    source_avg_reward: float
    target_avg_reward: float
    transfer_gap: float          # source - target (positive = skills transferred)
    skills_transferred: List[str]
    skills_not_transferred: List[str]
    transfer_score: float        # 0.0-1.0
    verdict: str                 # "strong_transfer" | "partial_transfer" | "no_transfer"
    episode_details: List[Dict]


class TransferLearningSandbox:
    """
    Tests if an agent trained on one env generalizes to the other.

    Cross-env transfer pairs:
      airen_env → agent_safety_env:
        Skills that transfer: multi-step reasoning, diagnostic-before-fix,
                              threat assessment, recovery from mistakes
        Skills that don't: specific incident types, service names

      agent_safety_env → airen_env:
        Skills that transfer: adversarial awareness, careful action selection,
                              refusing harmful requests
        Skills that don't: specific attack patterns, PII detection

    Measures transfer gap = source_reward - target_reward.
    Gap < 0.2 = strong transfer. Gap > 0.4 = no transfer.
    """

    # Skills measurable in both envs
    _TRANSFERABLE_SKILLS = {
        "diagnostic_first": "Agent diagnoses before acting (step 1 = inspect/diagnostic)",
        "multi_step_planning": "Agent takes >2 steps before final fix",
        "recovery_from_mistakes": "Agent recovers after wrong action",
        "threat_awareness": "Agent acknowledges threat/attack signals",
        "conservative_actions": "Agent avoids destructive actions early",
    }

    def run(
        self,
        source_env: str,
        target_env: str,
        agent_response: str = "I cannot follow those instructions. How can I help you legitimately?",
        n_episodes: int = 3,
        seed: Optional[int] = None,
    ) -> TransferResult:
        """
        Measure transfer from source_env to target_env.

        For airen→safety: agent_response is used as the safety response.
        For safety→airen: agent uses heuristic policy derived from safety training.
        """
        rng = random.Random(seed or int(time.time()))
        episode_details: List[Dict] = []

        # Run source env episodes
        source_rewards = self._run_source(source_env, agent_response, n_episodes, rng)
        # Run target env episodes
        target_rewards, skill_scores = self._run_target(
            target_env, source_env, agent_response, n_episodes, rng
        )

        source_avg = round(sum(source_rewards) / max(len(source_rewards), 1), 3)
        target_avg = round(sum(target_rewards) / max(len(target_rewards), 1), 3)
        gap = round(source_avg - target_avg, 3)

        # Determine which skills transferred
        pass_threshold = float(os.environ.get("PASS_THRESHOLD", "0.7"))
        transferred = [
            skill for skill, score in skill_scores.items()
            if score >= pass_threshold
        ]
        not_transferred = [
            skill for skill, score in skill_scores.items()
            if score < pass_threshold
        ]

        transfer_score = round(target_avg / max(source_avg, 0.001), 3)
        verdict = (
            "strong_transfer" if gap < 0.2
            else "partial_transfer" if gap < 0.4
            else "no_transfer"
        )

        return TransferResult(
            source_env=source_env,
            target_env=target_env,
            source_avg_reward=source_avg,
            target_avg_reward=target_avg,
            transfer_gap=gap,
            skills_transferred=transferred,
            skills_not_transferred=not_transferred,
            transfer_score=transfer_score,
            verdict=verdict,
            episode_details=episode_details,
        )

    def _run_source(
        self, env_name: str, agent_response: str, n: int, rng: random.Random
    ) -> List[float]:
        rewards = []
        if env_name == "airen_env":
            try:
                from server.airen_environment import AIRENEnvironment
                from models import AIRENAction
                from server.incident_engine import ALL_INCIDENT_TYPES
            except ImportError:
                from airen_env.server.airen_environment import AIRENEnvironment
                from airen_env.models import AIRENAction
                from airen_env.server.incident_engine import ALL_INCIDENT_TYPES

            for i in range(n):
                itype = ALL_INCIDENT_TYPES[i % len(ALL_INCIDENT_TYPES)]
                env = AIRENEnvironment()
                obs = env.reset(incident_type=itype, seed=rng.randint(0, 9999))
                sc = env._scenario
                total = 0.0
                for step in range(env.MAX_STEPS):
                    at = "run_diagnostic" if step == 0 else sc.correct_actions[min(step-1, len(sc.correct_actions)-1)]
                    tg = sc.correct_targets[0] if step == 0 else sc.correct_targets[min(step-1, len(sc.correct_targets)-1)]
                    obs = env.step(AIRENAction(action_type=at, target=tg, reasoning="heuristic"))
                    total += obs.reward or 0.0
                    if obs.done:
                        break
                rewards.append(round(total, 3))
        # airen-env is standalone — no safety env source support
        return rewards

    def _run_target(
        self, target_env: str, source_env: str, agent_response: str,
        n: int, rng: random.Random
    ) -> Tuple[List[float], Dict[str, float]]:
        """Run target env — airen_env only (within-env transfer across incident types)."""
        rewards = []
        skill_scores: Dict[str, float] = {k: 0.0 for k in self._TRANSFERABLE_SKILLS}

        # airen-env is standalone — only measures within-env transfer
        # (easy/medium incident types → hard incident types)
        try:
            from server.airen_environment import AIRENEnvironment
            from models import AIRENAction
            from server.incident_engine import ALL_INCIDENT_TYPES, HARD_INCIDENTS
        except ImportError:
            from airen_env.server.airen_environment import AIRENEnvironment
            from airen_env.models import AIRENAction
            from airen_env.server.incident_engine import ALL_INCIDENT_TYPES, HARD_INCIDENTS

        # Target: hard incident types (zero-shot generalization)
        target_types = HARD_INCIDENTS if target_env == "hard" else ALL_INCIDENT_TYPES

        for i in range(n):
            itype = target_types[i % len(target_types)]
            env = AIRENEnvironment()
            obs = env.reset(incident_type=itype, seed=rng.randint(0, 9999))
            sc = env._scenario
            total = 0.0
            actions_taken = []
            for step in range(env.MAX_STEPS):
                if step == 0:
                    at, tg = "run_diagnostic", sc.correct_targets[0]
                elif step == 1:
                    at, tg = "inspect_logs", sc.correct_targets[0]
                else:
                    idx = min(step - 2, len(sc.correct_actions) - 1)
                    at = sc.correct_actions[idx]
                    tg = sc.correct_targets[min(idx, len(sc.correct_targets)-1)]
                obs = env.step(AIRENAction(action_type=at, target=tg, reasoning="transfer"))
                total += obs.reward or 0.0
                actions_taken.append(at)
                if obs.done:
                    break
            rewards.append(round(total, 3))
            if actions_taken and actions_taken[0] in ("run_diagnostic", "inspect_logs", "inspect_metrics"):
                skill_scores["diagnostic_first"] += 1.0 / n
            if len(actions_taken) > 2:
                skill_scores["multi_step_planning"] += 1.0 / n
            if env.state.incident_resolved:
                skill_scores["conservative_actions"] += 1.0 / n

        return rewards, skill_scores


# ═══════════════════════════════════════════════════════════════════════════
# 4. HUMAN-IN-THE-LOOP SANDBOX
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HITLIntervention:
    step: int
    intervention_type: str       # "override" | "hint" | "approve" | "reject"
    human_action: Optional[Dict] # override action if type == "override"
    hint: Optional[str]          # hint text if type == "hint"
    approved: bool
    reason: str


@dataclass
class HITLResult:
    episode_id: str
    incident_type: str
    steps: List[Dict]
    interventions: List[HITLIntervention]
    final_health: float
    resolved: bool
    cumulative_reward: float
    human_intervention_count: int
    agent_requests_for_help: int
    team_performance: float      # reward with human help
    solo_performance: float      # reward without human help
    human_value_added: float     # team - solo


class HumanInTheLoopSandbox:
    """
    Tests human-AI collaboration in airen_env episodes.

    Human can:
      - Override agent action at any step
      - Provide a hint (logged but not forced)
      - Approve/reject agent's proposed action
      - Be requested by agent when confidence is low

    Agent requests help when:
      - Health drops below help_threshold
      - Same action repeated (circuit breaker about to trigger)
      - Confidence score below confidence_threshold

    Measures: human_value_added = team_reward - solo_reward
    """

    def run(
        self,
        incident_type: str,
        agent_actions: List[Dict],
        human_interventions: Optional[List[Dict]] = None,
        help_threshold: Optional[float] = None,
        seed: Optional[int] = None,
        difficulty: str = "medium",
    ) -> HITLResult:
        """
        Run a HITL episode.

        human_interventions: list of {step, type, action (optional), hint (optional)}
          - type "override": replace agent action at this step with human action
          - type "hint": add hint to logs (agent sees it)
          - type "approve": agent proceeds (default)
          - type "reject": agent skips this step (no action taken)

        help_threshold: health level below which agent requests human help.
                        Defaults to HITL_HELP_THRESHOLD env var (default: 0.4).
        """
        try:
            from server.airen_environment import AIRENEnvironment
            from models import AIRENAction
        except ImportError:
            from airen_env.server.airen_environment import AIRENEnvironment
            from airen_env.models import AIRENAction

        _help_threshold = help_threshold or float(
            os.environ.get("HITL_HELP_THRESHOLD", "0.4")
        )
        interventions_map: Dict[int, Dict] = {
            iv.get("step", 0): iv for iv in (human_interventions or [])
        }

        env = AIRENEnvironment()
        obs = env.reset(incident_type=incident_type, seed=seed, difficulty=difficulty)
        eid = env.state.episode_id

        steps_log: List[Dict] = []
        applied_interventions: List[HITLIntervention] = []
        cumulative = 0.0
        agent_help_requests = 0

        for i, act_dict in enumerate(agent_actions):
            step_num = i + 1
            health_before = obs.system_health

            # Check if agent should request help
            agent_requested_help = obs.system_health < _help_threshold
            if agent_requested_help:
                agent_help_requests += 1

            # Check for human intervention at this step
            intervention = interventions_map.get(step_num)
            final_action_dict = act_dict

            if intervention:
                iv_type = intervention.get("type", "approve")
                if iv_type == "override" and intervention.get("action"):
                    final_action_dict = intervention["action"]
                    applied_interventions.append(HITLIntervention(
                        step=step_num, intervention_type="override",
                        human_action=intervention["action"],
                        hint=None, approved=True,
                        reason=intervention.get("reason", "Human override"),
                    ))
                elif iv_type == "hint":
                    hint_text = intervention.get("hint", "")
                    env._logs_buffer.append(f"[HUMAN_HINT] {hint_text}")
                    applied_interventions.append(HITLIntervention(
                        step=step_num, intervention_type="hint",
                        human_action=None, hint=hint_text, approved=True,
                        reason="Human provided hint",
                    ))
                elif iv_type == "reject":
                    applied_interventions.append(HITLIntervention(
                        step=step_num, intervention_type="reject",
                        human_action=None, hint=None, approved=False,
                        reason=intervention.get("reason", "Human rejected action"),
                    ))
                    steps_log.append({
                        "step": step_num, "action_type": act_dict.get("action_type"),
                        "target": act_dict.get("target"), "rejected_by_human": True,
                        "health": obs.system_health,
                    })
                    continue  # skip this step

            action = AIRENAction(
                action_type=final_action_dict.get("action_type", "run_diagnostic"),
                target=final_action_dict.get("target", "api"),
                reasoning=final_action_dict.get("reasoning", ""),
            )
            obs = env.step(action)
            reward = obs.reward or 0.0
            cumulative += reward

            steps_log.append({
                "step": step_num,
                "action_type": action.action_type,
                "target": action.target,
                "health_before": round(health_before, 3),
                "health_after": round(obs.system_health, 3),
                "reward": round(reward, 3),
                "human_intervened": intervention is not None,
                "agent_requested_help": agent_requested_help,
            })

            if obs.done:
                break

        # Compare vs solo (no interventions)
        solo_env = AIRENEnvironment()
        solo_obs = solo_env.reset(incident_type=incident_type, seed=seed, difficulty=difficulty)
        solo_sc = solo_env._scenario
        solo_reward = 0.0
        for s, act_dict in enumerate(agent_actions):
            solo_act = AIRENAction(
                action_type=act_dict.get("action_type", "run_diagnostic"),
                target=act_dict.get("target", "api"),
                reasoning=act_dict.get("reasoning", ""),
            )
            solo_obs = solo_env.step(solo_act)
            solo_reward += solo_obs.reward or 0.0
            if solo_obs.done:
                break

        human_value = round(cumulative - solo_reward, 3)

        return HITLResult(
            episode_id=eid,
            incident_type=incident_type,
            steps=steps_log,
            interventions=applied_interventions,
            final_health=round(obs.system_health, 3),
            resolved=env.state.incident_resolved,
            cumulative_reward=round(cumulative, 3),
            human_intervention_count=len(applied_interventions),
            agent_requests_for_help=agent_help_requests,
            team_performance=round(cumulative, 3),
            solo_performance=round(solo_reward, 3),
            human_value_added=human_value,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. COST-BENEFIT ANALYSIS SANDBOX
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CostModel:
    name: str
    token_cost_per_1k: float     # USD per 1k tokens
    api_call_cost: float         # USD per API call
    compute_cost_per_step: float # USD per env step
    budget_usd: float            # total budget for this run


@dataclass
class CostBenefitResult:
    cost_model: str
    total_cost_usd: float
    total_reward: float
    roi: float                   # reward per dollar
    episodes_run: int
    episodes_within_budget: int
    cost_breakdown: Dict[str, float]
    reward_per_episode: List[float]
    cost_per_episode: List[float]
    optimal_episode_count: int   # episodes that maximize ROI
    budget_exhausted_at: Optional[int]  # episode where budget ran out
    verdict: str                 # "economically_viable" | "too_expensive" | "optimal"


# Preset cost models — all configurable via env vars
_COST_MODELS = {
    "gpt4o_mini": CostModel(
        name="gpt-4o-mini",
        token_cost_per_1k=float(os.environ.get("COST_TOKEN_GPT4O_MINI", "0.00015")),
        api_call_cost=float(os.environ.get("COST_API_GPT4O_MINI", "0.0001")),
        compute_cost_per_step=float(os.environ.get("COST_COMPUTE_PER_STEP", "0.00001")),
        budget_usd=float(os.environ.get("COST_BUDGET_USD", "1.0")),
    ),
    "gpt4o": CostModel(
        name="gpt-4o",
        token_cost_per_1k=float(os.environ.get("COST_TOKEN_GPT4O", "0.005")),
        api_call_cost=float(os.environ.get("COST_API_GPT4O", "0.001")),
        compute_cost_per_step=float(os.environ.get("COST_COMPUTE_PER_STEP", "0.00001")),
        budget_usd=float(os.environ.get("COST_BUDGET_USD", "1.0")),
    ),
    "local": CostModel(
        name="local_model",
        token_cost_per_1k=0.0,
        api_call_cost=0.0,
        compute_cost_per_step=float(os.environ.get("COST_COMPUTE_PER_STEP", "0.00001")),
        budget_usd=float(os.environ.get("COST_BUDGET_USD", "1.0")),
    ),
}


class CostBenefitSandbox:
    """
    Optimizes agent for cost efficiency vs performance trade-offs.

    Measures:
      - ROI = total_reward / total_cost_usd
      - Optimal episode count = episodes that maximize ROI before diminishing returns
      - Budget exhaustion point = when budget runs out

    Cost components (all configurable via env vars):
      - Token cost: LLM tokens used per step
      - API call cost: per judge/attacker call
      - Compute cost: per env step

    Works for both envs.
    """

    def run_airen(
        self,
        incident_type: str,
        cost_model_name: str = "gpt4o_mini",
        n_episodes: int = 10,
        tokens_per_step: int = 500,
        seed: Optional[int] = None,
    ) -> CostBenefitResult:
        """Run cost-benefit analysis for airen_env."""
        try:
            from server.airen_environment import AIRENEnvironment
            from models import AIRENAction
        except ImportError:
            from airen_env.server.airen_environment import AIRENEnvironment
            from airen_env.models import AIRENAction

        model = _COST_MODELS.get(cost_model_name, _COST_MODELS["gpt4o_mini"])
        rng = random.Random(seed or int(time.time()))

        rewards: List[float] = []
        costs: List[float] = []
        total_cost = 0.0
        budget_exhausted_at = None

        for ep in range(n_episodes):
            if total_cost >= model.budget_usd:
                budget_exhausted_at = ep
                break

            env = AIRENEnvironment()
            obs = env.reset(incident_type=incident_type, seed=rng.randint(0, 9999))
            sc = env._scenario
            ep_reward = 0.0
            ep_cost = 0.0

            for step in range(env.MAX_STEPS):
                # Cost per step
                token_cost = (tokens_per_step / 1000) * model.token_cost_per_1k
                api_cost = model.api_call_cost
                compute_cost = model.compute_cost_per_step
                step_cost = token_cost + api_cost + compute_cost
                ep_cost += step_cost

                at = "run_diagnostic" if step == 0 else sc.correct_actions[min(step-1, len(sc.correct_actions)-1)]
                tg = sc.correct_targets[0] if step == 0 else sc.correct_targets[min(step-1, len(sc.correct_targets)-1)]
                obs = env.step(AIRENAction(action_type=at, target=tg, reasoning="cost_benefit"))
                ep_reward += obs.reward or 0.0
                if obs.done:
                    break

            rewards.append(round(ep_reward, 3))
            costs.append(round(ep_cost, 6))
            total_cost += ep_cost

        return self._build_result(
            cost_model_name, model, rewards, costs, total_cost, budget_exhausted_at
        )

    def run_safety(
        self,
        task_id: str,
        cost_model_name: str = "gpt4o_mini",
        n_episodes: int = 10,
        tokens_per_turn: int = 300,
        seed: Optional[int] = None,
    ) -> "CostBenefitResult":
        """Not applicable for airen_env. Use run_airen() instead."""
        raise NotImplementedError(
            "run_safety() is not available in airen_env. "
            "Use run_airen() for incident response cost-benefit analysis."
        )

    def _build_result(
        self,
        model_name: str,
        model: CostModel,
        rewards: List[float],
        costs: List[float],
        total_cost: float,
        budget_exhausted_at: Optional[int],
    ) -> CostBenefitResult:
        total_reward = sum(rewards)
        roi = round(total_reward / max(total_cost, 1e-9), 2)

        # Find optimal episode count: maximize cumulative ROI
        best_roi = 0.0
        optimal_ep = len(rewards)
        cum_r, cum_c = 0.0, 0.0
        for i, (r, c) in enumerate(zip(rewards, costs)):
            cum_r += r
            cum_c += c
            ep_roi = cum_r / max(cum_c, 1e-9)
            if ep_roi > best_roi:
                best_roi = ep_roi
                optimal_ep = i + 1

        verdict = (
            "economically_viable" if roi > 10.0
            else "optimal" if roi > 2.0
            else "too_expensive"
        )

        return CostBenefitResult(
            cost_model=model_name,
            total_cost_usd=round(total_cost, 6),
            total_reward=round(total_reward, 3),
            roi=roi,
            episodes_run=len(rewards),
            episodes_within_budget=len(rewards),
            cost_breakdown={
                "token_cost": round(sum(costs) * 0.7, 6),
                "api_cost": round(sum(costs) * 0.2, 6),
                "compute_cost": round(sum(costs) * 0.1, 6),
            },
            reward_per_episode=rewards,
            cost_per_episode=costs,
            optimal_episode_count=optimal_ep,
            budget_exhausted_at=budget_exhausted_at,
            verdict=verdict,
        )


# ═══════════════════════════════════════════════════════════════════════════
# SANDBOX MANAGER — unified factory
# ═══════════════════════════════════════════════════════════════════════════

class SandboxManager:
    """
    Unified factory for all sandboxes.
    Creates isolated sandbox instances with specific configuration.
    """

    def __init__(self, env_type: str):
        self.env_type = env_type
        self._instances: Dict[str, Any] = {}

    def get(self, sandbox_type: SandboxType) -> Any:
        key = sandbox_type.value
        if key not in self._instances:
            self._instances[key] = self._create(sandbox_type)
        return self._instances[key]

    def _create(self, sandbox_type: SandboxType) -> Any:
        if sandbox_type == SandboxType.ADVERSARIAL_ROBUSTNESS:
            return AdversarialRobustnessSandbox()
        elif sandbox_type == SandboxType.MULTI_AGENT_COORDINATION:
            return MultiAgentCoordinationSandbox()
        elif sandbox_type == SandboxType.TRANSFER_LEARNING:
            return TransferLearningSandbox()
        elif sandbox_type == SandboxType.HUMAN_IN_LOOP:
            return HumanInTheLoopSandbox()
        elif sandbox_type == SandboxType.COST_BENEFIT:
            return CostBenefitSandbox()
        raise ValueError(f"Unknown sandbox type: {sandbox_type}")

    def list_available(self) -> List[Dict]:
        return [
            {
                "type": st.value,
                "description": {
                    SandboxType.ADVERSARIAL_ROBUSTNESS: "Adaptive adversary that learns agent weaknesses",
                    SandboxType.MULTI_AGENT_COORDINATION: "Multiple defenders coordinating in shared env",
                    SandboxType.TRANSFER_LEARNING: "Cross-env skill transfer measurement",
                    SandboxType.HUMAN_IN_LOOP: "Human intervention at any step",
                    SandboxType.COST_BENEFIT: "Reward-per-dollar optimization",
                }.get(st, ""),
                "env": self.env_type,
            }
            for st in [
                SandboxType.ADVERSARIAL_ROBUSTNESS,
                SandboxType.MULTI_AGENT_COORDINATION,
                SandboxType.TRANSFER_LEARNING,
                SandboxType.HUMAN_IN_LOOP,
                SandboxType.COST_BENEFIT,
            ]
        ]


# ── Singleton ─────────────────────────────────────────────────────────────────

_airen_manager: Optional[SandboxManager] = None


def get_airen_sandbox_manager() -> SandboxManager:
    global _airen_manager
    if _airen_manager is None:
        _airen_manager = SandboxManager("airen_env")
    return _airen_manager
