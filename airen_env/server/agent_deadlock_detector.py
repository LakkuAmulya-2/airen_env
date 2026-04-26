# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Agentic Deadlock Detector — AIREN's Deadlock Prevention Layer

Based on:
- "When AI Agents Wait for Each Other Forever" (tianpan.co, April 2026)
- DPBench benchmark: 95-100% deadlock rate with 3 agents, 25-65% with 5 agents
- ICLR 2026: sequential execution bottleneck in multi-agent workflows

Real production finding: Multi-agent systems deadlock at 25-95% under
normal operating conditions. Three failure modes:
  1. Mirror Mirror Loop: two agents bounce work back and forth forever
  2. Politeness Spiral: agents defer to each other endlessly (livelock)
  3. Resource Starvation: agent waits for resource held by another agent

Root cause: LLMs trained on similar data arrive at identical "rational"
strategies independently → circular wait when executing simultaneously.

Key insight: Natural language is a POOR synchronization primitive.
Distributed systems solved this with mutexes/semaphores decades ago.
AIREN implements structured coordination protocols to prevent deadlock.

Meta pain point: Multi-agent coordination failures cost $400+/incident
in wasted compute and require manual intervention to resolve.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class DeadlockEvent:
    deadlock_type: str          # mirror_loop | politeness_spiral | resource_starvation
    agents_involved: List[str]
    resource: str
    description: str
    resolution: str
    detected_at: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_steps: int = 0


@dataclass
class AgentActionRecord:
    agent_name: str
    action_type: str
    target: str
    timestamp: float
    response_to: Optional[str] = None  # which agent's action this responds to


class AgentDeadlockDetector:
    """
    Detects and resolves agentic deadlock patterns in real-time.

    Three detection algorithms:
    1. Mirror Loop: same action pair repeating between two agents
    2. Politeness Spiral: agents deferring to each other N+ times
    3. Resource Starvation: agent waiting for resource > timeout

    Resolution strategies:
    - Mirror Loop → inject tie-breaker (agent with lower ID acts first)
    - Politeness Spiral → force one agent to act (random selection)
    - Resource Starvation → preempt and reassign resource

    Integrated into AIREN's multi-agent mode.
    """

    MIRROR_LOOP_THRESHOLD = 3       # same pair repeating N times = deadlock
    POLITENESS_THRESHOLD = 4        # N consecutive deferrals = livelock
    STARVATION_TIMEOUT_STEPS = 5    # steps waiting = starvation

    def __init__(self) -> None:
        self._action_history: deque = deque(maxlen=50)
        self._resource_locks: Dict[str, str] = {}      # resource → agent holding it
        self._waiting_agents: Dict[str, Dict] = {}     # agent → {resource, since_step}
        self._deadlock_events: List[DeadlockEvent] = []
        self._total_deadlocks: int = 0
        self._total_resolved: int = 0
        self._step_count: int = 0

    def reset(self) -> None:
        self._action_history.clear()
        self._resource_locks.clear()
        self._waiting_agents.clear()
        self._step_count = 0

    def record_agent_action(
        self,
        agent_name: str,
        action_type: str,
        target: str,
        response_to: Optional[str] = None,
    ) -> None:
        """Record an agent action for deadlock analysis."""
        self._step_count += 1
        self._action_history.append(AgentActionRecord(
            agent_name=agent_name,
            action_type=action_type,
            target=target,
            timestamp=time.time(),
            response_to=response_to,
        ))

    def acquire_resource(
        self,
        agent_name: str,
        resource: str,
        step: int,
    ) -> Tuple[bool, Optional[str]]:
        """
        Try to acquire a resource for an agent.
        Returns (acquired, blocking_agent).
        Implements mutex-style locking to prevent simultaneous access.
        """
        if resource not in self._resource_locks:
            self._resource_locks[resource] = agent_name
            # Clear any waiting state
            self._waiting_agents.pop(agent_name, None)
            return True, None
        else:
            holder = self._resource_locks[resource]
            if holder == agent_name:
                return True, None  # already holds it
            # Resource held by another agent — wait
            self._waiting_agents[agent_name] = {
                "resource": resource,
                "since_step": step,
                "blocking_agent": holder,
            }
            return False, holder

    def release_resource(self, agent_name: str, resource: str) -> None:
        """Release a resource held by an agent."""
        if self._resource_locks.get(resource) == agent_name:
            del self._resource_locks[resource]
            # Wake up waiting agents
            for waiting_agent, info in list(self._waiting_agents.items()):
                if info["resource"] == resource:
                    del self._waiting_agents[waiting_agent]

    def detect_deadlock(
        self,
        current_step: int,
        active_agents: List[str],
    ) -> Optional[DeadlockEvent]:
        """
        Scan for deadlock patterns in recent action history.
        Returns DeadlockEvent if deadlock detected, None otherwise.
        """
        history = list(self._action_history)

        # Pattern 1: Mirror Mirror Loop
        mirror = self._detect_mirror_loop(history)
        if mirror:
            self._total_deadlocks += 1
            self._deadlock_events.append(mirror)
            return mirror

        # Pattern 2: Politeness Spiral (livelock)
        spiral = self._detect_politeness_spiral(history)
        if spiral:
            self._total_deadlocks += 1
            self._deadlock_events.append(spiral)
            return spiral

        # Pattern 3: Resource Starvation
        starvation = self._detect_resource_starvation(current_step)
        if starvation:
            self._total_deadlocks += 1
            self._deadlock_events.append(starvation)
            return starvation

        # Pattern 4: Circular Wait (A waits for B, B waits for A)
        circular = self._detect_circular_wait()
        if circular:
            self._total_deadlocks += 1
            self._deadlock_events.append(circular)
            return circular

        return None

    def resolve_deadlock(
        self,
        event: DeadlockEvent,
        agents: List[str],
    ) -> Dict[str, Any]:
        """
        Apply resolution strategy for detected deadlock.
        Returns resolution action to inject into the environment.
        """
        event.resolved = True
        self._total_resolved += 1

        if event.deadlock_type == "mirror_loop":
            # Tie-breaker: agent with lower alphabetical name acts first
            sorted_agents = sorted(event.agents_involved)
            primary = sorted_agents[0]
            return {
                "resolution_type": "tie_breaker",
                "primary_agent": primary,
                "action": f"Force {primary} to act first — other agents wait",
                "inject_message": (
                    f"DEADLOCK RESOLVED: {primary} has priority. "
                    f"Other agents must wait for {primary} to complete."
                ),
            }

        elif event.deadlock_type == "politeness_spiral":
            # Force random agent to act
            import random
            actor = random.choice(event.agents_involved)
            return {
                "resolution_type": "force_action",
                "forced_agent": actor,
                "action": f"Force {actor} to act immediately — stop deferring",
                "inject_message": (
                    f"LIVELOCK RESOLVED: {actor} must act now. "
                    f"Stop deferring — take the next logical action."
                ),
            }

        elif event.deadlock_type in ("resource_starvation", "circular_wait"):
            # Preempt: release all locks and reassign
            for resource in list(self._resource_locks.keys()):
                if self._resource_locks[resource] in event.agents_involved:
                    del self._resource_locks[resource]
            self._waiting_agents.clear()
            return {
                "resolution_type": "preempt",
                "resource": event.resource,
                "action": f"Preempt resource '{event.resource}' — reassign to waiting agent",
                "inject_message": (
                    f"DEADLOCK RESOLVED: Resource '{event.resource}' preempted. "
                    f"All agents may now proceed."
                ),
            }

        return {
            "resolution_type": "reset",
            "action": "Reset agent coordination state",
            "inject_message": "DEADLOCK RESOLVED: Coordination state reset.",
        }

    def _detect_mirror_loop(
        self, history: List[AgentActionRecord]
    ) -> Optional[DeadlockEvent]:
        """Detect: two agents bouncing the same action back and forth."""
        if len(history) < self.MIRROR_LOOP_THRESHOLD * 2:
            return None

        # Look for alternating pattern between two agents
        recent = history[-self.MIRROR_LOOP_THRESHOLD * 2:]
        agents_in_recent = list({r.agent_name for r in recent})

        if len(agents_in_recent) != 2:
            return None

        a1, a2 = agents_in_recent[0], agents_in_recent[1]
        a1_actions = [r for r in recent if r.agent_name == a1]
        a2_actions = [r for r in recent if r.agent_name == a2]

        if not a1_actions or not a2_actions:
            return None

        # Check if they're doing the same action on the same target
        a1_last = a1_actions[-1]
        a2_last = a2_actions[-1]

        if (a1_last.action_type == a2_last.action_type and
                a1_last.target == a2_last.target):
            # Check if this pattern has repeated
            pattern_count = sum(
                1 for r in recent
                if r.action_type == a1_last.action_type and r.target == a1_last.target
            )
            if pattern_count >= self.MIRROR_LOOP_THRESHOLD * 2:
                return DeadlockEvent(
                    deadlock_type="mirror_loop",
                    agents_involved=[a1, a2],
                    resource=a1_last.target,
                    description=(
                        f"Mirror loop: {a1} and {a2} both doing "
                        f"{a1_last.action_type}({a1_last.target}) repeatedly"
                    ),
                    resolution="Apply tie-breaker: lower-ID agent acts first",
                )
        return None

    def _detect_politeness_spiral(
        self, history: List[AgentActionRecord]
    ) -> Optional[DeadlockEvent]:
        """Detect: agents deferring to each other endlessly (livelock)."""
        if len(history) < self.POLITENESS_THRESHOLD:
            return None

        recent = history[-self.POLITENESS_THRESHOLD:]
        deferral_actions = {"acknowledge_incident", "inspect_metrics"}

        # All recent actions are deferrals
        if all(r.action_type in deferral_actions for r in recent):
            agents = list({r.agent_name for r in recent})
            if len(agents) >= 2:
                return DeadlockEvent(
                    deadlock_type="politeness_spiral",
                    agents_involved=agents,
                    resource="coordination",
                    description=(
                        f"Politeness spiral: {agents} all deferring for "
                        f"{self.POLITENESS_THRESHOLD}+ steps"
                    ),
                    resolution="Force one agent to act immediately",
                )
        return None

    def _detect_resource_starvation(
        self, current_step: int
    ) -> Optional[DeadlockEvent]:
        """Detect: agent waiting too long for a resource."""
        for agent, info in self._waiting_agents.items():
            wait_steps = current_step - info["since_step"]
            if wait_steps >= self.STARVATION_TIMEOUT_STEPS:
                return DeadlockEvent(
                    deadlock_type="resource_starvation",
                    agents_involved=[agent, info["blocking_agent"]],
                    resource=info["resource"],
                    description=(
                        f"{agent} waiting {wait_steps} steps for "
                        f"'{info['resource']}' held by {info['blocking_agent']}"
                    ),
                    resolution=f"Preempt '{info['resource']}' from {info['blocking_agent']}",
                )
        return None

    def _detect_circular_wait(self) -> Optional[DeadlockEvent]:
        """Detect: A waits for B, B waits for A (classic deadlock)."""
        # Build wait graph
        wait_graph: Dict[str, str] = {}
        for agent, info in self._waiting_agents.items():
            wait_graph[agent] = info["blocking_agent"]

        # Detect cycle in wait graph
        for start_agent in wait_graph:
            visited: Set[str] = set()
            current = start_agent
            while current in wait_graph:
                if current in visited:
                    # Cycle detected
                    cycle_agents = list(visited)
                    resource = self._waiting_agents.get(start_agent, {}).get("resource", "unknown")
                    return DeadlockEvent(
                        deadlock_type="circular_wait",
                        agents_involved=cycle_agents,
                        resource=resource,
                        description=f"Circular wait detected: {' → '.join(cycle_agents + [cycle_agents[0]])}",
                        resolution="Preempt all resources in the cycle",
                    )
                visited.add(current)
                current = wait_graph[current]

        return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_deadlocks_detected": self._total_deadlocks,
            "total_resolved": self._total_resolved,
            "active_resource_locks": len(self._resource_locks),
            "waiting_agents": len(self._waiting_agents),
            "recent_deadlocks": [
                {
                    "type": e.deadlock_type,
                    "agents": e.agents_involved,
                    "resource": e.resource,
                    "resolved": e.resolved,
                }
                for e in self._deadlock_events[-5:]
            ],
            "deadlock_types": {
                "mirror_loop": "Two agents bouncing same action back and forth",
                "politeness_spiral": "Agents deferring to each other endlessly (livelock)",
                "resource_starvation": "Agent waiting too long for held resource",
                "circular_wait": "A waits for B, B waits for A",
            },
            "reference": "DPBench 2026: 95-100% deadlock rate with 3 agents in simultaneous mode",
        }


# Module-level singleton
_detector: Optional[AgentDeadlockDetector] = None


def get_deadlock_detector() -> AgentDeadlockDetector:
    global _detector
    if _detector is None:
        _detector = AgentDeadlockDetector()
    return _detector
