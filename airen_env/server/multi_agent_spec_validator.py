# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Multi-Agent Specification Validator — Upgrade #4

JSON schema contracts for unambiguous agent coordination.
Solves the NeurIPS 2025 MAST taxonomy finding: 79% of multi-agent
failures stem from specification ambiguity.

Real pain point: When multiple agents share resources without clear
ownership contracts, they conflict — both try to restart the same
service, creating a split-brain scenario.

Integrated into AIREN's multi-agent mode to validate agent specs
before episode start and detect coordination failures in real-time.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class SpecValidationError:
    field: str
    error: str
    severity: str           # CRITICAL | HIGH | MEDIUM
    suggestion: str


@dataclass
class CoordinationFailure:
    failure_type: str       # resource_conflict | role_ambiguity | missing_protocol
    agents_involved: List[str]
    resource: str
    description: str
    resolution: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentSpec:
    name: str
    role: str
    tools: List[str]
    owned_resources: List[str]
    communication_protocol: Dict[str, Any]
    constraints: Dict[str, Any]


class MultiAgentSpecValidator:
    """
    Validate multi-agent specifications for MAST taxonomy compliance.

    Enforces:
    1. Unambiguous role descriptions (< 50 words)
    2. Explicit tool lists (no wildcards)
    3. Resource ownership — each resource owned by exactly one agent
    4. Communication protocol defined
    5. Constraint boundaries specified

    Detects coordination failures in real-time during episodes.
    """

    # MAST taxonomy failure categories
    MAST_CATEGORIES = {
        "specification_ambiguity": "Role or task description is ambiguous",
        "resource_conflict": "Multiple agents own the same resource",
        "missing_protocol": "No communication protocol defined",
        "tool_overlap": "Multiple agents have the same destructive tool",
        "constraint_missing": "No resource or action constraints defined",
    }

    def __init__(self) -> None:
        self._validation_history: List[Dict[str, Any]] = []
        self._coordination_failures: List[CoordinationFailure] = []
        self._specs_validated: int = 0

    def validate_agent_spec(
        self,
        spec: Dict[str, Any],
    ) -> List[SpecValidationError]:
        """
        Validate a single agent specification.
        Returns list of validation errors (empty = valid).
        """
        errors: List[SpecValidationError] = []

        # Required fields
        required = {
            "name": "string",
            "role": "string",
            "tools": "list",
            "owned_resources": "list",
            "communication_protocol": "dict",
            "constraints": "dict",
        }

        for field_name, expected_type in required.items():
            if field_name not in spec:
                errors.append(SpecValidationError(
                    field=field_name,
                    error=f"Missing required field: '{field_name}'",
                    severity="CRITICAL",
                    suggestion=f"Add '{field_name}' to agent spec",
                ))
                continue

            val = spec[field_name]
            type_ok = (
                (expected_type == "string" and isinstance(val, str)) or
                (expected_type == "list" and isinstance(val, list)) or
                (expected_type == "dict" and isinstance(val, dict))
            )
            if not type_ok:
                errors.append(SpecValidationError(
                    field=field_name,
                    error=f"Invalid type for '{field_name}': expected {expected_type}, got {type(val).__name__}",
                    severity="HIGH",
                    suggestion=f"Change '{field_name}' to {expected_type} type",
                ))

        # Role description must be concise (< 50 words)
        if "role" in spec and isinstance(spec["role"], str):
            word_count = len(spec["role"].split())
            if word_count > 50:
                errors.append(SpecValidationError(
                    field="role",
                    error=f"Role description too long ({word_count} words > 50 max). Ambiguous roles cause coordination failures.",
                    severity="HIGH",
                    suggestion="Condense role to a single clear sentence (< 50 words)",
                ))
            if len(spec["role"].strip()) < 5:
                errors.append(SpecValidationError(
                    field="role",
                    error="Role description too vague (< 5 chars)",
                    severity="CRITICAL",
                    suggestion="Provide a clear, specific role description",
                ))

        # Tools must be explicit (no wildcards)
        if "tools" in spec and isinstance(spec["tools"], list):
            if "*" in spec["tools"] or "all" in spec["tools"]:
                errors.append(SpecValidationError(
                    field="tools",
                    error="Wildcard tools ('*' or 'all') create ambiguity",
                    severity="HIGH",
                    suggestion="List specific tool names explicitly",
                ))
            if len(spec["tools"]) == 0:
                errors.append(SpecValidationError(
                    field="tools",
                    error="Agent has no tools — cannot take any actions",
                    severity="MEDIUM",
                    suggestion="Add at least one tool to the agent's tool list",
                ))

        # Communication protocol must specify escalation path
        if "communication_protocol" in spec and isinstance(spec["communication_protocol"], dict):
            proto = spec["communication_protocol"]
            if "escalation_path" not in proto and "escalate_to" not in proto:
                errors.append(SpecValidationError(
                    field="communication_protocol.escalation_path",
                    error="No escalation path defined — agent cannot hand off to human",
                    severity="MEDIUM",
                    suggestion="Add 'escalation_path' or 'escalate_to' to communication_protocol",
                ))

        self._specs_validated += 1
        return errors

    def validate_multi_agent_system(
        self,
        agent_specs: List[Dict[str, Any]],
    ) -> Tuple[List[SpecValidationError], List[CoordinationFailure]]:
        """
        Validate the entire multi-agent system for coordination issues.
        Checks cross-agent conflicts that single-agent validation misses.
        """
        all_errors: List[SpecValidationError] = []
        failures: List[CoordinationFailure] = []

        # Validate each agent individually
        for spec in agent_specs:
            errors = self.validate_agent_spec(spec)
            all_errors.extend(errors)

        # Check resource ownership conflicts
        resource_owners: Dict[str, List[str]] = {}
        for spec in agent_specs:
            agent_name = spec.get("name", "unknown")
            for resource in spec.get("owned_resources", []):
                if resource not in resource_owners:
                    resource_owners[resource] = []
                resource_owners[resource].append(agent_name)

        for resource, owners in resource_owners.items():
            if len(owners) > 1:
                failure = CoordinationFailure(
                    failure_type="resource_conflict",
                    agents_involved=owners,
                    resource=resource,
                    description=(
                        f"Resource '{resource}' claimed by {len(owners)} agents: "
                        f"{', '.join(owners)}. This causes split-brain conflicts."
                    ),
                    resolution=(
                        f"Assign '{resource}' to exactly one agent. "
                        f"Others should request access via communication_protocol."
                    ),
                )
                failures.append(failure)
                all_errors.append(SpecValidationError(
                    field=f"owned_resources[{resource}]",
                    error=f"Resource conflict: '{resource}' owned by {owners}",
                    severity="CRITICAL",
                    suggestion=f"Assign '{resource}' to exactly one agent",
                ))

        # Check for destructive tool overlap
        destructive_tools = {"restart_service", "rollback_deployment", "scale_service", "apply_fix"}
        tool_agents: Dict[str, List[str]] = {}
        for spec in agent_specs:
            agent_name = spec.get("name", "unknown")
            for tool in spec.get("tools", []):
                if tool in destructive_tools:
                    if tool not in tool_agents:
                        tool_agents[tool] = []
                    tool_agents[tool].append(agent_name)

        for tool, agents in tool_agents.items():
            if len(agents) > 1:
                failures.append(CoordinationFailure(
                    failure_type="tool_overlap",
                    agents_involved=agents,
                    resource=tool,
                    description=(
                        f"Destructive tool '{tool}' available to {len(agents)} agents: "
                        f"{', '.join(agents)}. Simultaneous use causes conflicts."
                    ),
                    resolution=(
                        f"Assign '{tool}' to one primary agent. "
                        f"Others should request via communication_protocol."
                    ),
                ))

        self._coordination_failures.extend(failures)
        self._validation_history.append({
            "timestamp": time.time(),
            "agents": len(agent_specs),
            "errors": len(all_errors),
            "failures": len(failures),
        })

        return all_errors, failures

    def detect_runtime_conflict(
        self,
        agent_name: str,
        action_type: str,
        target: str,
        active_agents: Dict[str, Dict[str, Any]],
    ) -> Optional[CoordinationFailure]:
        """
        Detect coordination conflicts during episode execution.
        Called before each agent action to check for simultaneous conflicts.
        """
        destructive = {"restart_service", "rollback_deployment", "scale_service", "apply_fix"}
        if action_type not in destructive:
            return None

        # Check if another agent is also acting on the same target
        for other_name, other_state in active_agents.items():
            if other_name == agent_name:
                continue
            if (other_state.get("last_action") == action_type and
                    other_state.get("last_target") == target and
                    time.time() - other_state.get("last_action_time", 0) < 5.0):
                return CoordinationFailure(
                    failure_type="resource_conflict",
                    agents_involved=[agent_name, other_name],
                    resource=target,
                    description=(
                        f"Simultaneous {action_type} on '{target}' by "
                        f"'{agent_name}' and '{other_name}'"
                    ),
                    resolution=(
                        f"Only one agent should {action_type} '{target}'. "
                        f"Use communication_protocol to coordinate."
                    ),
                )
        return None

    def generate_coordination_scenario(
        self,
        rng_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Generate a multi-agent coordination failure scenario.
        Based on PwC finding: structured orchestration gives 7x accuracy gain.
        """
        import random
        rng = random.Random(rng_seed)

        services = ["api", "db", "cache", "worker", "payment"]
        conflict_service = rng.choice(services)

        return {
            "scenario_type": "coordination_failure",
            "description": (
                "Two agents attempt to restart the same service simultaneously, "
                "causing a split-brain conflict and extended downtime."
            ),
            "agents": [
                {
                    "name": "Monitor",
                    "role": "Track system health and detect anomalies",
                    "tools": ["inspect_metrics", "inspect_logs", "acknowledge_incident"],
                    "owned_resources": ["metrics_db"],
                    "communication_protocol": {
                        "escalate_to": "Responder",
                        "channel": "incident_queue",
                    },
                    "constraints": {"max_actions_per_step": 1},
                },
                {
                    "name": "Responder",
                    "role": "Fix detected incidents using remediation tools",
                    "tools": ["restart_service", "scale_service", "apply_fix", "rollback_deployment"],
                    "owned_resources": [conflict_service],
                    "communication_protocol": {
                        "receives_from": "Monitor",
                        "escalate_to": "human_oncall",
                    },
                    "constraints": {"requires_monitor_signal": True},
                },
            ],
            "failure_mode": f"Both agents try to restart '{conflict_service}' simultaneously",
            "correct_behavior": (
                f"Monitor detects anomaly → escalates to Responder → "
                f"Responder acts on '{conflict_service}'"
            ),
            "wrong_behavior": "Both act simultaneously → conflict → extended downtime",
            "mast_category": "specification_ambiguity",
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "specs_validated": self._specs_validated,
            "coordination_failures_detected": len(self._coordination_failures),
            "recent_failures": [
                {
                    "type": f.failure_type,
                    "agents": f.agents_involved,
                    "resource": f.resource,
                }
                for f in self._coordination_failures[-5:]
            ],
            "mast_categories": list(self.MAST_CATEGORIES.keys()),
        }


# Module-level singleton
_validator: Optional[MultiAgentSpecValidator] = None


def get_spec_validator() -> MultiAgentSpecValidator:
    global _validator
    if _validator is None:
        _validator = MultiAgentSpecValidator()
    return _validator
