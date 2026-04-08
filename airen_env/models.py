# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
AIREN — AI Production Incident Response & Recovery Environment
Data models: Action, Observation, State
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State

ActionType = Literal[
    "inspect_logs",
    "inspect_metrics",
    "restart_service",
    "scale_service",
    "rollback_deployment",
    "run_diagnostic",
    "ignore_alert",
    "acknowledge_incident",
    "apply_fix",
]


class AIRENAction(Action):
    """Action taken by the agent to respond to a production incident."""

    action_type: ActionType = Field(..., description="Type of action to take")
    target: str = Field(default="", description="Target service/component (e.g. 'api', 'db', 'cache')")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action parameters (e.g. {'replicas': 3} for scale_service)",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning for this action (used for diagnosis scoring)",
    )


class AIRENObservation(Observation):
    """
    Observation of the production system state.
    Includes partial observability — agent sees symptoms, not root cause.
    """

    # Incident info
    incident_id: str = Field(description="Unique incident identifier")
    incident_type: str = Field(description="Type of incident (e.g. 'db_overload')")
    severity: str = Field(description="critical | high | medium | low")
    step_number: int = Field(default=0, description="Current step in episode")
    max_steps: int = Field(default=10, description="Max steps before timeout")

    # System state (partial — agent sees symptoms not root cause)
    services: Dict[str, Dict[str, Any]] = Field(
        description="Service states: {name: {status, latency_ms, error_rate, cpu_pct, memory_pct}}"
    )
    metrics: Dict[str, float] = Field(
        description="System metrics: {metric_name: value}"
    )
    logs: List[str] = Field(
        description="Recent log lines (may contain noise/red herrings)"
    )
    alerts: List[Dict[str, Any]] = Field(
        description="Active alerts: [{service, severity, message, triggered_at}]"
    )
    system_health: float = Field(
        description="Overall system health 0.0-1.0 (1.0 = fully healthy)"
    )

    # Action feedback (populated after step)
    action_result: Optional[str] = Field(default=None)
    action_success: Optional[bool] = Field(default=None)

    # Reward breakdown (populated after step)
    recovery_score: Optional[float] = Field(default=None)
    diagnosis_score: Optional[float] = Field(default=None)
    efficiency_score: Optional[float] = Field(default=None)
    cost_penalty: Optional[float] = Field(default=None)
    downtime_penalty: Optional[float] = Field(default=None)

    # Episode summary (populated on done=True)
    incident_resolved: Optional[bool] = Field(default=None)
    cumulative_reward: Optional[float] = Field(default=None)
    correct_actions_count: Optional[int] = Field(default=None)
    wrong_actions_count: Optional[int] = Field(default=None)
    reward_explanation: Optional[str] = Field(default=None)
    judge_used: Optional[str] = Field(default=None)
    diagnosis_quality: Optional[str] = Field(default=None)
    judge_reasoning: Optional[str] = Field(default=None)
    # RL-grade reward components (populated after every step)
    threat_mitigation: Optional[float] = Field(default=None)
    hallucination_penalty: Optional[float] = Field(default=None)
    security_violation_penalty: Optional[float] = Field(default=None)
    resolve_bonus: Optional[float] = Field(default=None)
    exploration_bonus: Optional[float] = Field(default=None, description="Bonus for multi-hypothesis testing")
    recovery_bonus: Optional[float] = Field(default=None, description="Bonus for recovering after wrong fix")
    # Current threat/attack state (partial observability — agent sees risk, not root cause)
    threat_level: Optional[float] = Field(default=None, description="Current threat level 0.0-1.0")
    attack_progress: Optional[float] = Field(default=None, description="Incident progression 0.0-1.0")
    # Metadata dict for test compatibility and extensibility
    metadata: Optional[Any] = Field(default=None)


class AIRENState(State):
    """Full episode state tracking incident response progress."""

    episode_id: Optional[str] = Field(default=None)
    incident_id: Optional[str] = Field(default=None)
    incident_type: Optional[str] = Field(default=None)
    root_cause: Optional[str] = Field(default=None, description="True root cause (hidden from agent)")
    step_count: int = Field(default=0)
    steps_taken: int = Field(default=0)
    actions_taken: List[str] = Field(default_factory=list)
    correct_actions: int = Field(default=0)
    wrong_actions: int = Field(default=0)
    system_health_history: List[float] = Field(default_factory=list)
    incident_resolved: bool = Field(default=False)
    resolution_step: Optional[int] = Field(default=None)
    total_downtime_steps: int = Field(default=0)
    cumulative_reward: float = Field(default=0.0)
    # RL-grade state tracking
    threat_level: float = Field(default=0.5, description="Current threat level 0.0-1.0")
    attack_progress: float = Field(default=0.0, description="How far the incident has progressed")
    # Failure + recovery tracking (exposed via /state endpoint)
    wrong_fixes_applied: int = Field(default=0, description="Number of wrong fixes applied this episode")
    recovery_attempts: int = Field(default=0, description="Number of recovery attempts after wrong fix")
    hypotheses_tested: int = Field(default=0, description="Number of unique services investigated")
