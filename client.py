# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""AIREN Environment Client."""

from typing import Any, Dict, List, Optional
from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import AIRENAction, AIRENObservation, AIRENState
except ImportError:
    from models import AIRENAction, AIRENObservation, AIRENState


class AIRENEnv(EnvClient[AIRENAction, AIRENObservation, AIRENState]):
    """
    Client for AIREN — AI Production Incident Response Environment.

    Example (sync):
        with AIRENEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(incident_type="db_overload")
            obs = result.observation
            print(obs.incident_type, obs.system_health)
            print(obs.logs)

            # Agent inspects then fixes
            result = env.step(AIRENAction(
                action_type="run_diagnostic", target="db",
                reasoning="DB CPU at 95%, slow queries in logs"
            ))
            result = env.step(AIRENAction(
                action_type="apply_fix", target="db",
                reasoning="Unindexed query causing full table scan"
            ))
            print(result.observation.system_health)   # should be > 0.8
            print(result.observation.reward)           # multi-objective reward
    """

    def _step_payload(self, action: AIRENAction) -> Dict[str, Any]:
        return {
            "action_type": action.action_type,
            "target": action.target,
            "parameters": action.parameters,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AIRENObservation]:
        obs_data = payload.get("observation", {})
        obs = AIRENObservation(
            incident_id=obs_data.get("incident_id", ""),
            incident_type=obs_data.get("incident_type", ""),
            severity=obs_data.get("severity", "high"),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 10),
            services=obs_data.get("services", {}),
            metrics=obs_data.get("metrics", {}),
            logs=obs_data.get("logs", []),
            alerts=obs_data.get("alerts", []),
            system_health=obs_data.get("system_health", 0.0),
            action_result=obs_data.get("action_result"),
            action_success=obs_data.get("action_success"),
            recovery_score=obs_data.get("recovery_score"),
            diagnosis_score=obs_data.get("diagnosis_score"),
            efficiency_score=obs_data.get("efficiency_score"),
            cost_penalty=obs_data.get("cost_penalty"),
            downtime_penalty=obs_data.get("downtime_penalty"),
            incident_resolved=obs_data.get("incident_resolved"),
            cumulative_reward=obs_data.get("cumulative_reward"),
            correct_actions_count=obs_data.get("correct_actions_count"),
            wrong_actions_count=obs_data.get("wrong_actions_count"),
            reward_explanation=obs_data.get("reward_explanation"),
            judge_used=obs_data.get("judge_used"),
            diagnosis_quality=obs_data.get("diagnosis_quality"),
            judge_reasoning=obs_data.get("judge_reasoning"),
            # RL-grade reward components
            threat_level=obs_data.get("threat_level"),
            attack_progress=obs_data.get("attack_progress"),
            threat_mitigation=obs_data.get("threat_mitigation"),
            hallucination_penalty=obs_data.get("hallucination_penalty"),
            security_violation_penalty=obs_data.get("security_violation_penalty"),
            resolve_bonus=obs_data.get("resolve_bonus"),
            metadata=obs_data.get("metadata"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AIRENState:
        return AIRENState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            incident_id=payload.get("incident_id"),
            incident_type=payload.get("incident_type"),
            root_cause=payload.get("root_cause"),
            steps_taken=payload.get("steps_taken", 0),
            actions_taken=payload.get("actions_taken", []),
            correct_actions=payload.get("correct_actions", 0),
            wrong_actions=payload.get("wrong_actions", 0),
            system_health_history=payload.get("system_health_history", []),
            incident_resolved=payload.get("incident_resolved", False),
            total_downtime_steps=payload.get("total_downtime_steps", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            threat_level=payload.get("threat_level", 0.5),
            attack_progress=payload.get("attack_progress", 0.0),
        )
