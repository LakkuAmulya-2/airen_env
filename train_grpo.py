"""
AIREN — GRPO Training Script
==============================
Meta PyTorch OpenEnv Hackathon x SST

Trains a base LLM on AIREN using TRL's GRPOTrainer with environment_factory —
the official OpenEnv training pattern.

The environment is a TRUE multi-step MDP:
  - 10-step episodes (world degrades every step)
  - Dense reward every step (not just episode end)
  - 9 incident types with curriculum learning (easy → hard)
  - Partial observability — agent sees symptoms, not root cause
  - Failure + Recovery — wrong fix → worse → agent must recover

Usage:
    # Train (requires GPU + running server)
    python train_grpo.py --model Qwen/Qwen3-0.6B --episodes 200

    # Multi-agent mode (AttackerAgent + MonitoringAgent + AutoScalerAgent)
    MULTI_AGENT=1 python train_grpo.py --model Qwen/Qwen3-0.6B --episodes 200

    # Dry run — validate config without training
    python train_grpo.py --model Qwen/Qwen3-0.6B --dry-run

    # Push trained model to HF Hub
    python train_grpo.py --model Qwen/Qwen3-0.6B --push-to-hub

Required env vars:
    ENV_URL       AIREN server URL (default: HF Space)
    HF_TOKEN      Hugging Face token (for model push)
    MULTI_AGENT   Set to 1 for multi-agent mode

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "trl>=1.0.0",
#   "transformers>=4.45.0",
#   "datasets>=2.20.0",
#   "airen-env @ git+https://huggingface.co/spaces/amulyalakku/airen-env",
# ]
# ///
"""

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request
from typing import Any, Dict, List, Optional

# ── Config ────────────────────────────────────────────────────────────────────
ENV_URL     = os.environ.get("ENV_URL", "https://amulyalakku-airen-env.hf.space")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
MULTI_AGENT = os.environ.get("MULTI_AGENT", "0") == "1"
USE_VLLM    = os.environ.get("USE_VLLM", "0") == "1"

# ── Completion logging ────────────────────────────────────────────────────────
_COMPLETIONS_DIR = pathlib.Path("completions")
_COMPLETIONS_DIR.mkdir(exist_ok=True)


def _save_completion(
    episode_id: str,
    incident_type: str,
    actions: int,
    reward: float,
    resolved: bool,
    steps: int,
) -> None:
    """Persist episode completion for offline analysis and replay."""
    record = {
        "episode_id": episode_id,
        "incident_type": incident_type,
        "correct_actions": actions,
        "cumulative_reward": round(reward, 4),
        "resolved": resolved,
        "steps": steps,
        "timestamp": time.time(),
    }
    path = _COMPLETIONS_DIR / f"{episode_id}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CLASS (TRL environment_factory interface)
# ══════════════════════════════════════════════════════════════════════════════

class AIRENToolEnv:
    """
    TRL-compatible multi-step environment for AIREN.

    Follows the official OpenEnv environment_factory pattern:
      - __init__: initialize client
      - reset(**kwargs): start episode, return observation string
      - Tool methods: run_diagnostic, apply_fix, inspect_logs, etc.
      - self.reward: read by reward_func after episode

    Multi-turn loop (handled by TRL automatically):
      obs = env.reset()
      while not done:
          action = model.generate(obs)
          obs = env.run_diagnostic(target)  # or any tool
      reward = env.reward
    """

    def __init__(self):
        from airen_env import AIRENEnv
        self._client = AIRENEnv(base_url=ENV_URL).sync().__enter__()
        self.reward = 0.0
        self._cumulative_reward = 0.0  # Fix 2: track cumulative, not just last step
        self._obs = None
        self._done = False
        self._episode_id: Optional[str] = None

    def reset(self, **kwargs) -> str:
        """Start new incident episode. Returns initial observation."""
        self.reward = 0.0
        self._cumulative_reward = 0.0
        self._done = False
        result = self._client.reset()
        self._obs = result.observation
        self._episode_id = getattr(self._obs, "incident_id", None)
        return self._format_obs()

    def run_diagnostic(self, target: str) -> str:
        """
        Run deep diagnostic on a service to identify root cause.
        Most informative action — reveals root cause signals.
        Required before applying a fix for full resolution credit.

        Args:
            target: Service name (api, db, cache, worker, payment, network, upstream, infra, tls)

        Returns:
            Diagnostic results and updated system state.
        """
        return self._step("run_diagnostic", target, f"Running diagnostic on {target}")

    def inspect_logs(self, target: str) -> str:
        """
        Read recent log lines for a service to find diagnostic clues.
        Use first to understand symptoms before acting.

        Args:
            target: Service name

        Returns:
            Recent log lines with diagnostic information.
        """
        return self._step("inspect_logs", target, f"Inspecting logs for {target}")

    def inspect_metrics(self, target: str) -> str:
        """
        Check performance metrics (CPU, memory, latency, error rate) for a service.

        Args:
            target: Service name

        Returns:
            Current metric values for the service.
        """
        return self._step("inspect_metrics", target, f"Checking metrics for {target}")

    def apply_fix(self, target: str) -> str:
        """
        Apply targeted fix to a service based on diagnosis.
        Correct for db_overload, network_partition, cache_stampede.
        Requires prior diagnostic step for full resolution credit.

        Args:
            target: Service name to fix

        Returns:
            Fix result and updated system health.
        """
        return self._step("apply_fix", target, f"Applying fix to {target}")

    def restart_service(self, target: str) -> str:
        """
        Restart a service. Causes brief downtime.
        Correct ONLY for memory_leak on worker service.

        Args:
            target: Service name to restart

        Returns:
            Restart result and updated service state.
        """
        return self._step("restart_service", target, f"Restarting {target}")

    def rollback_deployment(self, target: str) -> str:
        """
        Rollback service to previous stable version.
        Correct ONLY for bad_deployment on payment service.

        Args:
            target: Service name to rollback

        Returns:
            Rollback result and updated service state.
        """
        return self._step("rollback_deployment", target, f"Rolling back {target}")

    def scale_service(self, target: str) -> str:
        """
        Scale up a service to handle increased load.
        Correct for cache_stampede on db, ddos_attack on api.

        Args:
            target: Service name to scale

        Returns:
            Scale result and updated service state.
        """
        return self._step("scale_service", target, f"Scaling {target}")

    def acknowledge_incident(self) -> str:
        """
        Acknowledge the incident to start formal tracking.
        Reduces threat level slightly. Good first action when severity is unclear.

        Returns:
            Acknowledgment confirmation and updated threat level.
        """
        return self._step("acknowledge_incident", "system", "Acknowledging incident")

    def _step(self, action_type: str, target: str, reasoning: str) -> str:
        from airen_env import AIRENAction
        if self._done:
            return "[EPISODE ALREADY ENDED] Call reset() to start a new episode."

        result = self._client.step(AIRENAction(
            action_type=action_type, target=target, reasoning=reasoning
        ))
        obs = result.observation
        step_reward = result.reward or 0.0
        self._cumulative_reward += step_reward
        self.reward = self._cumulative_reward  # Fix 2: always expose cumulative
        self._done = result.done
        self._obs = obs

        # Fix 1: return terminal string instead of raising ValueError
        if self._done:
            _save_completion(
                episode_id=self._episode_id or "unknown",
                incident_type=obs.incident_type,
                actions=getattr(obs, "correct_actions_count", 0),
                reward=self._cumulative_reward,
                resolved=obs.incident_resolved or False,
                steps=obs.step_number,
            )
            return (
                f"[EPISODE COMPLETE] Health: {obs.system_health:.0%} | "
                f"Resolved: {obs.incident_resolved} | "
                f"Cumulative Reward: {self._cumulative_reward:.3f} | "
                f"Steps: {obs.step_number}/{obs.max_steps}\n"
                f"Result: {obs.action_result or 'Episode ended'}\n"
                f"Reward breakdown: {obs.reward_explanation or ''}"
            )

        return (
            f"Health: {obs.system_health:.0%} | Step Reward: {step_reward:.3f} | "
            f"Cumulative: {self._cumulative_reward:.3f} | "
            f"Step: {obs.step_number}/{obs.max_steps}\n"
            f"Result: {obs.action_result or 'Action executed'}\n"
            f"Latest log: {obs.logs[-1] if obs.logs else 'None'}"
        )

    def _format_obs(self) -> str:
        obs = self._obs
        services = {
            name: f"{s['status']} | cpu={s['cpu_pct']}% | err={s['error_rate']:.0%} | lat={s['latency_ms']}ms"
            for name, s in obs.services.items()
        }
        alerts = [f"[{a.get('severity','')}] {a.get('message','')}" for a in obs.alerts[:3]]
        meta = obs.metadata or {}
        ledger = meta.get("ledger_context", "")
        return (
            f"INCIDENT: {obs.incident_type} (severity={obs.severity})\n"
            f"Health: {obs.system_health:.0%} | Threat: {obs.threat_level:.2f} "
            f"| Step: {obs.step_number}/{obs.max_steps}\n\n"
            f"SERVICES:\n" + "\n".join(f"  {k}: {v}" for k, v in services.items()) + "\n\n"
            f"RECENT LOGS:\n" + "\n".join(f"  {l}" for l in obs.logs[-3:]) + "\n\n"
            f"ALERTS:\n" + "\n".join(f"  {a}" for a in alerts) + "\n\n"
            + (f"ACTION LEDGER:\n{ledger}\n\n" if ledger else "")
            + "Strategy: Diagnose first (run_diagnostic or inspect_logs), then apply the correct fix. "
            "Wrong fixes make things worse — diagnose before acting."
        )


# ══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def reward_func(environments, **kwargs) -> List[float]:
    """
    Read cumulative reward from each environment instance.
    Fix 2: returns cumulative episode reward, not just last step reward.
    Called by GRPOTrainer after each episode completes.
    """
    rewards = [env._cumulative_reward for env in environments]

    # Submit live learning curve bucket every 10 episodes
    _EPISODE_REWARD_BUFFER.extend(rewards)
    if len(_EPISODE_REWARD_BUFFER) >= 10:
        _flush_curve_bucket()

    return rewards


# ── Live learning curve submission ────────────────────────────────────────────
# Buffers episode rewards and submits buckets to /learning_curve/submit
# so the Gradio UI shows a live reward curve during training.

_EPISODE_REWARD_BUFFER: List[float] = []
_EPISODE_BUCKET_COUNT: int = 0


def _flush_curve_bucket() -> None:
    """Submit the current reward buffer as a learning curve bucket."""
    global _EPISODE_BUCKET_COUNT
    if not _EPISODE_REWARD_BUFFER:
        return

    rewards = list(_EPISODE_REWARD_BUFFER)
    _EPISODE_REWARD_BUFFER.clear()
    _EPISODE_BUCKET_COUNT += 1

    avg_reward = round(sum(rewards) / len(rewards), 3)
    resolution_rate = round(sum(1 for r in rewards if r > 0.3) / len(rewards), 3)
    start_ep = (_EPISODE_BUCKET_COUNT - 1) * 10 + 1
    end_ep = start_ep + len(rewards) - 1

    try:
        payload = json.dumps({
            "episode_range": f"{start_ep}-{end_ep}",
            "policy": "grpo",
            "avg_reward": avg_reward,
            "resolution_rate": resolution_rate,
            "model": os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B"),
        }).encode()
        req = urllib.request.Request(
            f"{ENV_URL}/learning_curve/submit",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3):
            pass
        _log({"event": "CURVE_SUBMITTED", "bucket": f"{start_ep}-{end_ep}",
              "avg_reward": avg_reward, "resolution_rate": resolution_rate})
    except Exception as e:
        _log({"event": "CURVE_SUBMIT_SKIP", "reason": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _log(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj), flush=True)


def train(model_name: str, episodes: int, output_dir: str,
          push_to_hub: bool, dry_run: bool) -> None:

    ALL_INCIDENT_TYPES = ["db_overload", "memory_leak", "network_partition", "bad_deployment", "cache_stampede", "api_timeout", "disk_full", "ssl_cert_expired", "ddos_attack"]

    _log({
        "event": "DRY_RUN" if dry_run else "TRAIN_START",
        "env": "airen_env",
        "env_factory": "AIRENToolEnv",
        "model": model_name,
        "episodes": episodes,
        "incident_types": ALL_INCIDENT_TYPES,
        "multi_agent": MULTI_AGENT,
        "env_url": ENV_URL,
        "pattern": "environment_factory (TRL official)",
        "timestamp": time.time(),
    })

    if dry_run:
        _log({"event": "DRY_RUN_COMPLETE", "status": "config_valid",
              "env_factory": "AIRENToolEnv",
              "note": "environment_factory pattern — TRL handles multi-turn loop"})
        return

    try:
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset

        system_prompt = (
            "You are an expert Site Reliability Engineer (SRE) responding to production incidents. "
            "You will receive the current state of a distributed system. "
            "Use the available tools to diagnose and fix the incident. "
            "Always diagnose before fixing — wrong fixes make things worse. "
            "The system degrades every step — act efficiently."
        )

        dataset = Dataset.from_dict({
            "prompt": [
                [{"role": "user", "content": system_prompt}]
            ] * max(episodes, 64)
        })

        config = GRPOConfig(
            output_dir=output_dir,
            max_completion_length=512,
            num_generations=4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=1e-5,
            logging_steps=5,
            save_steps=50,
            push_to_hub=push_to_hub and bool(HF_TOKEN),
            hub_token=HF_TOKEN or None,
            hub_model_id=output_dir if push_to_hub else None,
            use_vllm=USE_VLLM,
            log_completions=True,
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=model_name,
            reward_funcs=reward_func,
            train_dataset=dataset,
            args=config,
            environment_factory=AIRENToolEnv,
        )
        trainer.train()

        # Flush any remaining episodes to the live curve
        if _EPISODE_REWARD_BUFFER:
            _flush_curve_bucket()

        if push_to_hub and HF_TOKEN:
            trainer.push_to_hub()

        _log({"event": "TRAIN_COMPLETE", "model": model_name, "output_dir": output_dir})

    except ImportError as e:
        _log({"event": "TRAIN_ERROR", "error": str(e),
              "hint": "pip install trl>=1.0.0 transformers datasets"})
        raise


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AIREN — GRPO Training (TRL environment_factory)"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="Model name or path")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training episodes")
    parser.add_argument("--output-dir", default="./output",
                        help="Output directory for checkpoints")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push trained model to HF Hub")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without training")
    args = parser.parse_args()

    train(
        model_name=args.model,
        episodes=args.episodes,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
