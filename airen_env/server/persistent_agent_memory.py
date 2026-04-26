# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Persistent Agent Memory — Cross-Session State Persistence

Based on:
- "AI Agent Memory 2026: Why Agents Forget" (alexcloudstar.com, April 2026)
- "Continuum Memory Architectures for Long-Horizon LLM Agents" (arxiv 2601.09913)
- "The 90-Day Rebuild Problem" (cybernative.ai, 2026)

Real production finding: Only 5% of enterprises run AI agents in production.
The #1 reason: agents forget everything between sessions.
The memory market hit $6.27B in 2026 — this is the biggest unsolved problem.

AIREN's Persistent Memory solves three layers:
  1. Episode Memory: what happened in this episode (hot path)
  2. Cross-Episode Memory: patterns learned across episodes (warm path)
  3. Long-Term Skill Memory: generalizable skills (cold path, persisted to disk)

Memory architecture:
  - Working memory: current episode context (deque, 20 items)
  - Episodic memory: recent episode summaries (last 50 episodes)
  - Semantic memory: incident-type → best strategy mapping
  - Procedural memory: action sequences that worked (skill library)

This is what separates a toy RL env from a production system:
agents that LEARN and REMEMBER across sessions, not just within one.

Meta pain point: Amazon's 90-day code safety reset happened because
agents had no memory of past failures — they repeated the same mistakes.
"""

import json
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EpisodeMemory:
    """Summary of a completed episode — stored in episodic memory."""
    episode_id: str
    incident_type: str
    seed: int
    resolved: bool
    steps_taken: int
    cumulative_reward: float
    final_health: float
    key_actions: List[str]          # most impactful actions
    root_cause: str
    lesson: str                     # one-sentence lesson learned
    timestamp: float = field(default_factory=time.time)


@dataclass
class SkillRecord:
    """A learned skill: incident_type → optimal action sequence."""
    incident_type: str
    action_sequence: List[Dict[str, str]]   # [{action_type, target, reasoning}]
    success_rate: float
    avg_reward: float
    times_used: int
    last_updated: float = field(default_factory=time.time)


@dataclass
class MemoryRetrievalResult:
    relevant_episodes: List[EpisodeMemory]
    best_skill: Optional[SkillRecord]
    context_hint: str               # injected into agent's context
    memory_tokens_saved: int        # vs re-learning from scratch


class PersistentAgentMemory:
    """
    Cross-session persistent memory for AIREN agents.

    Three memory tiers:
    1. Working Memory (hot): current episode, last 20 actions
    2. Episodic Memory (warm): last 50 episode summaries
    3. Semantic/Procedural Memory (cold): skills + strategies, persisted to disk

    Memory retrieval: given incident_type, returns relevant past episodes
    and the best known strategy — injected into agent context.

    This reduces the "cold start" problem: new episodes benefit from
    everything the agent learned in previous episodes.
    """

    EPISODIC_MEMORY_SIZE = 50
    WORKING_MEMORY_SIZE = 20
    SKILL_MIN_EPISODES = 3          # min episodes before skill is trusted

    def __init__(self, storage_dir: Optional[str] = None) -> None:
        self._storage_dir = Path(storage_dir or "agent_memory")
        self._storage_dir.mkdir(exist_ok=True)

        # Working memory (current episode)
        self._working_memory: deque = deque(maxlen=self.WORKING_MEMORY_SIZE)

        # Episodic memory (recent episodes)
        self._episodic_memory: deque = deque(maxlen=self.EPISODIC_MEMORY_SIZE)

        # Semantic memory: incident_type → list of episode summaries
        self._semantic_index: Dict[str, List[str]] = defaultdict(list)  # type → episode_ids

        # Procedural memory: incident_type → best skill
        self._skill_library: Dict[str, SkillRecord] = {}

        # Stats
        self._total_retrievals: int = 0
        self._total_tokens_saved: int = 0
        self._session_start = time.time()

        # Load persisted memory
        self._load_from_disk()

    def record_step(
        self,
        action_type: str,
        target: str,
        reasoning: str,
        reward: float,
        health: float,
    ) -> None:
        """Add step to working memory."""
        self._working_memory.append({
            "action_type": action_type,
            "target": target,
            "reasoning": reasoning[:100],
            "reward": round(reward, 3),
            "health": round(health, 3),
            "timestamp": time.time(),
        })

    def consolidate_episode(
        self,
        episode_id: str,
        incident_type: str,
        seed: int,
        resolved: bool,
        steps_taken: int,
        cumulative_reward: float,
        final_health: float,
        root_cause: str,
        correct_actions: List[str],
        correct_targets: List[str],
    ) -> EpisodeMemory:
        """
        Consolidate working memory into episodic memory at episode end.
        Extract lesson and update skill library.
        """
        # Extract key actions (highest reward steps)
        working = list(self._working_memory)
        key_actions = sorted(working, key=lambda x: x["reward"], reverse=True)[:3]
        key_action_strs = [f"{a['action_type']}({a['target']})" for a in key_actions]

        # Generate lesson
        lesson = self._extract_lesson(
            incident_type, resolved, steps_taken,
            correct_actions, correct_targets, cumulative_reward
        )

        # Create episode memory
        mem = EpisodeMemory(
            episode_id=episode_id,
            incident_type=incident_type,
            seed=seed,
            resolved=resolved,
            steps_taken=steps_taken,
            cumulative_reward=round(cumulative_reward, 3),
            final_health=round(final_health, 3),
            key_actions=key_action_strs,
            root_cause=root_cause,
            lesson=lesson,
        )

        # Store in episodic memory
        self._episodic_memory.append(mem)
        self._semantic_index[incident_type].append(episode_id)

        # Update skill library if episode was successful
        if resolved and cumulative_reward > 0.5:
            self._update_skill(incident_type, working, cumulative_reward)

        # Persist to disk
        self._persist_episode(mem)

        # Clear working memory for next episode
        self._working_memory.clear()

        return mem

    def retrieve_for_incident(
        self,
        incident_type: str,
        current_health: float,
    ) -> MemoryRetrievalResult:
        """
        Retrieve relevant memories for a new incident.
        Returns context hint to inject into agent's prompt.
        """
        self._total_retrievals += 1

        # Find relevant past episodes
        relevant_ids = set(self._semantic_index.get(incident_type, []))
        relevant_episodes = [
            m for m in self._episodic_memory
            if m.episode_id in relevant_ids
        ][-5:]  # last 5 relevant episodes

        # Get best skill
        best_skill = self._skill_library.get(incident_type)

        # Build context hint
        context_hint = self._build_context_hint(
            incident_type, relevant_episodes, best_skill
        )

        # Estimate tokens saved (vs agent re-learning from scratch)
        tokens_saved = len(relevant_episodes) * 200 + (500 if best_skill else 0)
        self._total_tokens_saved += tokens_saved

        return MemoryRetrievalResult(
            relevant_episodes=relevant_episodes,
            best_skill=best_skill,
            context_hint=context_hint,
            memory_tokens_saved=tokens_saved,
        )

    def get_working_memory_context(self) -> str:
        """Get current episode context as a string for agent injection."""
        working = list(self._working_memory)
        if not working:
            return ""
        lines = []
        for step in working[-5:]:  # last 5 steps
            lines.append(
                f"Step: {step['action_type']}({step['target']}) "
                f"→ reward={step['reward']:.3f} health={step['health']:.2f}"
            )
        return "Recent actions:\n" + "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        skill_summary = {
            itype: {
                "success_rate": round(s.success_rate, 3),
                "avg_reward": round(s.avg_reward, 3),
                "times_used": s.times_used,
            }
            for itype, s in self._skill_library.items()
        }
        return {
            "working_memory_size": len(self._working_memory),
            "episodic_memory_size": len(self._episodic_memory),
            "skill_library_size": len(self._skill_library),
            "total_retrievals": self._total_retrievals,
            "total_tokens_saved": self._total_tokens_saved,
            "incident_types_in_memory": list(self._semantic_index.keys()),
            "skill_library": skill_summary,
            "session_uptime_seconds": round(time.time() - self._session_start, 1),
            "meta_pain_point": "90-day rebuild problem — agents forget everything between sessions",
            "solution": "3-tier persistent memory: working → episodic → procedural",
        }

    def _extract_lesson(
        self,
        incident_type: str,
        resolved: bool,
        steps_taken: int,
        correct_actions: List[str],
        correct_targets: List[str],
        reward: float,
    ) -> str:
        """Extract a one-sentence lesson from the episode."""
        correct_action = correct_actions[0] if correct_actions else "apply_fix"
        correct_target = correct_targets[0] if correct_targets else "the root cause service"

        if resolved and steps_taken <= 3:
            return (
                f"For {incident_type}: {correct_action}({correct_target}) in ≤3 steps "
                f"achieves max reward ({reward:.2f}). Fast diagnosis wins."
            )
        elif resolved:
            return (
                f"For {incident_type}: resolved in {steps_taken} steps with "
                f"{correct_action}({correct_target}). Diagnose first, then fix."
            )
        else:
            return (
                f"For {incident_type}: FAILED. Should have used "
                f"{correct_action}({correct_target}). Avoid wrong-service actions."
            )

    def _update_skill(
        self,
        incident_type: str,
        working_memory: List[Dict],
        reward: float,
    ) -> None:
        """Update skill library with successful episode."""
        # Extract action sequence
        action_seq = [
            {
                "action_type": step["action_type"],
                "target": step["target"],
                "reasoning": step["reasoning"][:50],
            }
            for step in working_memory
        ]

        existing = self._skill_library.get(incident_type)
        if existing is None:
            self._skill_library[incident_type] = SkillRecord(
                incident_type=incident_type,
                action_sequence=action_seq,
                success_rate=1.0,
                avg_reward=reward,
                times_used=1,
            )
        else:
            # Update with exponential moving average
            alpha = 0.3
            existing.avg_reward = alpha * reward + (1 - alpha) * existing.avg_reward
            existing.success_rate = min(1.0, existing.success_rate * 0.9 + 0.1)
            existing.times_used += 1
            # Update action sequence if this episode was better
            if reward > existing.avg_reward:
                existing.action_sequence = action_seq
            existing.last_updated = time.time()

        # Persist skill library
        self._persist_skills()

    def _build_context_hint(
        self,
        incident_type: str,
        episodes: List[EpisodeMemory],
        skill: Optional[SkillRecord],
    ) -> str:
        """Build context hint to inject into agent's prompt."""
        lines = [f"[MEMORY] Past experience with {incident_type}:"]

        if skill and skill.times_used >= self.SKILL_MIN_EPISODES:
            lines.append(
                f"Best strategy (success_rate={skill.success_rate:.0%}, "
                f"avg_reward={skill.avg_reward:.2f}):"
            )
            for i, step in enumerate(skill.action_sequence[:3]):
                lines.append(
                    f"  Step {i+1}: {step['action_type']}({step['target']})"
                )

        if episodes:
            resolved_count = sum(1 for e in episodes if e.resolved)
            lines.append(
                f"Recent episodes: {resolved_count}/{len(episodes)} resolved"
            )
            # Add most recent lesson
            latest = max(episodes, key=lambda e: e.timestamp)
            lines.append(f"Latest lesson: {latest.lesson}")

        return "\n".join(lines)

    def _persist_episode(self, mem: EpisodeMemory) -> None:
        """Persist episode memory to disk."""
        try:
            path = self._storage_dir / f"episode_{mem.episode_id}.json"
            with open(path, "w") as f:
                json.dump(asdict(mem), f)
        except Exception:
            pass

    def _persist_skills(self) -> None:
        """Persist skill library to disk."""
        try:
            path = self._storage_dir / "skill_library.json"
            data = {k: asdict(v) for k, v in self._skill_library.items()}
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_from_disk(self) -> None:
        """Load persisted memory from disk."""
        try:
            # Load skill library
            skill_path = self._storage_dir / "skill_library.json"
            if skill_path.exists():
                with open(skill_path) as f:
                    data = json.load(f)
                for itype, skill_data in data.items():
                    self._skill_library[itype] = SkillRecord(**skill_data)

            # Load recent episodes (last 50)
            episode_files = sorted(
                self._storage_dir.glob("episode_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:self.EPISODIC_MEMORY_SIZE]

            for ep_file in reversed(episode_files):
                try:
                    with open(ep_file) as f:
                        data = json.load(f)
                    mem = EpisodeMemory(**data)
                    self._episodic_memory.append(mem)
                    self._semantic_index[mem.incident_type].append(mem.episode_id)
                except Exception:
                    pass
        except Exception:
            pass


# Module-level singleton
_memory: Optional[PersistentAgentMemory] = None


def get_agent_memory() -> PersistentAgentMemory:
    global _memory
    if _memory is None:
        _memory = PersistentAgentMemory()
    return _memory
