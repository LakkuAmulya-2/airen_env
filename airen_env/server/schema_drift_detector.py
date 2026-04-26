# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Schema Drift Defender — Upgrade #1

Detects and auto-fixes tool schema changes that break agent behavior.
Solves the n8n February 2026 production incident pattern where schema
changes silently broke function calling across OpenAI and Anthropic.

Real pain point: Meta's production agents break when tool schemas drift
between versions — type='None' instead of 'object', missing required fields,
incompatible parameter names.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SchemaDriftEvent:
    tool_name: str
    old_version: str
    new_version: str
    breaking_changes: List[Dict[str, Any]]
    auto_fixed: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class SchemaBreakingChange:
    severity: str          # CRITICAL | HIGH | MEDIUM
    provider: str          # OpenAI | Anthropic | Generic
    error: str
    fix: str
    field_path: str = ""


class SchemaDriftDefender:
    """
    Detect tool schema changes that break agent behavior.

    Tracks schema evolution across versions and validates compatibility
    with OpenAI and Anthropic function-calling formats.

    Integrated into the AIREN environment to validate tool schemas
    on every reset — catches drift before it breaks training.
    """

    def __init__(self) -> None:
        # tool_name -> list of {version, schema, timestamp}
        self._registry: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._drift_events: List[SchemaDriftEvent] = []
        self._auto_fix_count: int = 0

    def reset(self) -> None:
        self._drift_events.clear()
        self._auto_fix_count = 0

    def register_tool_schema(
        self,
        tool_name: str,
        schema: Dict[str, Any],
        version: str,
    ) -> None:
        """Track schema evolution across versions."""
        self._registry[tool_name].append({
            "version": version,
            "schema": schema,
            "timestamp": time.time(),
        })

    def detect_drift(
        self,
        tool_name: str,
        old_version: str,
        new_version: str,
    ) -> List[SchemaBreakingChange]:
        """
        Compare two schema versions and return breaking changes.
        Validates against OpenAI and Anthropic requirements.
        """
        old = self._get_schema(tool_name, old_version)
        new = self._get_schema(tool_name, new_version)
        if old is None or new is None:
            return []

        breaking: List[SchemaBreakingChange] = []

        # ── OpenAI schema validation ──────────────────────────────────────────
        # n8n Feb 2026 bug: type was set to string "None" instead of "object"
        if str(new.get("type", "object")) == "None":
            breaking.append(SchemaBreakingChange(
                severity="CRITICAL",
                provider="OpenAI",
                error="Invalid schema: type='None' (must be 'object')",
                fix="Set type='object' in schema root",
                field_path="type",
            ))

        if new.get("type") not in ("object", None) and "properties" in new:
            breaking.append(SchemaBreakingChange(
                severity="HIGH",
                provider="OpenAI",
                error=f"Schema type must be 'object' when properties present, got '{new.get('type')}'",
                fix="Set type='object'",
                field_path="type",
            ))

        # Required fields removed
        old_required = set(old.get("required", []))
        new_required = set(new.get("required", []))
        removed_required = old_required - new_required
        for field_name in removed_required:
            breaking.append(SchemaBreakingChange(
                severity="MEDIUM",
                provider="OpenAI",
                error=f"Required field '{field_name}' removed — may break existing callers",
                fix=f"Add '{field_name}' back to required list or mark optional",
                field_path=f"required[{field_name}]",
            ))

        # Properties removed
        old_props = set(old.get("properties", {}).keys())
        new_props = set(new.get("properties", {}).keys())
        removed_props = old_props - new_props
        for prop in removed_props:
            breaking.append(SchemaBreakingChange(
                severity="HIGH",
                provider="OpenAI",
                error=f"Property '{prop}' removed — breaks agents using this field",
                fix=f"Keep '{prop}' as deprecated optional field",
                field_path=f"properties.{prop}",
            ))

        # ── Anthropic schema validation ───────────────────────────────────────
        if "input_schema" in new:
            if "type" not in new["input_schema"]:
                breaking.append(SchemaBreakingChange(
                    severity="CRITICAL",
                    provider="Anthropic",
                    error="Missing required field: input_schema.type",
                    fix="Add 'type': 'object' to input_schema",
                    field_path="input_schema.type",
                ))
            if str(new["input_schema"].get("type", "object")) == "None":
                breaking.append(SchemaBreakingChange(
                    severity="CRITICAL",
                    provider="Anthropic",
                    error="input_schema.type='None' is invalid",
                    fix="Set input_schema.type='object'",
                    field_path="input_schema.type",
                ))

        # Record drift event
        if breaking:
            self._drift_events.append(SchemaDriftEvent(
                tool_name=tool_name,
                old_version=old_version,
                new_version=new_version,
                breaking_changes=[
                    {"severity": b.severity, "provider": b.provider,
                     "error": b.error, "fix": b.fix}
                    for b in breaking
                ],
                auto_fixed=False,
            ))

        return breaking

    def auto_fix_schema(
        self,
        broken_schema: Dict[str, Any],
        provider: str,
    ) -> Dict[str, Any]:
        """
        Auto-fix common schema drift issues.
        Returns a corrected copy — never mutates the original.
        """
        fixed = _deep_copy(broken_schema)

        if provider in ("OpenAI", "Generic"):
            # Fix type='None' → 'object'
            if str(fixed.get("type", "object")) == "None":
                fixed["type"] = "object"
                self._auto_fix_count += 1

            # Ensure type is present when properties exist
            if "properties" in fixed and "type" not in fixed:
                fixed["type"] = "object"
                self._auto_fix_count += 1

        if provider == "Anthropic" and "input_schema" in fixed:
            if "type" not in fixed["input_schema"]:
                fixed["input_schema"]["type"] = "object"
                self._auto_fix_count += 1
            if str(fixed["input_schema"].get("type", "object")) == "None":
                fixed["input_schema"]["type"] = "object"
                self._auto_fix_count += 1

        return fixed

    def validate_schema(
        self,
        tool_name: str,
        schema: Dict[str, Any],
    ) -> List[SchemaBreakingChange]:
        """
        Validate a single schema against both OpenAI and Anthropic requirements.
        Used on every tool registration to catch issues immediately.
        """
        issues: List[SchemaBreakingChange] = []

        # OpenAI checks
        if str(schema.get("type", "object")) == "None":
            issues.append(SchemaBreakingChange(
                severity="CRITICAL", provider="OpenAI",
                error="type='None' is invalid", fix="Set type='object'",
                field_path="type",
            ))
        if "properties" in schema and schema.get("type") not in ("object", None):
            issues.append(SchemaBreakingChange(
                severity="HIGH", provider="OpenAI",
                error=f"type must be 'object' when properties present",
                fix="Set type='object'", field_path="type",
            ))

        # Anthropic checks
        if "input_schema" in schema:
            if "type" not in schema["input_schema"]:
                issues.append(SchemaBreakingChange(
                    severity="CRITICAL", provider="Anthropic",
                    error="input_schema.type missing",
                    fix="Add type='object' to input_schema",
                    field_path="input_schema.type",
                ))

        return issues

    def reward_schema_resilience(self, schema_changes: List[Dict[str, Any]]) -> float:
        """
        Reward bonus for gracefully handling schema drift.
        Integrated into the AIREN reward function.
        """
        if not schema_changes:
            return 0.0
        graceful = sum(
            1 for c in schema_changes
            if c.get("agent_action") == "request_schema_update"
            and c.get("fallback_used") is True
        )
        return round(0.05 * (graceful / len(schema_changes)), 4)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "registered_tools": len(self._registry),
            "drift_events": len(self._drift_events),
            "auto_fixes_applied": self._auto_fix_count,
            "recent_events": [
                {
                    "tool": e.tool_name,
                    "versions": f"{e.old_version} → {e.new_version}",
                    "breaking_changes": len(e.breaking_changes),
                    "auto_fixed": e.auto_fixed,
                }
                for e in self._drift_events[-5:]
            ],
        }

    def _get_schema(
        self, tool_name: str, version: str
    ) -> Optional[Dict[str, Any]]:
        entries = self._registry.get(tool_name, [])
        for entry in entries:
            if entry["version"] == version:
                return entry["schema"]
        return None


def _deep_copy(obj: Any) -> Any:
    """Simple deep copy without importing copy module."""
    import json
    return json.loads(json.dumps(obj))


# Module-level singleton
_defender: Optional[SchemaDriftDefender] = None


def get_schema_defender() -> SchemaDriftDefender:
    global _defender
    if _defender is None:
        _defender = SchemaDriftDefender()
    return _defender
