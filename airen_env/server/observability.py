# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
Production Observability — Upgrade #8

Prometheus-compatible metrics export for AIREN.
Enables enterprise monitoring via Grafana/Datadog dashboards.

Real pain point: You can't deploy AI agents in production without
observability. This module exports real-time metrics that ops teams
can monitor with their existing tooling.

Metrics exported:
- airen_resolution_time_seconds (histogram)
- airen_incidents_total (counter by type + resolved)
- airen_diagnosis_accuracy (gauge by incident type)
- airen_api_calls_total (counter)
- airen_reward_total (histogram)
- airen_loop_detections_total (counter)
- airen_schema_drift_total (counter)
- airen_context_poisoning_total (counter)
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricSample:
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: float = field(default_factory=time.time)
    metric_type: str = "gauge"      # gauge | counter | histogram


class PrometheusMetricsCollector:
    """
    Lightweight Prometheus-compatible metrics collector.

    Exports metrics in Prometheus text format at /metrics endpoint.
    Compatible with Grafana, Datadog, and any Prometheus scraper.

    No external dependencies — pure Python implementation.
    """

    def __init__(self) -> None:
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._start_time = time.time()

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        key = _labels_key(labels or {})
        self._counters[name][key] += value

    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric."""
        key = _labels_key(labels or {})
        self._gauges[name][key] = value

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation."""
        key = _labels_key(labels or {})
        self._histograms[name][key].append(value)
        # Keep last 1000 observations per label set
        if len(self._histograms[name][key]) > 1000:
            self._histograms[name][key] = self._histograms[name][key][-1000:]

    def export_prometheus(self) -> str:
        """
        Export all metrics in Prometheus text format.
        Compatible with /metrics endpoint scraping.
        """
        lines: List[str] = []
        lines.append(f"# HELP airen_uptime_seconds Time since server start")
        lines.append(f"# TYPE airen_uptime_seconds gauge")
        lines.append(f"airen_uptime_seconds {time.time() - self._start_time:.1f}")
        lines.append("")

        # Counters
        for name, label_values in self._counters.items():
            lines.append(f"# HELP {name} AIREN counter metric")
            lines.append(f"# TYPE {name} counter")
            for label_key, value in label_values.items():
                lines.append(f"{name}{{{label_key}}} {value:.4f}")
            lines.append("")

        # Gauges
        for name, label_values in self._gauges.items():
            lines.append(f"# HELP {name} AIREN gauge metric")
            lines.append(f"# TYPE {name} gauge")
            for label_key, value in label_values.items():
                lines.append(f"{name}{{{label_key}}} {value:.4f}")
            lines.append("")

        # Histograms (simplified: sum, count, quantiles)
        for name, label_observations in self._histograms.items():
            lines.append(f"# HELP {name} AIREN histogram metric")
            lines.append(f"# TYPE {name} histogram")
            for label_key, observations in label_observations.items():
                if not observations:
                    continue
                count = len(observations)
                total = sum(observations)
                sorted_obs = sorted(observations)
                p50 = _quantile(sorted_obs, 0.50)
                p95 = _quantile(sorted_obs, 0.95)
                p99 = _quantile(sorted_obs, 0.99)
                lk = f"{{{label_key}," if label_key else "{"
                lk_close = "}" if label_key else "}"
                lines.append(f'{name}_bucket{{{label_key}{"," if label_key else ""}le="0.5"}} {sum(1 for o in observations if o <= 0.5)}')
                lines.append(f'{name}_bucket{{{label_key}{"," if label_key else ""}le="1.0"}} {sum(1 for o in observations if o <= 1.0)}')
                lines.append(f'{name}_bucket{{{label_key}{"," if label_key else ""}le="+Inf"}} {count}')
                lines.append(f"{name}_sum{{{label_key}}} {total:.4f}")
                lines.append(f"{name}_count{{{label_key}}} {count}")
                lines.append(f"# p50={p50:.3f} p95={p95:.3f} p99={p99:.3f}")
            lines.append("")

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """Export metrics as JSON for /metrics/json endpoint."""
        result: Dict[str, Any] = {
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "counters": {},
            "gauges": {},
            "histograms": {},
        }

        for name, label_values in self._counters.items():
            result["counters"][name] = {
                lk: round(v, 4) for lk, v in label_values.items()
            }

        for name, label_values in self._gauges.items():
            result["gauges"][name] = {
                lk: round(v, 4) for lk, v in label_values.items()
            }

        for name, label_observations in self._histograms.items():
            result["histograms"][name] = {}
            for lk, obs in label_observations.items():
                if obs:
                    sorted_obs = sorted(obs)
                    result["histograms"][name][lk] = {
                        "count": len(obs),
                        "sum": round(sum(obs), 4),
                        "p50": round(_quantile(sorted_obs, 0.50), 4),
                        "p95": round(_quantile(sorted_obs, 0.95), 4),
                        "p99": round(_quantile(sorted_obs, 0.99), 4),
                        "min": round(min(obs), 4),
                        "max": round(max(obs), 4),
                    }

        return result


class ProductionObservabilityDashboard:
    """
    Grafana/Datadog-style metrics for AIREN production monitoring.

    Tracks all key operational metrics:
    - Incident resolution rates and times
    - Diagnosis accuracy by incident type
    - API cost tracking
    - Loop detection events
    - Schema drift events
    - Context poisoning events
    """

    def __init__(self) -> None:
        self._metrics = PrometheusMetricsCollector()
        self._episode_count: int = 0
        self._start_time = time.time()

    def track_episode_start(
        self,
        incident_type: str,
        severity: str,
    ) -> None:
        """Record episode start."""
        self._episode_count += 1
        self._metrics.counter(
            "airen_episodes_total",
            labels={"incident_type": incident_type, "severity": severity},
        )

    def track_episode_end(
        self,
        incident_type: str,
        resolved: bool,
        steps_taken: int,
        cumulative_reward: float,
        diagnosis_score: float,
        api_calls: int = 0,
    ) -> None:
        """Record episode completion metrics."""
        # Resolution counter
        self._metrics.counter(
            "airen_incidents_total",
            labels={
                "incident_type": incident_type,
                "resolved": str(resolved).lower(),
            },
        )

        # Resolution time (steps as proxy for time)
        self._metrics.histogram(
            "airen_resolution_steps",
            steps_taken,
            labels={"incident_type": incident_type},
        )

        # Reward histogram
        self._metrics.histogram(
            "airen_reward_total",
            cumulative_reward,
            labels={"incident_type": incident_type},
        )

        # Diagnosis accuracy gauge
        self._metrics.gauge(
            "airen_diagnosis_accuracy",
            diagnosis_score,
            labels={"incident_type": incident_type},
        )

        # API cost tracking
        if api_calls > 0:
            self._metrics.counter(
                "airen_api_calls_total",
                api_calls,
                labels={"incident_type": incident_type},
            )

    def track_loop_detection(
        self,
        pattern: str,
        tokens_saved: int,
    ) -> None:
        """Record loop detection event."""
        self._metrics.counter(
            "airen_loop_detections_total",
            labels={"pattern": pattern},
        )
        self._metrics.counter(
            "airen_tokens_saved_total",
            tokens_saved,
        )

    def track_schema_drift(
        self,
        tool_name: str,
        severity: str,
        auto_fixed: bool,
    ) -> None:
        """Record schema drift event."""
        self._metrics.counter(
            "airen_schema_drift_total",
            labels={
                "tool": tool_name,
                "severity": severity,
                "auto_fixed": str(auto_fixed).lower(),
            },
        )

    def track_context_poisoning(
        self,
        attack_type: str,
        quarantined: bool,
    ) -> None:
        """Record context poisoning detection."""
        self._metrics.counter(
            "airen_context_poisoning_total",
            labels={
                "attack_type": attack_type,
                "quarantined": str(quarantined).lower(),
            },
        )

    def track_compliance_violation(
        self,
        framework: str,
        severity: str,
        blocked: bool,
    ) -> None:
        """Record compliance violation."""
        self._metrics.counter(
            "airen_compliance_violations_total",
            labels={
                "framework": framework,
                "severity": severity,
                "blocked": str(blocked).lower(),
            },
        )

    def track_arl_event(
        self,
        event_type: str,   # circuit_break | rollback | ledger_compress
    ) -> None:
        """Record ARL event."""
        self._metrics.counter(
            "airen_arl_events_total",
            labels={"event_type": event_type},
        )

    def get_prometheus_text(self) -> str:
        """Get Prometheus-format metrics for /metrics endpoint."""
        return self._metrics.export_prometheus()

    def get_json_metrics(self) -> Dict[str, Any]:
        """Get JSON metrics for /metrics/json endpoint."""
        return {
            "metrics": self._metrics.export_json(),
            "episode_count": self._episode_count,
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    def get_grafana_dashboard(self) -> Dict[str, Any]:
        """
        Pre-built Grafana dashboard configuration.
        Import into Grafana for instant AIREN monitoring.
        """
        return {
            "dashboard": {
                "title": "AIREN Production Monitoring",
                "uid": "airen-prod-v1",
                "tags": ["airen", "ai-agents", "incident-response"],
                "refresh": "10s",
                "panels": [
                    {
                        "id": 1,
                        "title": "Incident Resolution Rate",
                        "type": "stat",
                        "targets": [{
                            "expr": (
                                "sum(rate(airen_incidents_total{resolved='true'}[5m])) / "
                                "sum(rate(airen_incidents_total[5m]))"
                            ),
                            "legendFormat": "Resolution Rate",
                        }],
                        "fieldConfig": {
                            "defaults": {"unit": "percentunit", "thresholds": {
                                "steps": [
                                    {"color": "red", "value": 0},
                                    {"color": "yellow", "value": 0.6},
                                    {"color": "green", "value": 0.8},
                                ]
                            }},
                        },
                    },
                    {
                        "id": 2,
                        "title": "Mean Steps To Resolution (MSTR)",
                        "type": "stat",
                        "targets": [{
                            "expr": "histogram_quantile(0.95, airen_resolution_steps_bucket)",
                            "legendFormat": "p95 Steps",
                        }],
                    },
                    {
                        "id": 3,
                        "title": "Diagnosis Accuracy by Incident Type",
                        "type": "bargauge",
                        "targets": [{
                            "expr": "avg(airen_diagnosis_accuracy) by (incident_type)",
                            "legendFormat": "{{incident_type}}",
                        }],
                    },
                    {
                        "id": 4,
                        "title": "Reward Distribution",
                        "type": "histogram",
                        "targets": [{
                            "expr": "airen_reward_total_bucket",
                            "legendFormat": "Reward",
                        }],
                    },
                    {
                        "id": 5,
                        "title": "Loop Detections (Tokens Saved)",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "rate(airen_loop_detections_total[5m])",
                            "legendFormat": "{{pattern}}",
                        }],
                    },
                    {
                        "id": 6,
                        "title": "Schema Drift Events",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "rate(airen_schema_drift_total[5m])",
                            "legendFormat": "{{tool}} ({{severity}})",
                        }],
                    },
                    {
                        "id": 7,
                        "title": "Context Poisoning Detections",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "rate(airen_context_poisoning_total[5m])",
                            "legendFormat": "{{attack_type}}",
                        }],
                    },
                    {
                        "id": 8,
                        "title": "Compliance Violations",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "rate(airen_compliance_violations_total[5m])",
                            "legendFormat": "{{framework}} ({{severity}})",
                        }],
                    },
                    {
                        "id": 9,
                        "title": "ARL Events (Circuit Breaks + Rollbacks)",
                        "type": "timeseries",
                        "targets": [{
                            "expr": "rate(airen_arl_events_total[5m])",
                            "legendFormat": "{{event_type}}",
                        }],
                    },
                    {
                        "id": 10,
                        "title": "API Cost Tracking",
                        "type": "stat",
                        "targets": [{
                            "expr": "sum(airen_api_calls_total)",
                            "legendFormat": "Total API Calls",
                        }],
                    },
                ],
            },
            "datasource": {
                "type": "prometheus",
                "url": "http://localhost:9090",
            },
            "import_instructions": (
                "1. Open Grafana → Dashboards → Import\n"
                "2. Paste this JSON\n"
                "3. Select your Prometheus datasource\n"
                "4. Click Import"
            ),
        }


def _labels_key(labels: Dict[str, str]) -> str:
    """Convert labels dict to Prometheus label string."""
    if not labels:
        return ""
    return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))


def _quantile(sorted_values: List[float], q: float) -> float:
    """Compute quantile from sorted list."""
    if not sorted_values:
        return 0.0
    idx = int(q * len(sorted_values))
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


# Module-level singleton
_dashboard: Optional[ProductionObservabilityDashboard] = None


def get_observability_dashboard() -> ProductionObservabilityDashboard:
    global _dashboard
    if _dashboard is None:
        _dashboard = ProductionObservabilityDashboard()
    return _dashboard
