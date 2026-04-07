# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""FastAPI application for AIREN environment."""

import os
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
_ENV_DIR = _SERVER_DIR.parent
_REPO_ROOT = _ENV_DIR.parents[1]
for _p in [str(_REPO_ROOT / "src"), str(_ENV_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import inspect as _inspect
from fastapi.responses import HTMLResponse, JSONResponse
from openenv.core.env_server.http_server import create_app

try:
    from ..models import AIRENAction, AIRENObservation
    from .airen_environment import AIRENEnvironment
except ImportError:
    from models import AIRENAction, AIRENObservation
    from server.airen_environment import AIRENEnvironment


# ── Gradio web UI (shown at /web) ─────────────────────────────────────────────

def _build_gradio_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    try:
        import gradio as gr
        import json as _json
    except ImportError:
        return None

    INCIDENT_TYPES = ["db_overload", "memory_leak", "network_partition", "bad_deployment", "cache_stampede"]
    ACTIONS = ["run_diagnostic", "inspect_logs", "inspect_metrics", "apply_fix",
               "rollback_deployment", "restart_service", "scale_service", "acknowledge_incident"]
    SERVICES = ["db", "api", "cache", "worker", "payment", "network"]

    async def run_step(incident_type, seed, action_type, target, reasoning):
        try:
            reset_data = await web_manager.reset_environment({
                "incident_type": incident_type, "seed": int(seed)
            })
            obs = reset_data.get("observation", {})
            health = obs.get("system_health", 0)
            severity = obs.get("severity", "")
            logs = "\n".join(obs.get("logs", [])[-3:])
            alerts = "\n".join(a.get("message", "") for a in obs.get("alerts", [])[:2])

            step_data = await web_manager.step_environment({
                "action_type": action_type, "target": target,
                "reasoning": reasoning, "parameters": {}
            })
            step_obs = step_data.get("observation", {})
            reward = step_data.get("reward", 0)
            new_health = step_obs.get("system_health", 0)
            resolved = step_obs.get("incident_resolved", False)
            explanation = step_obs.get("reward_explanation", "")
            diagnosis_quality = step_obs.get("diagnosis_quality", "N/A")

            incident_md = (
                f"### {incident_type.upper()} | Severity: {severity}\n\n"
                f"**Health:** {health:.0%}  **Threat:** {obs.get('threat_level', 0):.2f}\n\n"
                f"**Recent Logs:**\n```\n{logs}\n```\n\n"
                f"**Alerts:** {alerts or 'None'}"
            )
            result_md = (
                f"### {'✅ RESOLVED' if resolved else '⚡ Step Result'}\n\n"
                f"**Reward:** {reward:.3f}  **New Health:** {new_health:.0%}\n\n"
                f"**Reward Breakdown:** {explanation[:150]}\n\n"
                f"**Diagnosis Quality:** {diagnosis_quality}"
            )
            return incident_md, result_md, _json.dumps(step_data, indent=2)
        except Exception as e:
            return f"Error: {e}", "", ""

    with gr.Blocks(title="AIREN — AI Incident Response", theme=gr.themes.Soft()) as blocks:
        gr.Markdown(
            "# AIREN — AI Production Incident Response & Recovery\n"
            "A **true multi-step RL environment** — world degrades every step, "
            "cascading failures, partial observability."
        )

        with gr.Tabs():

            # ── Tab 1: Playground ─────────────────────────────────────────────
            with gr.Tab("Playground"):
                with gr.Row():
                    with gr.Column(scale=1):
                        incident_dd = gr.Dropdown(
                            choices=INCIDENT_TYPES, value="db_overload",
                            label="Incident Type"
                        )
                        seed_sl = gr.Slider(0, 100, value=42, step=1,
                                            label="Seed (change for different scenario)")
                        action_dd = gr.Dropdown(
                            choices=ACTIONS, value="run_diagnostic",
                            label="Action"
                        )
                        target_dd = gr.Dropdown(
                            choices=SERVICES, value="db",
                            label="Target Service"
                        )
                        reasoning_tb = gr.Textbox(
                            lines=2, label="Reasoning",
                            placeholder="DB CPU at 95%, slow queries in logs"
                        )
                        run_btn = gr.Button("Execute Action", variant="primary")
                        gr.Markdown("""
**Strategy:**
1. First `run_diagnostic` or `inspect_logs` to find root cause
2. Then apply the correct fix
3. Wrong actions make things worse
4. World degrades every step — act fast
""")
                    with gr.Column(scale=2):
                        incident_md = gr.Markdown("### Click 'Execute Action' to start\nA production incident will be generated.")
                        result_md = gr.Markdown("")
                        raw_json = gr.Code(label="Raw JSON Response", language="json", interactive=False)

                run_btn.click(
                    fn=run_step,
                    inputs=[incident_dd, seed_sl, action_dd, target_dd, reasoning_tb],
                    outputs=[incident_md, result_md, raw_json],
                )

            # ── Tab 2: Dashboard ──────────────────────────────────────────────
            with gr.Tab("Dashboard"):
                gr.HTML("""
<iframe
  src="/ui"
  style="width:100%;height:800px;border:none;border-radius:8px;"
  title="AIREN Dashboard">
</iframe>
<p style="font-size:0.8rem;color:#718096;margin-top:8px;">
  Direct links:
  <a href="/ui" target="_blank">/ui</a> |
  <a href="/docs" target="_blank">/docs</a> |
  <a href="/health" target="_blank">/health</a>
</p>
""")

    return blocks


# ── Core app ──────────────────────────────────────────────────────────────────

_sig = _inspect.signature(create_app)
if "gradio_builder" in _sig.parameters:
    app = create_app(
        AIRENEnvironment,
        AIRENAction,
        AIRENObservation,
        env_name="airen_env",
        max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "64")),
        gradio_builder=_build_gradio_ui,
    )
else:
    app = create_app(
        AIRENEnvironment,
        AIRENAction,
        AIRENObservation,
        env_name="airen_env",
        max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "64")),
    )


# ── /ui dashboard ─────────────────────────────────────────────────────────────

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AIREN Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #f7f8fa; color: #1a202c; }
  .header { background: linear-gradient(135deg, #c53030, #e53e3e);
            padding: 20px 30px; border-bottom: 2px solid #fc8181;
            display: flex; align-items: center; gap: 16px; }
  .header h1 { font-size: 1.4rem; color: #fff; }
  .badge { background: #48bb78; color: #000; padding: 3px 10px;
           border-radius: 12px; font-size: 0.75rem; font-weight: bold; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 16px; padding: 20px; }
  .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
  .card h2 { font-size: 0.85rem; color: #64748b; text-transform: uppercase;
             letter-spacing: 0.05em; margin-bottom: 12px; }
  .metric { display: flex; justify-content: space-between; align-items: center;
            padding: 6px 0; border-bottom: 1px solid #e2e8f0; }
  .metric:last-child { border-bottom: none; }
  .label { color: #64748b; font-size: 0.85rem; }
  .value { font-weight: bold; font-size: 0.9rem; }
  .green { color: #38a169; } .red { color: #e53e3e; }
  .blue { color: #3182ce; } .yellow { color: #d69e2e; }
  .score-bar { height: 8px; background: #e2e8f0; border-radius: 4px; margin-top: 8px; }
  .score-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
  .incident-row { padding: 8px; background: #f7f8fa; border-radius: 6px; margin-bottom: 6px; }
  .incident-name { font-size: 0.8rem; color: #64748b; }
  .event-item { font-size: 0.78rem; padding: 5px 0; border-bottom: 1px solid #e2e8f0;
                display: flex; gap: 8px; }
  .event-item:last-child { border-bottom: none; }
  .event-type { color: #3182ce; font-weight: bold; min-width: 80px; }
  .refresh-btn { background: #3182ce; color: #fff; border: none; padding: 8px 16px;
                 border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
  .refresh-btn:hover { background: #2c5282; }
  .loading { color: #94a3b8; font-size: 0.85rem; }
  .endpoint-list a { color: #3182ce; text-decoration: none; display: block;
                     padding: 3px 0; border-bottom: 1px solid #e2e8f0; font-size: 0.78rem; }
  .endpoint-list a:hover { color: #2c5282; }
  .endpoint-list a:last-child { border-bottom: none; }
  .layer-tag { font-size: 0.65rem; background: #ebf8ff; color: #2b6cb0;
               padding: 1px 6px; border-radius: 8px; margin-left: 6px; }
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>AIREN <span class="badge">LIVE</span></h1>
    <div style="font-size:0.75rem;color:rgba(255,255,255,0.8);margin-top:4px;">
      AI Production Incident Response &amp; Recovery — True RL Environment
    </div>
  </div>
  <div style="margin-left:auto;display:flex;gap:8px;align-items:center;">
    <button class="refresh-btn" onclick="loadAll()">Refresh</button>
    <span id="last-updated" class="loading" style="color:#fff">Loading...</span>
  </div>
</div>

<div class="grid">

  <div class="card">
    <h2>RL Environment <span class="layer-tag">Core</span></h2>
    <div class="metric"><span class="label">Episode Type</span><span class="value blue">Multi-Step MDP</span></div>
    <div class="metric"><span class="label">Max Steps</span><span class="value blue">10 per episode</span></div>
    <div class="metric"><span class="label">Reward Type</span><span class="value blue">Dense (every step)</span></div>
    <div class="metric"><span class="label">Observability</span><span class="value yellow">Partial (symptoms only)</span></div>
    <div class="metric"><span class="label">Concurrent Sessions</span><span class="value green">64</span></div>
    <div class="metric"><span class="label">World Dynamics</span><span class="value red">Autonomous degradation</span></div>
  </div>

  <div class="card">
    <h2>Incident Types <span class="layer-tag">5 Tasks</span></h2>
    <div id="incidents-content"><div class="loading">Loading...</div></div>
  </div>

  <div class="card">
    <h2>Live Episodes <span class="layer-tag">Real-time</span></h2>
    <div id="events-content"><div class="loading">Loading...</div></div>
  </div>

  <div class="card">
    <h2>Action Space <span class="layer-tag">9 Actions</span></h2>
    <div class="metric"><span class="label">inspect_logs</span><span class="value blue">diagnostic</span></div>
    <div class="metric"><span class="label">inspect_metrics</span><span class="value blue">diagnostic</span></div>
    <div class="metric"><span class="label">run_diagnostic</span><span class="value blue">diagnostic</span></div>
    <div class="metric"><span class="label">apply_fix</span><span class="value green">fix</span></div>
    <div class="metric"><span class="label">rollback_deployment</span><span class="value green">fix</span></div>
    <div class="metric"><span class="label">restart_service</span><span class="value yellow">risky</span></div>
    <div class="metric"><span class="label">scale_service</span><span class="value yellow">risky</span></div>
    <div class="metric"><span class="label">acknowledge_incident</span><span class="value blue">admin</span></div>
    <div class="metric"><span class="label">ignore_alert</span><span class="value red">always penalized</span></div>
  </div>

  <div class="card">
    <h2>Reward Function <span class="layer-tag">Multi-Objective</span></h2>
    <div class="metric"><span class="label">recovery × 0.25</span><span class="value green">health delta</span></div>
    <div class="metric"><span class="label">diagnosis × 0.20</span><span class="value green">right service+action</span></div>
    <div class="metric"><span class="label">efficiency × 0.10</span><span class="value green">act early</span></div>
    <div class="metric"><span class="label">threat_mitigation × 0.10</span><span class="value green">reduce threat</span></div>
    <div class="metric"><span class="label">resolve_bonus × 0.15</span><span class="value green">full resolution</span></div>
    <div class="metric"><span class="label">hallucination_penalty</span><span class="value red">−0.10</span></div>
    <div class="metric"><span class="label">cost_penalty</span><span class="value red">−0.05</span></div>
  </div>

  <div class="card">
    <h2>API Endpoints</h2>
    <div class="endpoint-list">
      <a href="/health" target="_blank">/health — Health check</a>
      <a href="/docs" target="_blank">/docs — Swagger UI</a>
      <a href="/reset" target="_blank">/reset — Start episode</a>
      <a href="/step" target="_blank">/step — Execute action</a>
      <a href="/state" target="_blank">/state — Episode state</a>
      <a href="/schema" target="_blank">/schema — Action/Obs schemas</a>
      <a href="/web" target="_blank">/web — Interactive playground</a>
    </div>
  </div>

</div>

<script>
async function fetchJSON(url) {
  try { const r = await fetch(url); return await r.json(); }
  catch(e) { return null; }
}

function scoreColor(s) {
  if (s >= 0.8) return 'green'; if (s >= 0.5) return 'yellow'; return 'red';
}

async function loadIncidents() {
  const incidents = [
    {id: "db_overload",       name: "Database Overload",    diff: "medium", fix: "run_diagnostic → apply_fix on db"},
    {id: "memory_leak",       name: "Memory Leak",          diff: "medium", fix: "inspect_logs → restart_service on worker"},
    {id: "network_partition", name: "Network Partition",    diff: "hard",   fix: "run_diagnostic → apply_fix on network"},
    {id: "bad_deployment",    name: "Bad Deployment",       diff: "easy",   fix: "rollback_deployment on payment"},
    {id: "cache_stampede",    name: "Cache Stampede",       diff: "hard",   fix: "apply_fix on cache → scale_service on db"},
  ];
  const colors = {easy: 'green', medium: 'yellow', hard: 'red'};
  let html = '';
  for (const inc of incidents) {
    html += `<div class="incident-row">
      <div style="display:flex;justify-content:space-between">
        <span style="font-weight:bold;font-size:0.85rem">${inc.name}</span>
        <span class="${colors[inc.diff]}" style="font-size:0.75rem">${inc.diff}</span>
      </div>
      <div class="incident-name" style="margin-top:3px">${inc.fix}</div>
    </div>`;
  }
  document.getElementById('incidents-content').innerHTML = html;
}

async function loadEvents() {
  const d = await fetchJSON('/state');
  if (!d) { document.getElementById('events-content').innerHTML = '<div class="loading">No episodes yet</div>'; return; }
  let html = '';
  const actions = d.actions_taken || [];
  if (actions.length === 0) {
    html = '<div class="loading">No episodes yet — run inference or use Playground</div>';
  } else {
    html += `<div class="metric"><span class="label">Steps taken</span><span class="value blue">${d.steps_taken || 0}</span></div>`;
    html += `<div class="metric"><span class="label">Incident type</span><span class="value blue">${d.incident_type || 'N/A'}</span></div>`;
    html += `<div class="metric"><span class="label">Resolved</span><span class="value ${d.incident_resolved ? 'green' : 'red'}">${d.incident_resolved ? 'YES' : 'NO'}</span></div>`;
    html += `<div class="metric"><span class="label">Cumulative reward</span><span class="value ${scoreColor(d.cumulative_reward || 0)}">${(d.cumulative_reward || 0).toFixed(3)}</span></div>`;
    html += `<div style="margin-top:8px;font-size:0.75rem;color:#64748b">Last actions: ${actions.slice(-3).join(', ')}</div>`;
  }
  document.getElementById('events-content').innerHTML = html;
}

async function loadAll() {
  document.getElementById('last-updated').textContent = 'Refreshing...';
  await Promise.all([loadIncidents(), loadEvents()]);
  document.getElementById('last-updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

loadAll();
setInterval(loadAll, 15000);
</script>
</body>
</html>"""


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return HTMLResponse(content=_UI_HTML)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
