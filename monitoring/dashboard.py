"""
Monitoring dashboard server.
Provides a web-based dashboard for system observability.

Usage:
    python monitoring/dashboard.py
"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

load_dotenv()

from monitoring.metrics_collector import get_metrics_collector
from monitoring.alerting import get_alert_manager

app = FastAPI(title="LLM Platform Monitoring", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def dashboard_ui():
    return DASHBOARD_HTML


@app.get("/api/metrics")
async def get_metrics(hours: int = Query(24, ge=1, le=168)):
    collector = get_metrics_collector()
    return collector.collect_all(hours)


@app.get("/api/alerts")
async def get_alerts():
    mgr = get_alert_manager()
    return {
        "active": mgr.get_active_alerts(),
        "recent": mgr.get_all_alerts(limit=20),
    }


@app.post("/api/alerts/check")
async def check_alerts(hours: int = Query(1, ge=1, le=24)):
    mgr = get_alert_manager()
    new_alerts = mgr.evaluate_rules(hours)
    return {
        "new_alerts": len(new_alerts),
        "alerts": [
            {"severity": a.severity, "message": a.message}
            for a in new_alerts
        ],
    }


@app.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    mgr = get_alert_manager()
    mgr.resolve_alert(alert_id)
    return {"status": "resolved"}


@app.get("/api/timeseries")
async def get_timeseries(
    tenant_id: str = Query(None),
    hours: int = Query(24),
):
    collector = get_metrics_collector()
    return collector.collect_latency_timeseries(tenant_id, hours)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Platform — Monitoring</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }
.header { background: #1e293b; padding: 16px 24px; border-bottom: 1px solid #334155;
  display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 1.2em; }
.header .controls { display: flex; gap: 8px; align-items: center; }
.header select, .header button { padding: 6px 12px; border-radius: 6px; border: 1px solid #475569;
  background: #1e293b; color: #e2e8f0; font-size: 13px; cursor: pointer; }
.header button:hover { background: #334155; }

.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px; padding: 16px 24px; }
.card { background: #1e293b; border-radius: 10px; padding: 16px; border: 1px solid #334155; }
.card-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
.card-value { font-size: 28px; font-weight: 700; margin-top: 4px; }
.card-sub { font-size: 12px; color: #64748b; margin-top: 2px; }
.green { color: #4ade80; } .yellow { color: #fbbf24; } .red { color: #f87171; } .blue { color: #60a5fa; }

.section { padding: 0 24px 16px; }
.section h2 { font-size: 1em; color: #94a3b8; margin-bottom: 12px; padding-top: 16px;
  border-top: 1px solid #1e293b; }

.alerts { display: flex; flex-direction: column; gap: 6px; }
.alert-item { padding: 10px 14px; border-radius: 8px; font-size: 13px; display: flex;
  align-items: center; gap: 8px; }
.alert-critical { background: #450a0a; border: 1px solid #7f1d1d; }
.alert-warning { background: #422006; border: 1px solid #78350f; }
.alert-info { background: #0c1a3d; border: 1px solid #1e3a5f; }
.alert-badge { font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; }

table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 8px 12px; color: #94a3b8; font-weight: 500;
  border-bottom: 1px solid #334155; }
td { padding: 8px 12px; border-bottom: 1px solid #1e293b; }
tr:hover { background: #1e293b; }

.status-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.status-production { background: #065f46; color: #6ee7b7; }
.status-staging { background: #78350f; color: #fcd34d; }
.status-archived { background: #374151; color: #9ca3af; }

.bar-chart { display: flex; align-items: flex-end; gap: 2px; height: 60px; margin-top: 8px; }
.bar { background: #3b82f6; border-radius: 2px 2px 0 0; min-width: 4px; flex: 1;
  transition: height 0.3s; }
.bar:hover { background: #60a5fa; }

.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 768px) { .two-col { grid-template-columns: 1fr; } }

.refresh-indicator { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.refresh-active { background: #4ade80; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
</style>
</head>
<body>
<div class="header">
  <h1>LLM Platform Monitor</h1>
  <div class="controls">
    <span class="refresh-indicator refresh-active" id="refreshDot"></span>
    <select id="periodSelect" onchange="refresh()">
      <option value="1">Last 1 hour</option>
      <option value="6">Last 6 hours</option>
      <option value="24" selected>Last 24 hours</option>
    </select>
    <button onclick="checkAlerts()">Check Alerts</button>
    <button onclick="refresh()">Refresh</button>
  </div>
</div>

<div class="grid" id="metricsGrid"></div>

<div class="section" id="alertsSection">
  <h2>Alerts</h2>
  <div class="alerts" id="alertsList">
    <div style="color: #64748b; font-size: 13px;">No alerts</div>
  </div>
</div>

<div class="section">
  <h2>Request Volume</h2>
  <div class="bar-chart" id="volumeChart"></div>
</div>

<div class="section two-col">
  <div>
    <h2>SIS Tenant</h2>
    <div id="sisMetrics"></div>
  </div>
  <div>
    <h2>MFG Tenant</h2>
    <div id="mfgMetrics"></div>
  </div>
</div>

<div class="section">
  <h2>Model Registry</h2>
  <table id="modelsTable">
    <thead>
      <tr><th>Tenant</th><th>Type</th><th>Version</th><th>Status</th><th>Train Loss</th><th>Eval Loss</th></tr>
    </thead>
    <tbody id="modelsBody"></tbody>
  </table>
</div>

<div class="section">
  <h2>System Resources</h2>
  <div class="grid" id="systemGrid"></div>
</div>

<script>
let _tenantData = {};

async function refresh() {
  const hours = document.getElementById('periodSelect').value;
  try {
    const [metricsResp, alertsResp, tsResp] = await Promise.all([
      fetch('/api/metrics?hours=' + hours),
      fetch('/api/alerts'),
      fetch('/api/timeseries?hours=' + hours),
    ]);
    const metrics = await metricsResp.json();
    const alerts = await alertsResp.json();
    const timeseries = await tsResp.json();

    _tenantData = metrics.tenants || {};
    renderMetrics(metrics);
    renderAlerts(alerts);
    renderTimeseries(timeseries);
    renderModels(metrics.models || []);
    renderSystem(metrics.system || {});
    renderTenantDetails();
  } catch (e) {
    console.error('Refresh failed:', e);
  }
}

function renderMetrics(data) {
  const grid = document.getElementById('metricsGrid');
  let totalReqs = 0, totalAvgLat = 0, totalGround = 0, tenantCount = 0;

  for (const [tid, t] of Object.entries(data.tenants || {})) {
    totalReqs += t.total_requests || 0;
    totalAvgLat += t.avg_latency_ms || 0;
    totalGround += t.avg_grounding_score || 0;
    tenantCount++;
  }

  const avgLat = tenantCount ? totalAvgLat / tenantCount : 0;
  const avgGround = tenantCount ? totalGround / tenantCount : 0;
  const latColor = avgLat > 5000 ? 'red' : avgLat > 2000 ? 'yellow' : 'green';
  const groundColor = avgGround > 0.7 ? 'green' : avgGround > 0.4 ? 'yellow' : 'red';

  grid.innerHTML =
    '<div class="card"><div class="card-label">Total Requests</div><div class="card-value blue">' + totalReqs + '</div><div class="card-sub">' + data.period_hours + 'h window</div></div>' +
    '<div class="card"><div class="card-label">Avg Latency</div><div class="card-value ' + latColor + '">' + Math.round(avgLat) + 'ms</div></div>' +
    '<div class="card"><div class="card-label">Avg Grounding</div><div class="card-value ' + groundColor + '">' + (avgGround*100).toFixed(0) + '%</div></div>' +
    '<div class="card"><div class="card-label">Active Tenants</div><div class="card-value blue">' + tenantCount + '</div></div>';
}

function renderTenantDetails() {
  document.getElementById('sisMetrics').innerHTML = renderTenantDetail('sis', _tenantData.sis || {});
  document.getElementById('mfgMetrics').innerHTML = renderTenantDetail('mfg', _tenantData.mfg || {});
}

function renderTenantDetail(tid, t) {
  const latColor = (t.avg_latency_ms||0) > 5000 ? 'red' : (t.avg_latency_ms||0) > 2000 ? 'yellow' : 'green';
  return '<div class="grid" style="padding:0">' +
    '<div class="card"><div class="card-label">Requests</div><div class="card-value">' + (t.total_requests||0) + '</div></div>' +
    '<div class="card"><div class="card-label">Avg Latency</div><div class="card-value ' + latColor + '">' + Math.round(t.avg_latency_ms||0) + 'ms</div></div>' +
    '<div class="card"><div class="card-label">P95 Latency</div><div class="card-value">' + Math.round(t.p95_latency_ms||0) + 'ms</div></div>' +
    '<div class="card"><div class="card-label">Grounding</div><div class="card-value">' + ((t.avg_grounding_score||0)*100).toFixed(0) + '%</div></div>' +
    '<div class="card"><div class="card-label">Feedback</div><div class="card-value">' + (t.avg_feedback_rating||0).toFixed(1) + '<span class="card-sub"> (' + (t.feedback_count||0) + ')</span></div></div>' +
    '<div class="card"><div class="card-label">Errors</div><div class="card-value ' + ((t.error_count||0)>0?'red':'green') + '">' + (t.error_count||0) + '</div></div>' +
    '</div>';
}

function renderAlerts(data) {
  const list = document.getElementById('alertsList');
  const active = data.active || [];
  if (active.length === 0) {
    list.innerHTML = '<div style="color:#64748b;font-size:13px;">No active alerts</div>';
    return;
  }
  list.innerHTML = active.map(a =>
    '<div class="alert-item alert-' + a.severity + '">' +
    '<span class="alert-badge" style="background:' + (a.severity==='critical'?'#dc2626':a.severity==='warning'?'#d97706':'#2563eb') + ';color:white">' +
    a.severity.toUpperCase() + '</span>' +
    '<span>' + a.message + '</span>' +
    '<span style="margin-left:auto;font-size:11px;color:#64748b">' + (a.triggered_at||'').substring(11,19) + '</span>' +
    '</div>'
  ).join('');
}

function renderTimeseries(data) {
  const chart = document.getElementById('volumeChart');
  if (!data || data.length === 0) { chart.innerHTML = '<span style="color:#64748b;font-size:12px">No data</span>'; return; }
  const maxCount = Math.max(...data.map(d => d.request_count), 1);
  chart.innerHTML = data.map(d => {
    const h = Math.max(4, (d.request_count / maxCount) * 56);
    return '<div class="bar" style="height:' + h + 'px" title="' + d.timestamp + ': ' + d.request_count + ' requests"></div>';
  }).join('');
}

function renderModels(models) {
  const body = document.getElementById('modelsBody');
  if (!models.length) { body.innerHTML = '<tr><td colspan="6" style="color:#64748b">No models registered</td></tr>'; return; }
  body.innerHTML = models.map(m =>
    '<tr>' +
    '<td>' + m.tenant_id + '</td>' +
    '<td>' + m.model_type + '</td>' +
    '<td style="font-family:monospace;font-size:12px">' + m.version + '</td>' +
    '<td><span class="status-badge status-' + m.status + '">' + m.status + '</span></td>' +
    '<td>' + (m.last_train_loss != null ? m.last_train_loss.toFixed(3) : '—') + '</td>' +
    '<td>' + (m.last_eval_loss != null ? m.last_eval_loss.toFixed(3) : '—') + '</td>' +
    '</tr>'
  ).join('');
}

function renderSystem(sys) {
  const grid = document.getElementById('systemGrid');
  const cpuColor = (sys.cpu_percent||0) > 80 ? 'red' : (sys.cpu_percent||0) > 50 ? 'yellow' : 'green';
  const memColor = (sys.memory_percent||0) > 80 ? 'red' : (sys.memory_percent||0) > 60 ? 'yellow' : 'green';
  grid.innerHTML =
    '<div class="card"><div class="card-label">CPU</div><div class="card-value ' + cpuColor + '">' + (sys.cpu_percent||0).toFixed(0) + '%</div></div>' +
    '<div class="card"><div class="card-label">Memory</div><div class="card-value ' + memColor + '">' + (sys.memory_percent||0).toFixed(0) + '%</div><div class="card-sub">' + (sys.memory_used_gb||0) + ' / ' + ((sys.memory_used_gb||0)+(sys.memory_available_gb||0)).toFixed(1) + ' GB</div></div>' +
    '<div class="card"><div class="card-label">GPU VRAM</div><div class="card-value blue">' + (sys.gpu_allocated_gb != null ? sys.gpu_allocated_gb + ' GB' : 'N/A') + '</div><div class="card-sub">' + (sys.gpu_total_gb ? 'of '+sys.gpu_total_gb+' GB' : '') + '</div></div>' +
    '<div class="card"><div class="card-label">Disk</div><div class="card-value">' + (sys.disk_used_percent||0).toFixed(0) + '%</div></div>';
}

async function checkAlerts() {
  const resp = await fetch('/api/alerts/check?hours=1', { method: 'POST' });
  const data = await resp.json();
  alert(data.new_alerts > 0 ? data.new_alerts + ' new alert(s) detected!' : 'No new alerts.');
  refresh();
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("MONITOR_PORT", "8002"))
    logger.info(f"Starting monitoring dashboard on port {port}")
    uvicorn.run("monitoring.dashboard:app", host="0.0.0.0", port=port, reload=False)
