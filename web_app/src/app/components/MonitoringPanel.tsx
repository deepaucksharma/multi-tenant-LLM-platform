'use client';

import { useState, useEffect, useCallback } from 'react';

interface Stats {
  total_requests: number;
  avg_latency_ms: number;
  max_latency_ms: number;
  min_latency_ms: number;
  refusals: number;
  blocked: number;
  canary_requests: number;
  avg_grounding_score: number;
  avg_feedback_rating: number;
  feedback_count: number;
  tenant_id: string;
  period_hours: number;
}

interface RecentRequest {
  request_id: string;
  tenant_id: string;
  timestamp: string;
  user_message: string;
  model_type: string;
  latency_ms: number;
  was_refused: number;
  grounding_score: number | null;
}

interface ModelStats {
  base_loaded: boolean;
  active_adapter: string | null;
  available_adapters: string[];
  load_count: number;
  generation_count: number;
  gpu_memory: Record<string, any>;
}

const API_BASE = '/api';

export default function MonitoringPanel({ tenantId }: { tenantId: string }) {
  const [stats, setStats] = useState<Stats | null>(null);
  const [recentRequests, setRecentRequests] = useState<RecentRequest[]>([]);
  const [modelStats, setModelStats] = useState<ModelStats | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [statsRes, recentRes, modelRes] = await Promise.allSettled([
        fetch(`${API_BASE}/stats?tenant_id=${tenantId}&hours=24`),
        fetch(`${API_BASE}/stats/recent?tenant_id=${tenantId}&limit=15`),
        fetch(`${API_BASE}/model/stats`),
      ]);

      if (statsRes.status === 'fulfilled' && statsRes.value.ok) {
        setStats(await statsRes.value.json());
      }
      if (recentRes.status === 'fulfilled' && recentRes.value.ok) {
        setRecentRequests(await recentRes.value.json());
      }
      if (modelRes.status === 'fulfilled' && modelRes.value.ok) {
        setModelStats(await modelRes.value.json());
      }
      setError(null);
    } catch (e: any) {
      setError(`Connection error: ${e.message}`);
    }
  }, [tenantId]);

  useEffect(() => {
    fetchData();
    if (autoRefresh) {
      const interval = setInterval(fetchData, 5000);
      return () => clearInterval(interval);
    }
  }, [fetchData, autoRefresh]);

  const StatCard = ({
    label,
    value,
    unit,
    color,
  }: {
    label: string;
    value: string | number;
    unit?: string;
    color?: string;
  }) => (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wider">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${color || 'text-gray-800'}`}>
        {value}
        {unit && <span className="text-sm font-normal text-gray-400 ml-1">{unit}</span>}
      </p>
    </div>
  );

  return (
    <div className="h-full overflow-y-auto p-6 bg-gray-50">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-bold text-gray-800">Monitoring Dashboard</h2>
          <p className="text-sm text-gray-500">
            Tenant: {tenantId.toUpperCase()} | Last 24 hours
          </p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 text-sm text-gray-500">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded"
            />
            Auto-refresh
          </label>
          <button
            onClick={fetchData}
            className="px-3 py-1.5 text-sm bg-gray-200 rounded-md hover:bg-gray-300"
          >
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard label="Total Requests" value={stats?.total_requests || 0} />
        <StatCard
          label="Avg Latency"
          value={Math.round(stats?.avg_latency_ms || 0)}
          unit="ms"
          color={
            (stats?.avg_latency_ms || 0) > 5000
              ? 'text-red-600'
              : (stats?.avg_latency_ms || 0) > 2000
              ? 'text-yellow-600'
              : 'text-green-600'
          }
        />
        <StatCard
          label="Avg Grounding"
          value={
            stats?.avg_grounding_score
              ? `${(stats.avg_grounding_score * 100).toFixed(0)}%`
              : 'N/A'
          }
          color={
            (stats?.avg_grounding_score || 0) > 0.7
              ? 'text-green-600'
              : 'text-yellow-600'
          }
        />
        <StatCard
          label="Feedback Rating"
          value={stats?.avg_feedback_rating?.toFixed(1) || 'N/A'}
          unit={stats?.feedback_count ? `(${stats.feedback_count})` : ''}
        />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard label="Refusals" value={stats?.refusals || 0} />
        <StatCard label="Blocked" value={stats?.blocked || 0} />
        <StatCard label="Canary Requests" value={stats?.canary_requests || 0} />
        <StatCard
          label="Max Latency"
          value={Math.round(stats?.max_latency_ms || 0)}
          unit="ms"
        />
      </div>

      {/* Model Info */}
      {modelStats && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Model Status</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Base Model</p>
              <p className="font-medium flex items-center gap-1">
                <span className={`w-2 h-2 rounded-full ${modelStats.base_loaded ? 'bg-green-400' : 'bg-gray-400'}`}></span>
                {modelStats.base_loaded ? 'Loaded' : 'Not loaded'}
              </p>
            </div>
            <div>
              <p className="text-gray-500">Active Adapter</p>
              <p className="font-medium">{modelStats.active_adapter || 'None'}</p>
            </div>
            <div>
              <p className="text-gray-500">Total Generations</p>
              <p className="font-medium">{modelStats.generation_count}</p>
            </div>
            <div>
              <p className="text-gray-500">GPU Memory</p>
              <p className="font-medium">
                {modelStats.gpu_memory?.allocated_gb
                  ? `${modelStats.gpu_memory.allocated_gb} GB`
                  : 'N/A'}
              </p>
            </div>
          </div>
          {modelStats.available_adapters.length > 0 && (
            <div className="mt-3">
              <p className="text-xs text-gray-500 mb-1">Available Adapters:</p>
              <div className="flex gap-1 flex-wrap">
                {modelStats.available_adapters.map((a) => (
                  <span
                    key={a}
                    className={`text-xs px-2 py-0.5 rounded-full ${
                      a === modelStats.active_adapter
                        ? 'bg-blue-100 text-blue-700 font-medium'
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    {a}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Recent Requests */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">
          Recent Requests ({tenantId.toUpperCase()})
        </h3>
        {recentRequests.length === 0 ? (
          <p className="text-sm text-gray-400 py-4 text-center">No requests yet</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100">
                  <th className="text-left py-2 px-2 text-gray-500 font-medium">Time</th>
                  <th className="text-left py-2 px-2 text-gray-500 font-medium">Message</th>
                  <th className="text-left py-2 px-2 text-gray-500 font-medium">Model</th>
                  <th className="text-right py-2 px-2 text-gray-500 font-medium">Latency</th>
                  <th className="text-center py-2 px-2 text-gray-500 font-medium">Grounding</th>
                  <th className="text-center py-2 px-2 text-gray-500 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {recentRequests.map((req) => (
                  <tr key={req.request_id} className="border-b border-gray-50 hover:bg-gray-50">
                    <td className="py-2 px-2 text-gray-400 text-xs whitespace-nowrap">
                      {new Date(req.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2 px-2 text-gray-700 max-w-[200px] truncate">
                      {req.user_message}
                    </td>
                    <td className="py-2 px-2">
                      <span className="text-xs bg-gray-100 px-1.5 py-0.5 rounded">
                        {req.model_type || 'base'}
                      </span>
                    </td>
                    <td className="py-2 px-2 text-right text-gray-600">
                      {Math.round(req.latency_ms)}ms
                    </td>
                    <td className="py-2 px-2 text-center">
                      {req.grounding_score !== null ? (
                        <span className={`text-xs font-medium ${
                          req.grounding_score > 0.7
                            ? 'text-green-600'
                            : req.grounding_score > 0.4
                            ? 'text-yellow-600'
                            : 'text-red-600'
                        }`}>
                          {(req.grounding_score * 100).toFixed(0)}%
                        </span>
                      ) : (
                        <span className="text-gray-300">—</span>
                      )}
                    </td>
                    <td className="py-2 px-2 text-center">
                      {req.was_refused ? (
                        <span className="text-xs bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded">
                          refused
                        </span>
                      ) : (
                        <span className="text-xs bg-green-100 text-green-700 px-1.5 py-0.5 rounded">
                          ok
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
