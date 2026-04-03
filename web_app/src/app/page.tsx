'use client';

import { useState } from 'react';
import Chat from './components/Chat';
import TenantSelector from './components/TenantSelector';
import Sidebar from './components/Sidebar';
import MonitoringPanel from './components/MonitoringPanel';

type ViewMode = 'chat' | 'monitoring' | 'compare';

export default function Home() {
  const [tenantId, setTenantId] = useState<string>('sis');
  const [viewMode, setViewMode] = useState<ViewMode>('chat');

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <Sidebar
        currentTenant={tenantId}
        onTenantChange={setTenantId}
        viewMode={viewMode}
        onViewChange={setViewMode}
      />

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold text-gray-800">
              {tenantId === 'sis' ? 'SIS Assistant' : 'Manufacturing Assistant'}
            </h1>
            <span className={`text-xs px-2 py-1 rounded-full font-medium ${
              tenantId === 'sis'
                ? 'bg-blue-100 text-blue-700'
                : 'bg-green-100 text-green-700'
            }`}>
              {tenantId.toUpperCase()}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <TenantSelector
              currentTenant={tenantId}
              onTenantChange={setTenantId}
            />
            <div className="flex gap-1">
              {(['chat', 'monitoring', 'compare'] as ViewMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-3 py-1.5 text-sm rounded-md capitalize ${
                    viewMode === mode
                      ? 'bg-gray-800 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {viewMode === 'chat' && (
            <Chat tenantId={tenantId} />
          )}
          {viewMode === 'monitoring' && (
            <MonitoringPanel tenantId={tenantId} />
          )}
          {viewMode === 'compare' && (
            <div className="flex h-full gap-2 p-2">
              <div className="flex-1 border rounded-lg overflow-hidden">
                <div className="bg-blue-50 px-3 py-2 text-sm font-medium text-blue-700 border-b">
                  SIS Tenant
                </div>
                <Chat tenantId="sis" compact />
              </div>
              <div className="flex-1 border rounded-lg overflow-hidden">
                <div className="bg-green-50 px-3 py-2 text-sm font-medium text-green-700 border-b">
                  MFG Tenant
                </div>
                <Chat tenantId="mfg" compact />
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
