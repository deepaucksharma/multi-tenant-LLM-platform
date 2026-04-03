'use client';

interface SidebarProps {
  currentTenant: string;
  onTenantChange: (tenant: string) => void;
  viewMode: string;
  onViewChange: (view: any) => void;
}

const SAMPLE_QUESTIONS: Record<string, string[]> = {
  sis: [
    'What documents are required for enrollment?',
    'How does FERPA protect student records?',
    'What happens after 5 unexcused absences?',
    'What is the grade change procedure?',
    'Who can access IEP records?',
    'Can a homeless student be denied enrollment?',
    'How are mid-year transfer grades calculated?',
    'What is the directory information opt-out process?',
  ],
  mfg: [
    'What is the assembly line startup sequence?',
    'How are critical defects handled?',
    'What is the lockout/tagout procedure?',
    'What triggers a CAPA?',
    'What PPE is required in the welding zone?',
    'What is the First Article Inspection process?',
    'How is the Risk Priority Number calculated?',
    'What are the maintenance priority response times?',
  ],
};

export default function Sidebar({ currentTenant, onTenantChange, viewMode, onViewChange }: SidebarProps) {
  const questions = SAMPLE_QUESTIONS[currentTenant] || [];

  return (
    <aside className="w-72 bg-gray-900 text-white flex flex-col h-screen">
      {/* Logo */}
      <div className="px-4 py-5 border-b border-gray-700">
        <h2 className="text-lg font-bold">LLM Platform</h2>
        <p className="text-xs text-gray-400 mt-1">Multi-Tenant POC</p>
      </div>

      {/* Tenant switch */}
      <div className="px-4 py-3 border-b border-gray-700">
        <p className="text-xs text-gray-400 uppercase tracking-wider mb-2">Tenant</p>
        <div className="flex gap-2">
          <button
            onClick={() => onTenantChange('sis')}
            className={`flex-1 py-2 rounded-md text-sm font-medium transition-colors ${
              currentTenant === 'sis'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            SIS
          </button>
          <button
            onClick={() => onTenantChange('mfg')}
            className={`flex-1 py-2 rounded-md text-sm font-medium transition-colors ${
              currentTenant === 'mfg'
                ? 'bg-green-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            MFG
          </button>
        </div>
      </div>

      {/* Sample questions */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        <p className="text-xs text-gray-400 uppercase tracking-wider mb-2">Try these questions</p>
        <div className="space-y-1.5">
          {questions.map((q, i) => (
            <button
              key={i}
              onClick={() => {
                const event = new CustomEvent('sample-question', { detail: q });
                window.dispatchEvent(event);
                onViewChange('chat');
              }}
              className="w-full text-left px-3 py-2 text-sm text-gray-300 rounded-md
                         hover:bg-gray-700 transition-colors line-clamp-2"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-gray-700">
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span className="w-2 h-2 bg-green-400 rounded-full"></span>
          Server connected
        </div>
        <p className="text-xs text-gray-600 mt-1">
          Qwen2.5-1.5B + LoRA adapters
        </p>
      </div>
    </aside>
  );
}
