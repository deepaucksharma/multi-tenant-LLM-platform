'use client';

import ReactMarkdown from 'react-markdown';

interface Citation {
  title: string;
  topic: string;
  source_file: string;
  relevance_score: number;
  citation_key: string;
}

interface MessageBubbleProps {
  role: 'user' | 'assistant' | 'system';
  content: string;
  citations?: Citation[];
  metadata?: {
    model_type?: string;
    latency_ms?: number;
    grounding_score?: number;
    retrieval_method?: string;
    is_canary?: boolean;
    request_id?: string;
  };
  isStreaming?: boolean;
  tenantId?: string;
  onFeedback?: (rating: number) => void;
}

export default function MessageBubble({
  role,
  content,
  citations,
  metadata,
  isStreaming,
  tenantId,
  onFeedback,
}: MessageBubbleProps) {
  if (role === 'user') {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[75%] bg-blue-600 text-white rounded-2xl rounded-br-md px-4 py-3">
          <p className="text-sm whitespace-pre-wrap">{content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start mb-4">
      <div className="max-w-[85%]">
        {/* Avatar */}
        <div className="flex items-center gap-2 mb-1">
          <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs
            ${tenantId === 'mfg' ? 'bg-green-100 text-green-700' : 'bg-blue-100 text-blue-700'}`}
          >
            {tenantId === 'mfg' ? 'M' : 'S'}
          </span>
          <span className="text-xs text-gray-500">
            {tenantId?.toUpperCase()} Assistant
          </span>
          {metadata?.is_canary && (
            <span className="text-xs bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded">
              canary
            </span>
          )}
        </div>

        {/* Message body */}
        <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-md px-4 py-3 shadow-sm">
          <div className={`message-content text-sm text-gray-800 ${isStreaming ? 'streaming-cursor' : ''}`}>
            <ReactMarkdown>{content || ' '}</ReactMarkdown>
          </div>

          {/* Citations */}
          {citations && citations.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-100">
              <p className="text-xs font-medium text-gray-500 mb-1.5">Sources</p>
              <div className="space-y-1">
                {citations.map((c, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 text-xs bg-gray-50 rounded-md px-2 py-1.5"
                  >
                    <span className={`w-5 h-5 rounded flex items-center justify-center text-white text-[10px] font-bold
                      ${tenantId === 'mfg' ? 'bg-green-500' : 'bg-blue-500'}`}
                    >
                      {i + 1}
                    </span>
                    <span className="text-gray-700 font-medium">{c.title}</span>
                    <span className="text-gray-400">•</span>
                    <span className="text-gray-500">{c.topic}</span>
                    <span className="ml-auto text-gray-400">
                      {(c.relevance_score * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Metadata & Feedback */}
          {metadata && !isStreaming && (
            <div className="mt-3 pt-2 border-t border-gray-100 flex items-center justify-between">
              <div className="flex gap-3 text-[11px] text-gray-400">
                {metadata.latency_ms && (
                  <span>{Math.round(metadata.latency_ms)}ms</span>
                )}
                {metadata.model_type && (
                  <span>{metadata.model_type}</span>
                )}
                {metadata.grounding_score !== undefined && metadata.grounding_score !== null && (
                  <span>{(metadata.grounding_score * 100).toFixed(0)}% grounded</span>
                )}
                {metadata.retrieval_method && (
                  <span>{metadata.retrieval_method}</span>
                )}
              </div>
              {onFeedback && (
                <div className="flex gap-1">
                  <button
                    onClick={() => onFeedback(5)}
                    className="p-1 hover:bg-green-50 rounded text-gray-400 hover:text-green-600"
                    title="Good response"
                  >
                    +1
                  </button>
                  <button
                    onClick={() => onFeedback(1)}
                    className="p-1 hover:bg-red-50 rounded text-gray-400 hover:text-red-600"
                    title="Bad response"
                  >
                    -1
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
