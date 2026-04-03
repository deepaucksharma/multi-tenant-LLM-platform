'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import MessageBubble from './MessageBubble';

interface Citation {
  title: string;
  topic: string;
  source_file: string;
  relevance_score: number;
  citation_key: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  metadata?: Record<string, any>;
  isStreaming?: boolean;
}

interface ChatProps {
  tenantId: string;
  compact?: boolean;
}

const API_BASE = '/api';

export default function Chat({ tenantId, compact = false }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [useRag, setUseRag] = useState(true);
  const [useStreaming, setUseStreaming] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Listen for sample question events from sidebar
  useEffect(() => {
    const handler = (e: CustomEvent) => {
      setInput(e.detail);
      inputRef.current?.focus();
    };
    window.addEventListener('sample-question', handler as EventListener);
    return () => window.removeEventListener('sample-question', handler as EventListener);
  }, []);

  // Clear messages when tenant changes
  useEffect(() => {
    setMessages([]);
  }, [tenantId]);

  const generateId = () => Math.random().toString(36).substring(2, 10);

  const sendMessage = async () => {
    const userMessage = input.trim();
    if (!userMessage || isLoading) return;

    const userMsg: Message = {
      id: generateId(),
      role: 'user',
      content: userMessage,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    const assistantId = generateId();

    if (useStreaming) {
      await sendStreaming(userMessage, assistantId);
    } else {
      await sendNonStreaming(userMessage, assistantId);
    }

    setIsLoading(false);
  };

  const sendStreaming = async (userMessage: string, assistantId: string) => {
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: 'assistant', content: '', isStreaming: true },
    ]);

    try {
      const conversationHistory = messages.slice(-6).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      abortRef.current = new AbortController();

      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: tenantId,
          message: userMessage,
          conversation_history: conversationHistory,
          use_rag: useRag,
          use_streaming: true,
          max_new_tokens: 512,
          temperature: 0.7,
        }),
        signal: abortRef.current.signal,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';
      let fullContent = '';
      let citations: Citation[] = [];
      let metadata: Record<string, any> = {};

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event:')) continue;
          if (line.startsWith('data:')) {
            const dataStr = line.substring(5).trim();
            if (!dataStr) continue;

            try {
              const data = JSON.parse(dataStr);

              if (data.token !== undefined) {
                fullContent += data.token;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: fullContent, isStreaming: true }
                      : m
                  )
                );
              }

              if (data.citations) {
                citations = data.citations;
              }

              if (data.request_id) {
                metadata = {
                  request_id: data.request_id,
                  latency_ms: data.latency_ms,
                  model_type: data.model_type,
                  is_canary: data.is_canary,
                };
              }

              if (data.error) {
                fullContent += `\n\nError: ${data.error}`;
              }
            } catch {
              // Skip malformed JSON
            }
          }
        }
      }

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: fullContent, citations, metadata, isStreaming: false }
            : m
        )
      );
    } catch (error: any) {
      if (error.name === 'AbortError') return;

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: `Connection error: ${error.message}. Is the server running on port 8000?`,
                isStreaming: false,
              }
            : m
        )
      );
    }
  };

  const sendNonStreaming = async (userMessage: string, assistantId: string) => {
    try {
      const conversationHistory = messages.slice(-6).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: tenantId,
          message: userMessage,
          conversation_history: conversationHistory,
          use_rag: useRag,
          use_streaming: false,
          max_new_tokens: 512,
          temperature: 0.7,
        }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      const assistantMsg: Message = {
        id: assistantId,
        role: 'assistant',
        content: data.message,
        citations: data.citations,
        metadata: {
          request_id: data.request_id,
          latency_ms: data.latency_ms,
          model_type: data.model_type,
          grounding_score: data.grounding_score,
          retrieval_method: data.retrieval_method,
          is_canary: false,
        },
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (error: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: 'assistant',
          content: `Error: ${error.message}. Make sure the inference server is running.`,
        },
      ]);
    }
  };

  const handleFeedback = async (requestId: string, rating: number) => {
    try {
      await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          request_id: requestId,
          tenant_id: tenantId,
          rating,
          feedback_type: 'thumbs',
        }),
      });
    } catch {
      // Silently fail for feedback
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <p className="text-lg font-medium">
                {tenantId === 'sis' ? 'SIS Assistant' : 'Manufacturing Assistant'}
              </p>
              <p className="text-sm mt-1">
                Ask a question about{' '}
                {tenantId === 'sis'
                  ? 'enrollment, attendance, grading, FERPA...'
                  : 'SOPs, quality control, safety, maintenance...'}
              </p>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            role={msg.role}
            content={msg.content}
            citations={msg.citations}
            metadata={msg.metadata}
            isStreaming={msg.isStreaming}
            tenantId={tenantId}
            onFeedback={
              msg.role === 'assistant' && msg.metadata?.request_id
                ? (rating) => handleFeedback(msg.metadata!.request_id, rating)
                : undefined
            }
          />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-gray-200 bg-white px-4 py-3">
        {/* Options */}
        <div className="flex items-center gap-4 mb-2">
          <label className="flex items-center gap-1.5 text-xs text-gray-500">
            <input
              type="checkbox"
              checked={useRag}
              onChange={(e) => setUseRag(e.target.checked)}
              className="rounded border-gray-300 text-blue-600"
            />
            RAG
          </label>
          <label className="flex items-center gap-1.5 text-xs text-gray-500">
            <input
              type="checkbox"
              checked={useStreaming}
              onChange={(e) => setUseStreaming(e.target.checked)}
              className="rounded border-gray-300 text-blue-600"
            />
            Stream
          </label>
          {messages.length > 0 && (
            <button
              onClick={() => setMessages([])}
              className="text-xs text-gray-400 hover:text-gray-600 ml-auto"
            >
              Clear chat
            </button>
          )}
        </div>

        {/* Input */}
        <div className="flex gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Ask about ${tenantId === 'sis' ? 'student information' : 'manufacturing operations'}...`}
            className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-2.5
                       text-sm focus:outline-none focus:ring-2 focus:ring-blue-500
                       focus:border-transparent min-h-[44px] max-h-[120px]"
            rows={1}
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            className={`px-4 py-2.5 rounded-xl text-sm font-medium text-white
                       transition-colors disabled:opacity-50 disabled:cursor-not-allowed
                       ${tenantId === 'mfg'
                         ? 'bg-green-600 hover:bg-green-700'
                         : 'bg-blue-600 hover:bg-blue-700'
                       }`}
          >
            {isLoading ? '...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}
