import React, { useState, useEffect, useRef } from 'react';
import { Header } from './components/Header';
import { DocumentSidebar } from './components/DocumentSidebar';
import { ChatPanel } from './components/ChatPanel';
import { IndexedDocument, ChatMessage, SystemConfig, SourceCitation } from './types';

export default function App() {
  const [documents, setDocuments] = useState<IndexedDocument[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Load documents and config on mount
  const loadInitialData = async () => {
    setLoadingDocs(true);
    setError(null);
    try {
      const [docsRes, cfgRes] = await Promise.all([
        fetch('/api/documents'),
        fetch('/api/config'),
      ]);

      if (docsRes.ok) {
        const docsData = await docsRes.json();
        setDocuments(docsData.documents || []);
      }

      if (cfgRes.ok) {
        const cfgData = await cfgRes.json();
        setConfig(cfgData);
      }
    } catch (err: any) {
      console.error('Initial load failed:', err);
      setError('Could not connect to RAG server. Make sure the backend service is running.');
    } finally {
      setLoadingDocs(false);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  // Upload file
  const handleUpload = async (file: File) => {
    setUploading(true);
    setUploadProgress(`Processing ${file.name}...`);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      setUploadProgress('Extracting text & generating embeddings...');
      const res = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || data.detail || 'Upload failed.');
      }

      // Refresh documents list & config
      await loadInitialData();
    } catch (err: any) {
      console.error('Upload error:', err);
      setError(err.message || 'Failed to upload document.');
    } finally {
      setUploading(false);
      setUploadProgress(null);
    }
  };

  // Delete document
  const handleDelete = async (filename: string) => {
    setError(null);
    try {
      const encoded = encodeURIComponent(filename);
      const res = await fetch(`/api/documents/${encoded}`, {
        method: 'DELETE',
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || data.detail || 'Delete failed.');
      }

      setDocuments((prev) => prev.filter((d) => d.filename !== filename));
      // Refresh count in config
      if (config) {
        const cfgRes = await fetch('/api/config');
        if (cfgRes.ok) {
          const cfgData = await cfgRes.json();
          setConfig(cfgData);
        }
      }
    } catch (err: any) {
      console.error('Delete error:', err);
      setError(err.message || 'Failed to delete document.');
    }
  };

  // Seed sample knowledge base
  const handleSeedSample = async (sampleKey: string) => {
    setUploading(true);
    setUploadProgress('Indexing sample knowledge base...');
    setError(null);

    try {
      const res = await fetch('/api/seed-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sampleKey }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Failed to seed sample knowledge base.');
      }

      await loadInitialData();
    } catch (err: any) {
      console.error('Seed error:', err);
      setError(err.message || 'Failed to index sample.');
    } finally {
      setUploading(false);
      setUploadProgress(null);
    }
  };

  // Switch LLM Provider
  const handleProviderChange = async (provider: 'gemini' | 'groq') => {
    try {
      const res = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider }),
      });

      if (res.ok) {
        setConfig((prev) => (prev ? { ...prev, current_provider: provider } : null));
      }
    } catch (err) {
      console.error('Provider change failed:', err);
    }
  };

  // Stop streaming
  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setStreaming(false);
    setMessages((prev) =>
      prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m))
    );
  };

  // Send question and stream answer
  const handleSendMessage = async (text: string) => {
    if (streaming) return;

    const userMessage: ChatMessage = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    const assistantMessageId = `asst_${Date.now()}`;
    const assistantPlaceholder: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      sources: [],
      isStreaming: true,
      timestamp: Date.now(),
      provider: config?.current_provider || 'gemini',
    };

    setMessages((prev) => [...prev, userMessage, assistantPlaceholder]);
    setStreaming(true);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      // Build history for conversational query rewriting
      const historyPayload = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: text,
          history: historyPayload,
          provider: config?.current_provider,
          top_k: config?.default_top_k || 4,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || 'Chat request failed.');
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No readable stream returned.');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        let currentEvent = 'message';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.substring(7).trim();
          } else if (line.startsWith('data: ')) {
            const dataStr = line.substring(6).trim();
            if (!dataStr) continue;

            try {
              const parsed = JSON.parse(dataStr);

              if (currentEvent === 'sources') {
                const sourcesList: SourceCitation[] = parsed;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessageId
                      ? { ...m, sources: sourcesList }
                      : m
                  )
                );
              } else if (currentEvent === 'token') {
                const token = parsed.token || '';
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessageId
                      ? { ...m, content: m.content + token }
                      : m
                  )
                );
              } else if (currentEvent === 'error') {
                const errMsg = parsed.error || 'Generation error.';
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessageId
                      ? {
                          ...m,
                          content:
                            m.content +
                            `\n\n⚠️ **Generation Error**: ${errMsg}`,
                          isStreaming: false,
                        }
                      : m
                  )
                );
              } else if (currentEvent === 'done') {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessageId
                      ? { ...m, isStreaming: false }
                      : m
                  )
                );
              }
            } catch (jsonErr) {
              // incomplete JSON chunk
            }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        console.error('Chat stream error:', err);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessageId
              ? {
                  ...m,
                  content:
                    m.content ||
                    `Sorry, an error occurred while generating the answer: ${err.message}`,
                  isStreaming: false,
                }
              : m
          )
        );
      }
    } finally {
      setStreaming(false);
      abortControllerRef.current = null;
      setMessages((prev) =>
        prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false } : m))
      );
    }
  };

  return (
    <div className="min-h-screen bg-[#F7F5F0] text-[#1C1C1A] flex flex-col font-serif selection:bg-[#E2DAC8] selection:text-[#1C1C1A]">
      {/* Top Header */}
      <Header
        config={config}
        onProviderChange={handleProviderChange}
        documentCount={documents.length}
        totalChunks={documents.reduce((acc, d) => acc + d.chunk_count, 0)}
        onToggleSidebar={() => setMobileSidebarOpen(!mobileSidebarOpen)}
        onUploadFile={handleUpload}
      />

      {/* Main Workspace Layout */}
      <div className="flex-1 flex relative overflow-hidden bg-[#F7F5F0]">
        {/* Mobile Sidebar Overlay */}
        {mobileSidebarOpen && (
          <div
            onClick={() => setMobileSidebarOpen(false)}
            className="fixed inset-0 z-25 bg-[#1C1C1A]/20 backdrop-blur-xs lg:hidden"
          />
        )}

        {/* Document Management Sidebar */}
        <DocumentSidebar
          documents={documents}
          isLoading={loadingDocs}
          uploading={uploading}
          uploadProgress={uploadProgress}
          error={error}
          onUpload={handleUpload}
          onDelete={handleDelete}
          onSeedSample={handleSeedSample}
          onRefresh={loadInitialData}
          isOpen={mobileSidebarOpen}
          onToggle={() => setMobileSidebarOpen(!mobileSidebarOpen)}
        />

        {/* Chat Interface Panel */}
        <ChatPanel
          messages={messages}
          streaming={streaming}
          onSendMessage={handleSendMessage}
          onStopStreaming={handleStopStreaming}
          onClearHistory={() => setMessages([])}
          activeProvider={config?.current_provider || 'gemini'}
          hasDocuments={documents.length > 0}
          onSeedPrompt={handleSendMessage}
          onUploadFile={handleUpload}
          uploading={uploading}
        />
      </div>
    </div>
  );
}
