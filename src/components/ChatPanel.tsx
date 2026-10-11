import React, { useState, useRef, useEffect, useMemo } from 'react';
import { ArrowRight, Square, Copy, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage, SourceCitation } from '../types';

interface ChatPanelProps {
  messages: ChatMessage[];
  streaming: boolean;
  onSendMessage: (text: string) => void;
  onStopStreaming: () => void;
  onClearHistory: () => void;
  activeProvider: string;
  hasDocuments: boolean;
  onSeedPrompt: (prompt: string) => void;
  onUploadFile?: (file: File) => void;
  uploading?: boolean;
}

// Inline Citation Pill or Super-script
const renderCitationsInText = (text: string, sources?: SourceCitation[]) => {
  const parts = text.split(/(\[Source:[^\]]+\]|\[\d+\])/g);
  return parts.map((part, i) => {
    if (part.startsWith('[Source:') || /^\[\d+\]$/.test(part)) {
      const numMatch = part.match(/\d+/);
      const label = numMatch ? `[${numMatch[0]}]` : part;
      return (
        <span
          key={i}
          className="inline-flex items-center px-1.5 py-0.2 mx-0.5 text-[11px] font-sans font-medium text-[#2C3E50] bg-[#EAE5D9] border border-[#DDD6C8] rounded hover:bg-[#DDD6C8] cursor-help align-baseline transition-colors select-none"
          title={part.replace(/^\[|\]$/g, '')}
        >
          {label}
        </span>
      );
    }
    return part;
  });
};

const MarkdownCodeBlock: React.FC<{
  className?: string;
  children?: React.ReactNode;
}> = ({ className, children }) => {
  const [copied, setCopied] = useState(false);
  const codeString = String(children).replace(/\n$/, '');
  const match = /language-(\w+)/.exec(className || '');
  const language = match ? match[1] : '';

  const handleCopy = () => {
    navigator.clipboard.writeText(codeString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative my-3 rounded border border-[#DDD6C8] bg-[#22252A] text-[#F3F4F6] text-xs font-mono overflow-hidden shadow-xs">
      <div className="flex items-center justify-between px-3 py-1.5 bg-[#181B1F] border-b border-[#32363D] text-[11px] text-[#9CA3AF]">
        <span>{language || 'code'}</span>
        <button
          onClick={handleCopy}
          type="button"
          className="flex items-center gap-1 hover:text-white transition-colors cursor-pointer px-1.5 py-0.5 rounded hover:bg-[#2A2E35]"
          title="Copy code"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre className="p-3.5 overflow-x-auto leading-relaxed">
        <code>{children}</code>
      </pre>
    </div>
  );
};

export const ChatPanel: React.FC<ChatPanelProps> = ({
  messages,
  streaming,
  onSendMessage,
  onStopStreaming,
  onClearHistory,
  activeProvider,
  hasDocuments,
  onSeedPrompt,
  onUploadFile,
  uploading,
}) => {
  const [input, setInput] = useState('');
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Time-aware greeting & archival quote
  const greetingData = useMemo(() => {
    const hour = new Date().getHours();
    let salutation = 'Good day';
    if (hour >= 5 && hour < 12) {
      salutation = 'Good morning';
    } else if (hour >= 12 && hour < 18) {
      salutation = 'Good afternoon';
    } else if (hour >= 18 && hour < 22) {
      salutation = 'Good evening';
    } else {
      salutation = 'Nice to see you';
    }

    const quotes = [
      '“A library is a house of records for the human spirit.”',
      '“In the quiet of the archive, every page awaits inquiry.”',
      '“Words, once gathered, form a bridge between questions and truth.”',
    ];
    const quote = quotes[Math.floor(Math.random() * quotes.length)];

    return { salutation, quote };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streaming]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || streaming) return;
    onSendMessage(input.trim());
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFilePicked = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0] && onUploadFile) {
      onUploadFile(e.target.files[0]);
      e.target.value = '';
    }
  };

  const toggleSources = (messageId: string) => {
    setExpandedSources((prev) => ({
      ...prev,
      [messageId]: !prev[messageId],
    }));
  };

  const renderMarkdownMessage = (content: string, sources?: SourceCitation[]) => {
    return (
      <div className="prose-editorial text-[#1C1C1A]">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            // Headers
            h1: ({ children }) => (
              <h1 className="text-xl sm:text-2xl font-serif font-bold text-[#1C1C1A] mt-5 mb-2.5 pb-1 border-b border-[#E5DFD3]">
                {children}
              </h1>
            ),
            h2: ({ children }) => (
              <h2 className="text-lg sm:text-xl font-serif font-semibold text-[#1C1C1A] mt-4 mb-2 pb-0.5 border-b border-[#E5DFD3]/70">
                {children}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="text-base sm:text-lg font-serif font-semibold text-[#1C1C1A] mt-3.5 mb-1.5">
                {children}
              </h3>
            ),
            h4: ({ children }) => (
              <h4 className="text-sm sm:text-base font-serif font-medium text-[#2C3E50] mt-3 mb-1">
                {children}
              </h4>
            ),
            // Paragraphs with inline citation detection
            p: ({ children }) => {
              const elements = React.Children.map(children, (child) => {
                if (typeof child === 'string') {
                  return renderCitationsInText(child, sources);
                }
                return child;
              });
              return <p className="my-2.5 leading-[1.75] text-[#1C1C1A]">{elements}</p>;
            },
            // Unordered list
            ul: ({ children }) => (
              <ul className="list-disc list-outside pl-5 my-2.5 space-y-1.5 marker:text-[#8B8578]">
                {children}
              </ul>
            ),
            // Ordered list
            ol: ({ children }) => (
              <ol className="list-decimal list-outside pl-5 my-2.5 space-y-1.5 marker:text-[#2C3E50] marker:font-medium">
                {children}
              </ol>
            ),
            li: ({ children }) => {
              const elements = React.Children.map(children, (child) => {
                if (typeof child === 'string') {
                  return renderCitationsInText(child, sources);
                }
                return child;
              });
              return <li className="pl-1 leading-relaxed text-[#1C1C1A]">{elements}</li>;
            },
            // Blockquote
            blockquote: ({ children }) => (
              <blockquote className="border-l-3 border-[#2C3E50] pl-3.5 py-1 my-3 italic text-[#4A4843] bg-[#F4EFE6]/60 rounded-r">
                {children}
              </blockquote>
            ),
            // Links
            a: ({ href, children }) => (
              <a
                href={href}
                target="_blank"
                rel="noreferrer noopener"
                className="text-[#2C3E50] underline underline-offset-2 decoration-[#2C3E50]/50 hover:decoration-[#2C3E50] font-medium transition-colors"
              >
                {children}
              </a>
            ),
            // Strong & emphasis
            strong: ({ children }) => (
              <strong className="font-semibold text-[#111110]">{children}</strong>
            ),
            em: ({ children }) => (
              <em className="italic text-[#33312E]">{children}</em>
            ),
            // Tables
            table: ({ children }) => (
              <div className="overflow-x-auto my-3 border border-[#DDD6C8] rounded">
                <table className="w-full text-left text-xs border-collapse">
                  {children}
                </table>
              </div>
            ),
            thead: ({ children }) => (
              <thead className="bg-[#EFEAE0] text-[#1C1C1A] font-semibold border-b border-[#DDD6C8]">
                {children}
              </thead>
            ),
            tbody: ({ children }) => (
              <tbody className="divide-y divide-[#DDD6C8]/60 bg-[#FAF8F5]">
                {children}
              </tbody>
            ),
            tr: ({ children }) => (
              <tr className="hover:bg-[#F4EFE6] transition-colors">{children}</tr>
            ),
            th: ({ children }) => (
              <th className="px-3 py-2 text-left font-serif font-medium text-[#2C3E50]">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="px-3 py-2 text-[#1C1C1A]">{children}</td>
            ),
            // Code & Code Blocks
            code: ({ className, children, ...props }) => {
              const isInline = !className && !String(children).includes('\n');
              if (isInline) {
                return (
                  <code
                    className="px-1.5 py-0.5 rounded bg-[#EAE5D9] text-[#7A2E2E] font-mono text-[13px] border border-[#DDD6C8]"
                    {...props}
                  >
                    {children}
                  </code>
                );
              }
              return <MarkdownCodeBlock className={className}>{children}</MarkdownCodeBlock>;
            },
            hr: () => <hr className="my-4 border-t border-[#DDD6C8]" />,
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    );
  };

  return (
    <main className="flex-1 flex flex-col h-[calc(100vh-3.25rem)] sm:h-[calc(100vh-3.5rem)] bg-[#F7F5F0] overflow-hidden font-serif">
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.txt,.md,.markdown,.py,.ts,.tsx,.js,.json,.csv,.sql,.html,.sh"
        className="hidden"
        onChange={handleFilePicked}
      />

      {/* Reading / Exchange stream */}
      <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-8">
        <div className="max-w-[700px] mx-auto w-full">
          {messages.length === 0 ? (
            /* Minimal empty state with warm greeting and literary quote */
            <div className="py-24 text-center space-y-3">
              <h2 className="text-xl sm:text-2xl font-normal text-[#1C1C1A] tracking-tight font-serif">
                {greetingData.salutation}, nice to see you.
              </h2>
              <blockquote className="text-xs sm:text-[13px] text-[#8B8578] italic font-serif max-w-sm mx-auto leading-relaxed">
                {greetingData.quote}
              </blockquote>
              <div className="w-10 h-[1px] bg-[#DDD6C8] mx-auto my-3" />
              <p className="text-xs text-[#8B8578] font-serif">
                {hasDocuments ? 'Ask a question' : 'No documents indexed yet'}
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((msg) => (
                <article
                  key={msg.id}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {msg.role === 'user' ? (
                    <div className="max-w-[85%] sm:max-w-[75%] bg-[#EFEAE0] border border-[#DDD6C8] p-3 sm:p-4 text-[#1C1C1A] text-[15px] sm:text-[16px] leading-[1.68]">
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                    </div>
                  ) : (
                    <div className="w-full bg-[#FDFCFA] border border-[#DDD6C8] p-4 sm:p-5 text-[#1C1C1A] text-[15px] sm:text-[16px] leading-[1.7]">
                      {/* Response text with Markdown and citations */}
                      <div className="break-words">
                        {renderMarkdownMessage(msg.content, msg.sources)}
                        {msg.isStreaming && (
                          <span className="inline-block w-1.5 h-4 ml-1 bg-[#2C3E50] animate-pulse align-middle" />
                        )}
                      </div>

                      {/* Footnote citations */}
                      {msg.sources && msg.sources.length > 0 && (
                        <footer className="mt-4 pt-3 border-t border-[#DDD6C8] text-xs">
                          <button
                            onClick={() => toggleSources(msg.id)}
                            className="text-[#2C3E50] hover:underline flex items-baseline justify-between w-full text-left"
                          >
                            <span className="italic">
                              Sources ({msg.sources.length})
                            </span>
                            <span className="text-[11px] text-[#8B8578]">
                              {expandedSources[msg.id] ? '−' : '+'}
                            </span>
                          </button>

                          {expandedSources[msg.id] && (
                            <ol className="mt-2.5 space-y-2 text-xs divide-y divide-[#DDD6C8]/50">
                              {msg.sources.map((src, index) => (
                                <li key={src.id} className="pt-2 first:pt-0">
                                  <div className="flex items-baseline justify-between text-[#1C1C1A]">
                                    <div>
                                      <span className="text-[#2C3E50] mr-1">[{index + 1}]</span>
                                      <span>{src.source}</span>
                                      {src.section && (
                                        <span className="text-[#8B8578] italic ml-1">
                                          ({src.section})
                                        </span>
                                      )}
                                    </div>
                                    <span className="text-[11px] text-[#8B8578] font-mono shrink-0 ml-2">
                                      {Math.round(src.similarity * 100)}%
                                    </span>
                                  </div>
                                  <p className="mt-0.5 text-[#8B8578] text-[11px] italic">
                                    "{src.text_snippet}"
                                  </p>
                                </li>
                              ))}
                            </ol>
                          )}
                        </footer>
                      )}
                    </div>
                  )}
                </article>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input Area: styled with #E8E3D7 for visual depth and subtle thin-stroked arrow-right */}
      <footer className="border-t border-[#DDD6C8] bg-[#E8E3D7] px-4 sm:px-8 py-3.5 sm:py-4">
        <div className="max-w-[700px] mx-auto w-full">
          <form onSubmit={handleSubmit}>
            <div className="relative border-b border-[#C8BFAD] focus-within:border-[#2C3E50] transition-colors pb-1.5 flex items-end">
              <textarea
                ref={textareaRef}
                rows={1}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask…"
                disabled={streaming}
                className="w-full bg-transparent text-[#1C1C1A] placeholder-[#7D7669] font-serif text-[15px] sm:text-[16px] focus:outline-none resize-none leading-relaxed pr-10"
              />

              <div className="absolute right-0 bottom-1 flex items-center space-x-1">
                {streaming ? (
                  <button
                    type="button"
                    onClick={onStopStreaming}
                    className="p-1.5 rounded text-[#7A2E2E] hover:bg-[#DDD6C8]/40 transition-colors"
                    title="Stop"
                    aria-label="Stop"
                  >
                    <Square className="w-3.5 h-3.5 fill-current" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    className={`p-1.5 rounded transition-colors ${
                      input.trim()
                        ? 'text-[#2C3E50] hover:text-[#1C1C1A] hover:bg-[#DDD6C8]/50'
                        : 'text-[#DDD6C8] cursor-not-allowed'
                    }`}
                    title="Ask"
                    aria-label="Ask"
                  >
                    <ArrowRight className="w-4 h-4" strokeWidth={1.5} />
                  </button>
                )}
              </div>
            </div>

            {messages.length > 0 && (
              <div className="flex justify-end pt-1.5">
                <button
                  type="button"
                  onClick={onClearHistory}
                  className="text-[11px] text-[#7D7669] hover:text-[#7A2E2E]"
                >
                  Clear
                </button>
              </div>
            )}
          </form>
        </div>
      </footer>
    </main>
  );
};
