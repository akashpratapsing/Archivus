export interface IndexedDocument {
  filename: string;
  chunk_count: number;
  total_chars: number;
  estimated_tokens: number;
  sections_count: number;
  timestamp: number;
}

export interface SourceCitation {
  id: number;
  source: string;
  section: string;
  similarity: number;
  text_snippet: string;
  full_text?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceCitation[];
  isStreaming?: boolean;
  timestamp: number;
  provider?: string;
}

export interface SystemConfig {
  current_provider: 'gemini' | 'groq';
  gemini_configured: boolean;
  groq_configured: boolean;
  gemini_model: string;
  groq_model: string;
  embedding_model: string;
  total_chunks: number;
  default_top_k: number;
}
