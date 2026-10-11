import express from 'express';
import dotenv from 'dotenv';
import multer from 'multer';
import { PDFParse } from 'pdf-parse';
import path from 'path';
import fs from 'fs';
import { GoogleGenAI } from '@google/genai';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT) : 3000;

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Configure multer for memory storage
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 }, // 25MB
});

// Configure Google GenAI client
const geminiApiKey = process.env.GEMINI_API_KEY || '';
const groqApiKey = process.env.GROQ_API_KEY || '';

let aiClient: GoogleGenAI | null = null;
if (geminiApiKey) {
  aiClient = new GoogleGenAI({
    apiKey: geminiApiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Current LLM provider setting
let currentProvider = (process.env.LLM_PROVIDER || 'gemini').toLowerCase();
const GEMINI_GEN_MODEL = process.env.GEMINI_GENERATION_MODEL || 'gemini-2.5-flash';
const GEMINI_EMBED_MODEL = process.env.GEMINI_EMBEDDING_MODEL || 'gemini-embedding-2-preview';
const GROQ_GEN_MODEL = process.env.GROQ_GENERATION_MODEL || 'openai/gpt-oss-120b';

// In-Memory Persistent Vector Store
interface DocumentChunk {
  id: string;
  text: string;
  metadata: {
    source: string;
    section: string;
    chunk_index: number;
    char_count: number;
    estimated_tokens: number;
    timestamp: number;
  };
  embedding: number[];
}

const STORAGE_FILE = path.join(__dirname, '.vector_store.json');
let vectorStore: DocumentChunk[] = [];

// Load persisted vectors if exists
function loadStoredVectors() {
  try {
    if (fs.existsSync(STORAGE_FILE)) {
      const data = fs.readFileSync(STORAGE_FILE, 'utf-8');
      vectorStore = JSON.parse(data);
      console.log(`[VectorStore] Loaded ${vectorStore.length} chunks from storage.`);
    }
  } catch (err) {
    console.error('[VectorStore] Error loading storage:', err);
  }
}

function saveStoredVectors() {
  try {
    fs.writeFileSync(STORAGE_FILE, JSON.stringify(vectorStore), 'utf-8');
  } catch (err) {
    console.error('[VectorStore] Error saving storage:', err);
  }
}

loadStoredVectors();

// Cosine similarity helper
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length || vecA.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Recursive text splitter implementation
function recursiveCharacterSplit(
  text: string,
  chunkSize: number = 2400,
  chunkOverlap: number = 450
): string[] {
  const separators = ['\n\n# ', '\n\n## ', '\n\n### ', '\n\n', '\n', '. ', '? ', '! ', '; ', ' ', ''];

  function splitRecursive(subText: string, seps: string[]): string[] {
    if (!subText) return [];
    if (subText.length <= chunkSize) return [subText];

    let chosenSep = '';
    let nextSeps: string[] = [];
    for (let i = 0; i < seps.length; i++) {
      if (seps[i] === '') {
        chosenSep = '';
        break;
      }
      if (subText.includes(seps[i])) {
        chosenSep = seps[i];
        nextSeps = seps.slice(i + 1);
        break;
      }
    }

    if (chosenSep === '') {
      const splits: string[] = [];
      for (let i = 0; i < subText.length; i += chunkSize - chunkOverlap) {
        splits.push(subText.substring(i, i + chunkSize));
      }
      return splits;
    }

    const pieces = subText.split(chosenSep);
    const result: string[] = [];
    for (let i = 0; i < pieces.length; i++) {
      const piece = i === 0 ? pieces[i] : (chosenSep.startsWith('\n\n#') ? chosenSep.replace(/^\n+/, '') + pieces[i] : pieces[i]);
      if (piece.length > chunkSize) {
        result.push(...splitRecursive(piece, nextSeps));
      } else if (piece.trim()) {
        result.push(piece);
      }
    }
    return result;
  }

  const rawSplits = splitRecursive(text, separators);

  // Merge splits with overlap
  const merged: string[] = [];
  let current: string[] = [];
  let currentLen = 0;

  for (const part of rawSplits) {
    if (currentLen + part.length > chunkSize && current.length > 0) {
      merged.push(current.join(' ').trim());
      // Keep overlap
      const overlap: string[] = [];
      let overlapLen = 0;
      for (let i = current.length - 1; i >= 0; i--) {
        if (overlapLen + current[i].length <= chunkOverlap) {
          overlap.unshift(current[i]);
          overlapLen += current[i].length;
        } else {
          break;
        }
      }
      current = overlap;
      currentLen = overlapLen;
    }
    current.push(part);
    currentLen += part.length;
  }

  if (current.length > 0) {
    const finalStr = current.join(' ').trim();
    if (finalStr) merged.push(finalStr);
  }

  return merged;
}

// Generate embedding for text
async function getEmbedding(text: string): Promise<number[]> {
  if (!aiClient) {
    throw new Error('GEMINI_API_KEY is not configured in environment.');
  }

  // gemini-embedding-2-preview is the supported modern embedding model in @google/genai
  const primaryModel = GEMINI_EMBED_MODEL || 'gemini-embedding-2-preview';
  try {
    const response = await aiClient.models.embedContent({
      model: primaryModel,
      contents: text,
    });
    const values = response.embeddings?.[0]?.values;
    if (values && values.length > 0) {
      return values;
    }
  } catch (err: any) {
    if (primaryModel !== 'gemini-embedding-2-preview') {
      try {
        const response = await aiClient.models.embedContent({
          model: 'gemini-embedding-2-preview',
          contents: text,
        });
        const values = response.embeddings?.[0]?.values;
        if (values && values.length > 0) {
          return values;
        }
      } catch (fbErr: any) {
        throw new Error(`Embedding API failed: ${fbErr.message}`);
      }
    } else {
      throw new Error(`Embedding API failed: ${err.message}`);
    }
  }
  return [];
}

// Grounded system prompt
const SYSTEM_PROMPT_TEMPLATE = `You are an authoritative, accurate, and helpful AI assistant answering questions based STRICTLY on the provided knowledge base context.

CRITICAL INSTRUCTIONS:
1. Grounded Answers: Use ONLY the information provided in the Context below. Do not extrapolate, speculate, or introduce external facts.
2. Explicit Unknowns: If the answer cannot be determined directly from the Context, you MUST clearly and explicitly state: "I could not find information about that in the uploaded knowledge base documents." Do not attempt to guess or hallucinate.
3. Citations: When making factual statements, cite the relevant document and section, e.g., [Source: filename.pdf, Section: Overview].
4. Tone & Formatting: Be structured, clear, and professional. Use markdown formatting (headers, bullet points, bold key terms, code blocks) where helpful.

================ CONTEXT ================
{context_text}
=========================================`;

// Rewrite query to standalone format
async function rewriteQuery(question: string, history: Array<{ role: string; content: string }>): Promise<string> {
  if (!history || history.length === 0) return question.trim();

  const historyText = history.slice(-4).map((m) => `${m.role.toUpperCase()}: ${m.content}`).join('\n');
  const prompt = `Given the conversation history and a user's follow-up question, rewrite the follow-up question into a standalone, concise search query for vector retrieval.\n- Resolve all pronouns (it, that, they, this).\n- Preserve key technical terminology.\n- Return ONLY the standalone question.\n\nConversation History:\n${historyText}\n\nLatest Question: ${question}\n\nStandalone Search Query:`;

  try {
    if (aiClient) {
      const response = await aiClient.models.generateContent({
        model: GEMINI_GEN_MODEL,
        contents: prompt,
        config: {
          temperature: 0.0,
          maxOutputTokens: 80,
          thinkingConfig: {
            thinkingBudget: 0,
          },
        },
      });
      const text = response.text?.trim();
      if (text) return text;
    }
  } catch (err) {
    console.warn('[QueryRewrite] Failed to rewrite query:', err);
  }
  return question.trim();
}

// ----------------- API ROUTES -----------------

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    chroma_compatible_store: true,
    total_chunks: vectorStore.length,
    active_provider: currentProvider,
    gemini_ready: !!process.env.GEMINI_API_KEY,
    groq_ready: !!process.env.GROQ_API_KEY,
  });
});

// Config info & toggle
app.get('/api/config', (req, res) => {
  res.json({
    current_provider: currentProvider,
    gemini_configured: !!process.env.GEMINI_API_KEY,
    groq_configured: !!process.env.GROQ_API_KEY,
    gemini_model: GEMINI_GEN_MODEL,
    groq_model: GROQ_GEN_MODEL,
    embedding_model: GEMINI_EMBED_MODEL,
    total_chunks: vectorStore.length,
    default_top_k: 4,
  });
});

app.post('/api/config', (req, res) => {
  const { provider } = req.body;
  if (!provider || !['gemini', 'groq'].includes(provider.toLowerCase())) {
    return res.status(400).json({ error: "Provider must be 'gemini' or 'groq'" });
  }
  currentProvider = provider.toLowerCase();
  res.json({ status: 'success', current_provider: currentProvider });
});

// List indexed documents
app.get('/api/documents', (req, res) => {
  const docsMap = new Map<string, {
    filename: string;
    chunk_count: number;
    total_chars: number;
    estimated_tokens: number;
    sections: Set<string>;
    timestamp: number;
  }>();

  for (const chunk of vectorStore) {
    const src = chunk.metadata.source || 'Unknown';
    if (!docsMap.has(src)) {
      docsMap.set(src, {
        filename: src,
        chunk_count: 0,
        total_chars: 0,
        estimated_tokens: 0,
        sections: new Set(),
        timestamp: chunk.metadata.timestamp || 0,
      });
    }
    const item = docsMap.get(src)!;
    item.chunk_count++;
    item.total_chars += chunk.metadata.char_count || 0;
    item.estimated_tokens += chunk.metadata.estimated_tokens || 0;
    if (chunk.metadata.section) item.sections.add(chunk.metadata.section);
    item.timestamp = Math.max(item.timestamp, chunk.metadata.timestamp || 0);
  }

  const documents = Array.from(docsMap.values()).map((d) => ({
    filename: d.filename,
    chunk_count: d.chunk_count,
    total_chars: d.total_chars,
    estimated_tokens: d.estimated_tokens,
    sections_count: d.sections.size,
    timestamp: d.timestamp,
  })).sort((a, b) => b.timestamp - a.timestamp);

  res.json({
    documents,
    total_documents: documents.length,
    total_chunks: vectorStore.length,
  });
});

// Delete document and all its chunks
app.delete('/api/documents/:filename(*)', (req, res) => {
  const target = req.params.filename;
  const initialCount = vectorStore.length;
  vectorStore = vectorStore.filter((c) => c.metadata.source !== target);
  const deletedCount = initialCount - vectorStore.length;

  if (deletedCount === 0) {
    return res.status(404).json({ error: `Document '${target}' not found.` });
  }

  saveStoredVectors();
  res.json({
    status: 'success',
    filename: target,
    deleted_chunks: deletedCount,
    remaining_chunks: vectorStore.length,
  });
});

// Upload and index document
app.post('/api/documents/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded.' });
    }

    const filename = req.file.originalname;
    const ext = path.extname(filename).toLowerCase().replace('.', '');
    const buffer = req.file.buffer;

    let textSections: Array<{ text: string; section: string }> = [];

    if (ext === 'pdf') {
      try {
        const parser = new PDFParse({ data: buffer });
        const textResult = await parser.getText();
        const fullText = textResult.text;
        await parser.destroy();
        if (!fullText || !fullText.trim()) {
          return res.status(400).json({ error: 'PDF file is empty or cannot be read.' });
        }
        textSections.push({ text: fullText.trim(), section: 'Document Content' });
      } catch (pdfErr: any) {
        return res.status(400).json({ error: `Failed to parse PDF: ${pdfErr.message}` });
      }
    } else {
      // Text, Markdown, code files
      const text = buffer.toString('utf-8');
      if (!text.trim()) {
        return res.status(400).json({ error: 'Uploaded file is empty.' });
      }

      // Check for Markdown headers to extract sections
      const headerRegex = /^(#{1,4}\s+[^\n]+)/gm;
      const matches = Array.from(text.matchAll(headerRegex));

      if (matches.length > 0) {
        let lastIdx = 0;
        let currentSection = 'Overview';
        for (const m of matches) {
          const secText = text.substring(lastIdx, m.index).trim();
          if (secText) {
            textSections.push({ text: secText, section: currentSection });
          }
          currentSection = m[0].replace(/^#+\s*/, '').trim();
          lastIdx = m.index! + m[0].length;
        }
        const remaining = text.substring(lastIdx).trim();
        if (remaining) {
          textSections.push({ text: remaining, section: currentSection });
        }
      } else {
        textSections.push({ text: text.trim(), section: 'General' });
      }
    }

    // Split each section
    const newChunks: DocumentChunk[] = [];
    const now = Math.floor(Date.now() / 1000);

    for (const sec of textSections) {
      const splits = recursiveCharacterSplit(sec.text, 2400, 450);
      for (let i = 0; i < splits.length; i++) {
        const chunkText = splits[i];
        if (!chunkText.trim()) continue;

        // Generate embedding
        let embedding: number[] = [];
        try {
          embedding = await getEmbedding(chunkText);
        } catch (embedErr: any) {
          console.error(`[Embedding] Failed chunk embed:`, embedErr);
          throw new Error(`Failed to generate embedding: ${embedErr.message}`);
        }

        const chunkId = `${filename.replace(/[^a-zA-Z0-9]/g, '_')}_${sec.section.replace(/[^a-zA-Z0-9]/g, '_')}_${i}_${Date.now()}`;
        newChunks.push({
          id: chunkId,
          text: chunkText,
          metadata: {
            source: filename,
            section: sec.section,
            chunk_index: newChunks.length,
            char_count: chunkText.length,
            estimated_tokens: Math.max(1, Math.floor(chunkText.length / 4)),
            timestamp: now,
          },
          embedding,
        });
      }
    }

    if (newChunks.length === 0) {
      return res.status(400).json({ error: 'Could not extract valid text chunks from document.' });
    }

    // Upsert into vectorStore
    // Remove previous chunks for the same filename if any
    vectorStore = vectorStore.filter((c) => c.metadata.source !== filename);
    vectorStore.push(...newChunks);
    saveStoredVectors();

    res.json({
      status: 'success',
      filename,
      chunks_created: newChunks.length,
      estimated_tokens: newChunks.reduce((acc, c) => acc + c.metadata.estimated_tokens, 0),
      message: `Successfully indexed '${filename}' into vector store.`,
    });
  } catch (err: any) {
    console.error('[Upload] Error:', err);
    res.status(500).json({ error: err.message || 'Indexing failed.' });
  }
});

// Seed sample knowledge base
app.post('/api/seed-sample', async (req, res) => {
  try {
    const { sampleKey } = req.body || {};
    
    // Curated high quality knowledge base documents
    const samplePacks: Record<string, { filename: string; content: string }> = {
      cloud_architecture: {
        filename: 'Acme_Cloud_ZeroTrust_Architecture.md',
        content: `# Acme Cloud Platform: Zero-Trust Microservices & Security Whitepaper

## Architecture Overview
Acme Cloud platform uses a zero-trust microservices architecture deployed across multi-region Google Cloud Kubernetes clusters (GKE).
All intra-service communication is encrypted with mutual TLS (mTLS) using SPIFFE/SPIRE cryptographic identity tokens.
Traffic enters through Envoy ingress gateways with distributed rate limiting and automated DDoS mitigation.

## Authentication & Authorization
User authentication is managed via OAuth 2.0 and OpenID Connect (OIDC) backed by Okta and Google Workspace.
Access tokens are short-lived JSON Web Tokens (JWTs) with an expiration time of 15 minutes.
Fine-grained authorization uses Attribute-Based Access Control (ABAC) enforced by Open Policy Agent (OPA).
Service accounts must rotate credentials every 30 days via HashiCorp Vault automated rotation engines.

## Data Retention and Cryptographic Compliance
Customer data at rest is encrypted with AES-256-GCM using customer-managed encryption keys (CMEK) stored in GCP Cloud KMS.
Backups are performed every 4 hours with point-in-time recovery (PITR) retained for 35 days.
Deleted accounts enter a soft-delete grace period of 30 days before permanent cryptographic erasure.
Acme complies with SOC 2 Type II, ISO 27001, and HIPAA Security Rule requirements.

## Incident Response SLAs
Security incidents are classified into 4 severity levels:
- SEV-0 (Critical breach or data exposure): Initial response within 15 minutes, customer notification within 2 hours.
- SEV-1 (High risk outage or privilege escalation): Initial response within 30 minutes.
- SEV-2 (Moderate issue without data compromise): Initial response within 4 hours.
- SEV-3 (Low risk or informational bug): Initial response within 24 hours.

## High Availability & Disaster Recovery
The platform maintains an RPO (Recovery Point Objective) of 5 minutes and an RTO (Recovery Time Objective) of 30 minutes.
Global DNS routing uses Cloudflare with automatic failover between primary (us-central1) and secondary (europe-west1) regions.`,
      },
      api_handbook: {
        filename: 'Acme_API_Gateway_Engineering_Handbook.md',
        content: `# Acme Developer Platform: API Gateway & Integration Guide

## Rate Limiting Algorithms
The Acme API Gateway enforces a Token Bucket rate limiting algorithm backed by Redis clusters.
- Public Tier: 60 requests per minute per IP address.
- Authenticated Tier: 1,000 requests per minute per API key.
- Enterprise Tier: 10,000 requests per minute with dedicated bursting capacity up to 15,000 req/min for 30 seconds.
When limits are exceeded, the gateway returns HTTP 429 Too Many Requests with headers 'Retry-After' and 'X-RateLimit-Reset'.

## Webhooks & Event Delivery
Webhooks are dispatched with an HMAC-SHA256 signature calculated over the payload using your webhook secret in the 'X-Acme-Signature' header.
Webhook delivery attempts follow an exponential backoff schedule:
- 1st retry: after 10 seconds
- 2nd retry: after 1 minute
- 3rd retry: after 15 minutes
- 4th retry: after 2 hours
Events are marked failed and moved to the Dead Letter Queue (DLQ) after 5 consecutive delivery failures.

## API Versioning Policy
All endpoints follow semantic URL versioning, e.g. '/v1/customers' and '/v2/billing'.
Endpoints are supported for a minimum of 18 months following deprecation notice.
Deprecation announcements are delivered via response headers ('Sunset' and 'Deprecation').`,
      },
    };

    const targetPack = samplePacks[sampleKey] || samplePacks.cloud_architecture;
    const filename = targetPack.filename;
    const content = targetPack.content;

    // Check if already seeded
    const exists = vectorStore.some((c) => c.metadata.source === filename);
    if (exists) {
      return res.json({
        status: 'success',
        filename,
        message: `'${filename}' is already indexed in the vector store.`,
        chunks_count: vectorStore.filter((c) => c.metadata.source === filename).length,
      });
    }

    const splits = recursiveCharacterSplit(content, 1800, 300);
    const newChunks: DocumentChunk[] = [];
    const now = Math.floor(Date.now() / 1000);

    for (let i = 0; i < splits.length; i++) {
      const chunkText = splits[i];
      const embedding = await getEmbedding(chunkText);
      const chunkId = `${filename.replace(/[^a-zA-Z0-9]/g, '_')}_chunk_${i}_${Date.now()}`;
      
      // Determine section from text
      const firstLine = chunkText.split('\n')[0] || '';
      const section = firstLine.startsWith('#') ? firstLine.replace(/^#+\s*/, '') : 'Whitepaper Section';

      newChunks.push({
        id: chunkId,
        text: chunkText,
        metadata: {
          source: filename,
          section,
          chunk_index: i,
          char_count: chunkText.length,
          estimated_tokens: Math.max(1, Math.floor(chunkText.length / 4)),
          timestamp: now,
        },
        embedding,
      });
    }

    vectorStore.push(...newChunks);
    saveStoredVectors();

    res.json({
      status: 'success',
      filename,
      chunks_created: newChunks.length,
      message: `Sample knowledge base '${filename}' indexed successfully!`,
    });
  } catch (err: any) {
    console.error('[SeedSample] Error:', err);
    res.status(500).json({ error: err.message || 'Failed to seed sample knowledge base.' });
  }
});

// Streaming RAG Chat Endpoint
app.post('/api/chat/stream', async (req, res) => {
  const { question, history = [], provider, top_k = 4 } = req.body;

  if (!question || typeof question !== 'string' || !question.trim()) {
    return res.status(400).json({ error: 'Question is required.' });
  }

  // Set SSE headers
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  const sendEvent = (event: string, data: any) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    // 1. Query rewriting if history exists
    const standaloneQuery = await rewriteQuery(question, history);

    // 2. Similarity search in vectorStore
    let retrievedChunks: Array<{ text: string; metadata: any; similarity: number }> = [];

    if (vectorStore.length > 0) {
      const queryEmbedding = await getEmbedding(standaloneQuery);
      const scored = vectorStore.map((chunk) => {
        const sim = cosineSimilarity(queryEmbedding, chunk.embedding);
        return {
          text: chunk.text,
          metadata: chunk.metadata,
          similarity: Math.round(sim * 1000) / 1000,
        };
      });

      scored.sort((a, b) => b.similarity - a.similarity);
      retrievedChunks = scored.slice(0, Math.min(top_k, scored.length));
    }

    // 3. Emit sources event immediately
    const sourcesPayload = retrievedChunks.map((c, idx) => ({
      id: idx + 1,
      source: c.metadata.source || 'Unknown',
      section: c.metadata.section || 'General',
      similarity: c.similarity,
      text_snippet: c.text.length > 280 ? c.text.substring(0, 280) + '...' : c.text,
      full_text: c.text,
    }));

    sendEvent('sources', sourcesPayload);

    // 4. Build grounded context
    let contextText = '';
    if (retrievedChunks.length === 0) {
      contextText = 'NO RELEVANT DOCUMENTS FOUND IN KNOWLEDGE BASE.';
    } else {
      contextText = retrievedChunks
        .map(
          (c, idx) =>
            `[DOCUMENT CHUNK ${idx + 1}]\nSource: ${c.metadata.source}\nSection: ${c.metadata.section}\nRelevance Score: ${c.similarity}\nContent:\n${c.text}\n`
        )
        .join('\n-----------------------------------------\n');
    }

    const systemPrompt = SYSTEM_PROMPT_TEMPLATE.replace('{context_text}', contextText);
    const activeProvider = (provider || currentProvider).toLowerCase();

    // 5. Stream LLM generation with fallback
    let streamSuccess = false;

    if (activeProvider === 'gemini') {
      try {
        await streamGemini(question, systemPrompt, history, (token) => {
          sendEvent('token', { token });
        });
        streamSuccess = true;
      } catch (geminiErr: any) {
        console.error('[GeminiStream] Primary failed:', geminiErr.message);
        // Fallback to Groq if key is present
        if (process.env.GROQ_API_KEY) {
          sendEvent('token', { token: '\n\n*(Gemini rate limit or error encountered. Switched to Groq fallback...)*\n\n' });
          await streamGroq(question, systemPrompt, history, (token) => {
            sendEvent('token', { token });
          });
          streamSuccess = true;
        } else {
          throw geminiErr;
        }
      }
    } else {
      // Groq requested
      try {
        if (!process.env.GROQ_API_KEY) {
          throw new Error('GROQ_API_KEY is not configured in environment.');
        }
        await streamGroq(question, systemPrompt, history, (token) => {
          sendEvent('token', { token });
        });
        streamSuccess = true;
      } catch (groqErr: any) {
        console.error('[GroqStream] Primary failed:', groqErr.message);
        if (process.env.GEMINI_API_KEY) {
          sendEvent('token', { token: '\n\n*(Groq failed. Switched to Gemini fallback...)*\n\n' });
          await streamGemini(question, systemPrompt, history, (token) => {
            sendEvent('token', { token });
          });
          streamSuccess = true;
        } else {
          throw groqErr;
        }
      }
    }

    sendEvent('done', {});
    res.end();
  } catch (err: any) {
    console.error('[ChatStream] Fatal Error:', err);
    sendEvent('error', { error: err.message || 'Generation failed.' });
    sendEvent('done', {});
    res.end();
  }
});

// Helper for Gemini stream
async function streamGemini(
  question: string,
  systemPrompt: string,
  history: Array<{ role: string; content: string }>,
  onToken: (t: string) => void
) {
  if (!aiClient) throw new Error('GEMINI_API_KEY is not configured.');

  const candidateModels = Array.from(new Set([GEMINI_GEN_MODEL, 'gemini-2.5-flash', 'gemini-flash-latest']));
  let lastError: Error | null = null;

  for (const model of candidateModels) {
    try {
      const contents: any[] = [];
      if (history && history.length > 0) {
        for (const msg of history.slice(-4)) {
          contents.push({
            role: msg.role === 'assistant' ? 'model' : 'user',
            parts: [{ text: msg.content }],
          });
        }
      }
      contents.push({ role: 'user', parts: [{ text: question }] });

      const streamResponse = await aiClient.models.generateContentStream({
        model,
        contents,
        config: {
          systemInstruction: systemPrompt,
          temperature: 0.2,
          thinkingConfig: {
            thinkingBudget: 0,
          },
        },
      });

      for await (const chunk of streamResponse) {
        if (chunk.text) {
          onToken(chunk.text);
        }
      }
      return; // Succeeded!
    } catch (err: any) {
      console.warn(`[GeminiStream] Model ${model} error: ${err.message}. Checking next candidate...`);
      lastError = err;
    }
  }

  if (lastError) throw lastError;
}

// Helper for Groq stream
async function streamGroq(
  question: string,
  systemPrompt: string,
  history: Array<{ role: string; content: string }>,
  onToken: (t: string) => void
) {
  const key = process.env.GROQ_API_KEY;
  if (!key) throw new Error('GROQ_API_KEY is not configured.');

  const candidateModels = Array.from(
    new Set([
      GROQ_GEN_MODEL,
      'openai/gpt-oss-120b',
      'qwen/qwen3.8-27b',
      'openai/gpt-oss-20b',
      'llama-3.3-70b-versatile',
      'llama-3.1-70b-versatile',
    ])
  );

  const messages: any[] = [{ role: 'system', content: systemPrompt }];
  if (history && history.length > 0) {
    for (const msg of history.slice(-4)) {
      messages.push({ role: msg.role, content: msg.content });
    }
  }
  messages.push({ role: 'user', content: question });

  let lastError: Error | null = null;

  for (const model of candidateModels) {
    try {
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${key}`,
        },
        body: JSON.stringify({
          model,
          messages,
          temperature: 0.2,
          stream: true,
        }),
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Groq stream error (${response.status}): ${errText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('Cannot get reader for Groq stream.');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.substring(6).trim();
            if (!jsonStr || jsonStr === '[DONE]') continue;
            try {
              const parsed = JSON.parse(jsonStr);
              const delta = parsed.choices?.[0]?.delta?.content;
              if (delta) onToken(delta);
            } catch {
              // ignore
            }
          }
        }
      }
      return; // Succeeded!
    } catch (err: any) {
      console.warn(`[GroqStream] Model ${model} failed: ${err.message}. Trying next model...`);
      lastError = err;
    }
  }

  if (lastError) throw lastError;
}

// Vite integration for development or static serving
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[RAG Server] Running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
