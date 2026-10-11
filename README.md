# KnowledgePulse RAG Chatbot

An enterprise-ready Retrieval-Augmented Generation (RAG) chatbot over custom knowledge bases (company documents, codebase documentation, legal/medical/internal wikis).

## Architecture & Tech Stack

```
[ Documents (PDF, TXT, MD, Code) ]
               │
               ▼
   [ Text Extraction & Chunker ] ─── Recursive Splitting (~800 tokens, ~150 overlap)
               │
               ▼
   [ Gemini Embedding Engine ]  ─── text-embedding-004 (or gemini-embedding-2-preview)
               │
               ▼
    [ Chroma Vector Store ]     ─── Persistent Local / Docker Vector DB
               ▲
               │ Semantic Search (Cosine Similarity Top-k)
               │
   [ Context + Rewritten Query ]
               │
               ▼
    [ Swappable LLM Engine ]    ─── Primary: Gemini 2.0 Flash
                                ─── Fallback/Alternate: Groq (Llama 3.3 70B)
               │
               ▼
    [ SSE Token Streaming ]     ─── Real-time token-by-token answer + collapsible citations
```

- **Backend**: Python 3.11 with FastAPI (`/backend`)
  - `chunking.py`: Hierarchical recursive character text splitting with section detection.
  - `embedding.py`: Gemini Embedding API with batching, retries, and backoff.
  - `retrieval.py`: Chroma vector database client for persistent indexing and similarity search.
  - `generation.py`: Query reformulation, grounded system prompt, and token-by-token streaming (Gemini & Groq with automatic fallback).
  - `main.py`: FastAPI routes for document upload, listing, deletion, config toggle, and streaming chat.
- **Vector DB**: Chroma (containerized with persistent volume).
- **Frontend**: React + TypeScript + Vite with Tailwind CSS.
  - Token-by-token streaming responses.
  - Collapsible source citations showing source document, section, relevance score, and content snippet.
  - Document management sidebar with file upload (drag & drop), delete option, and pre-loaded sample domain packs.
  - Provider switcher (Gemini 2.0 Flash ⟷ Groq Llama 3.3 70B).
- **Orchestration**: Docker & `docker-compose.yml`.

---

## Quick Start with Docker Compose

### 1. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your API keys:
```env
GEMINI_API_KEY="your_gemini_api_key_here"
GROQ_API_KEY="your_optional_groq_api_key_here"
LLM_PROVIDER="gemini"  # or "groq"
```

### 2. Build and Run All Services
```bash
docker-compose up --build
```

This starts:
- **Chroma Vector DB** on `http://localhost:8002` (internal `chroma:8000`)
- **FastAPI Backend** on `http://localhost:8000` (interactive Swagger docs at `http://localhost:8000/docs`)
- **React Frontend** on `http://localhost:3000`

Visit `http://localhost:3000` in your web browser.

---

## Local Development (Without Docker)

### Backend (FastAPI):
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example .env   # Update with your API keys
uvicorn main:app --reload --port 8000
```

### Frontend (React):
```bash
npm install
npm run dev
```

---

## How to Index Documents and Chat

1. **Upload Documents**:
   - Drag & drop or browse files in the left sidebar.
   - Supported formats: **PDF, Markdown (`.md`), Plain Text (`.txt`), Code files (`.py`, `.ts`, `.js`, `.json`, `.sql`, `.html`)**.
   - The backend splits each document into semantic chunks (~800 tokens), embeds them using Gemini's `text-embedding-004`, and indexes them into Chroma.
2. **Instant Sample Packs**:
   - Click "Load Sample Knowledge Base" to preload curated corporate architecture and compliance docs to test immediately.
3. **Ask Questions**:
   - Enter your prompt into the chat box.
   - The query rewriter resolves pronouns using conversational context.
   - Chroma retrieves top-k chunks by cosine similarity.
   - The LLM streams an answer grounded strictly in the retrieved text.
4. **Inspect Source Citations**:
   - Every response features a collapsible "Sources & Citations" drawer showing the exact chunk text, section, file name, and similarity score.
5. **Manage Documents**:
   - Delete any document from the sidebar to purge its associated vectors from Chroma.

---

## Running Retrieval Verification Test

To verify chunking, embedding, indexing, and retrieval accuracy without the frontend:
```bash
cd backend
python test_retrieval.py
```
This script indexes an Acme Cloud Security whitepaper and tests semantic retrieval against 4 representative queries (including out-of-domain rejection test).
