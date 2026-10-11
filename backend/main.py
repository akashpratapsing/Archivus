import os
import time
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import (
    logger,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    DEFAULT_LLM_PROVIDER,
    GEMINI_GENERATION_MODEL,
    GROQ_GENERATION_MODEL,
    GEMINI_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
)
from chunking import process_and_chunk_document
from retrieval import vector_store
from generation import rag_generator

app = FastAPI(
    title="KnowledgePulse RAG API",
    description="Production-grade RAG backend with FastAPI, Chroma, Gemini, and Groq fallback.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str = Field(..., description="The user query or question")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation messages")
    provider: Optional[str] = Field(default=None, description="Optional override: 'gemini' or 'groq'")
    top_k: Optional[int] = Field(default=DEFAULT_TOP_K, description="Number of chunks to retrieve")

class ConfigUpdateRequest(BaseModel):
    provider: str

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "chroma_connected": True,
        "indexed_chunks": vector_store.collection.count(),
        "timestamp": int(time.time()),
    }

@app.get("/api/config")
async def get_config():
    return {
        "current_provider": rag_generator.provider,
        "gemini_configured": bool(GEMINI_API_KEY),
        "groq_configured": bool(GROQ_API_KEY),
        "gemini_model": GEMINI_GENERATION_MODEL,
        "groq_model": GROQ_GENERATION_MODEL,
        "embedding_model": GEMINI_EMBEDDING_MODEL,
        "total_chunks": vector_store.collection.count(),
        "default_top_k": DEFAULT_TOP_K,
    }

@app.post("/api/config")
async def update_config(config: ConfigUpdateRequest):
    try:
        rag_generator.set_provider(config.provider)
        return {"status": "success", "provider": rag_generator.provider}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/documents")
async def list_documents():
    docs = vector_store.list_documents()
    return {
        "documents": docs,
        "total_documents": len(docs),
        "total_chunks": vector_store.collection.count(),
    }

@app.delete("/api/documents/{document_name:path}")
async def delete_document(document_name: str):
    try:
        deleted_count = vector_store.delete_document(document_name)
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found.")
        return {"status": "success", "deleted_chunks": deleted_count, "filename": document_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
):
    """
    Upload and index a document (PDF, TXT, MD, code files).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file missing filename.")

    allowed_extensions = {
        "pdf", "txt", "md", "markdown", "py", "ts", "tsx", "js", "jsx",
        "json", "csv", "html", "css", "yaml", "yml", "sql", "sh"
    }
    ext = file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Supported types: {', '.join(sorted(allowed_extensions))}"
        )

    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Chunk document
        chunks = process_and_chunk_document(
            filename=file.filename,
            content_bytes=content_bytes,
            chunk_size=chunk_size or 2400,
            chunk_overlap=chunk_overlap or 450,
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract any valid text chunks from document.")

        # Upsert into Chroma
        upserted_count = vector_store.upsert_chunks(chunks)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": upserted_count,
            "estimated_tokens": sum(c["metadata"].get("estimated_tokens", 0) for c in chunks),
            "message": f"Successfully indexed '{file.filename}' into vector database."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload and indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Full RAG conversational endpoint with streaming Server-Sent Events.
    1. Rewrites conversational query if history exists.
    2. Retrieves top-k chunks from Chroma via similarity search.
    3. Streams grounded answer from Gemini / Groq with source citations.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Convert pydantic history models to dicts
    history_dicts = [{"role": m.role, "content": m.content} for m in (request.history or [])]

    # Step 1: Query rewriting
    search_query = await rag_generator.rewrite_query(question, history_dicts)

    # Step 2: Vector retrieval
    retrieved_chunks = vector_store.search_similar_chunks(search_query, top_k=request.top_k or DEFAULT_TOP_K)

    # Step 3: Stream generation
    async def event_generator():
        async for sse_chunk in rag_generator.stream_rag_answer(
            question=question,
            chunks=retrieved_chunks,
            history=history_dicts,
            provider_override=request.provider,
        ):
            yield sse_chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
