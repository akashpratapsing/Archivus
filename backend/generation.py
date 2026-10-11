import json
import os
from typing import List, Dict, Any, AsyncGenerator, Optional
import httpx
from config import (
    logger,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    DEFAULT_LLM_PROVIDER,
    GEMINI_GENERATION_MODEL,
    GROQ_GENERATION_MODEL,
)

SYSTEM_PROMPT_TEMPLATE = """You are an authoritative, accurate, and helpful AI assistant answering questions based STRICTLY on the provided knowledge base context.

CRITICAL INSTRUCTIONS:
1. Grounded Answers: Use ONLY the information provided in the Context below. Do not assume or extrapolate facts not present in the context.
2. Explicit Unknowns: If the answer cannot be determined directly from the Context, you MUST clearly and explicitly state: "I could not find information about that in the uploaded knowledge base documents." Do not attempt to guess or hallucinate.
3. Citations: When you provide factual claims, cite the relevant source document and section using brackets, like [Source: filename.pdf, Section: Page 2] or [Source: README.md, Section: Architecture].
4. Tone & Formatting: Be structured, clear, and professional. Use markdown formatting (headers, bullet points, bold key terms, code blocks) where helpful.

================ CONTEXT ================
{context_text}
=========================================
"""

QUERY_REWRITE_SYSTEM_PROMPT = """You are a search query reformulation assistant. Given a conversation history and a user's latest follow-up question, rewrite the follow-up question into a standalone, concise search query suitable for semantic vector retrieval.
- Preserve all key technical terms, names, and entities.
- Resolve pronouns (it, they, that, this) to their explicit referents from previous turns.
- Do NOT answer the question. Output ONLY the reformulated query text.
- If the question is already standalone, return it unchanged.
"""

class RAGGenerator:
    def __init__(self):
        self.provider = DEFAULT_LLM_PROVIDER

    def set_provider(self, provider: str):
        p = provider.lower().strip()
        if p in ["gemini", "groq"]:
            self.provider = p
            logger.info(f"LLM Provider switched to: {self.provider}")
        else:
            raise ValueError(f"Unsupported provider '{provider}'. Must be 'gemini' or 'groq'.")

    async def rewrite_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Rewrites conversational follow-up questions into standalone vector queries.
        """
        if not history:
            return query.strip()

        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-4:]])
        prompt = f"Conversation History:\n{history_text}\n\nLatest Question: {query}\n\nStandalone Search Query:"

        try:
            if self.provider == "gemini" and GEMINI_API_KEY:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GENERATION_MODEL}:generateContent?key={GEMINI_API_KEY}"
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "systemInstruction": {"parts": [{"text": QUERY_REWRITE_SYSTEM_PROMPT}]},
                        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 100}
                    }
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text.strip():
                            logger.info(f"Query rewritten: '{query}' -> '{text.strip()}'")
                            return text.strip()
            elif GROQ_API_KEY:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                        json={
                            "model": GROQ_GENERATION_MODEL,
                            "messages": [
                                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": 0.0,
                            "max_tokens": 100,
                        }
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data["choices"][0]["message"]["content"]
                        if text.strip():
                            logger.info(f"Query rewritten (Groq): '{query}' -> '{text.strip()}'")
                            return text.strip()
        except Exception as e:
            logger.warning(f"Query rewrite failed ({e}), using original query.")
        
        return query.strip()

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "NO RELEVANT DOCUMENTS FOUND IN KNOWLEDGE BASE."

        formatted_parts = []
        for idx, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Unknown Document")
            section = meta.get("section", "General")
            text = chunk.get("text", "").strip()
            sim = chunk.get("similarity", 0.0)
            formatted_parts.append(
                f"[DOCUMENT CHUNK {idx + 1}]\n"
                f"Source: {source}\n"
                f"Section: {section}\n"
                f"Relevance Score: {sim}\n"
                f"Content:\n{text}\n"
            )
        return "\n-----------------------------------------\n".join(formatted_parts)

    async def stream_rag_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]] = None,
        provider_override: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Streams generated RAG response token by token formatted as Server-Sent Events (SSE).
        """
        active_provider = (provider_override or self.provider).lower()
        context_str = self.format_context(chunks)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context_text=context_str)

        # First yield the retrieved source metadata event so frontend displays citation sources immediately
        sources_payload = [
            {
                "id": idx + 1,
                "source": c.get("metadata", {}).get("source", "Unknown"),
                "section": c.get("metadata", {}).get("section", "General"),
                "similarity": c.get("similarity", 0.0),
                "text_snippet": c.get("text", "")[:280] + ("..." if len(c.get("text", "")) > 280 else ""),
            }
            for idx, c in enumerate(chunks)
        ]
        yield f"event: sources\ndata: {json.dumps(sources_payload)}\n\n"

        # Stream generation with fallback mechanism
        try:
            if active_provider == "gemini":
                if not GEMINI_API_KEY:
                    raise ValueError("GEMINI_API_KEY is not set.")
                async for token in self._stream_gemini(question, system_prompt, history):
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
            else:
                if not GROQ_API_KEY:
                    raise ValueError("GROQ_API_KEY is not set.")
                async for token in self._stream_groq(question, system_prompt, history):
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
        except Exception as primary_error:
            logger.error(f"Primary provider '{active_provider}' error: {primary_error}")
            
            # Automatic fallback to alternate provider if key is available
            fallback_provider = "groq" if active_provider == "gemini" else "gemini"
            fallback_key = GROQ_API_KEY if fallback_provider == "groq" else GEMINI_API_KEY
            
            if fallback_key:
                logger.info(f"Attempting fallback to '{fallback_provider}'...")
                yield f"event: token\ndata: {json.dumps({'token': f'*(Switched to {fallback_provider.title()} fallback provider)*\n\n'})}\n\n"
                try:
                    if fallback_provider == "gemini":
                        async for token in self._stream_gemini(question, system_prompt, history):
                            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                    else:
                        async for token in self._stream_groq(question, system_prompt, history):
                            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                except Exception as fallback_error:
                    logger.error(f"Fallback '{fallback_provider}' also failed: {fallback_error}")
                    yield f"event: error\ndata: {json.dumps({'error': f'Generation error: {str(primary_error)}'})}\n\n"
            else:
                yield f"event: error\ndata: {json.dumps({'error': f'Generation failed: {str(primary_error)}'})}\n\n"

        yield "event: done\ndata: {}\n\n"

    async def _stream_gemini(
        self, question: str, system_prompt: str, history: Optional[List[Dict[str, str]]]
    ) -> AsyncGenerator[str, None]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_GENERATION_MODEL}:streamGenerateContent?alt=sse&key={GEMINI_API_KEY}"
        
        contents = []
        if history:
            for msg in history[-4:]:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": question}]})

        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.95,
            }
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    raise RuntimeError(f"Gemini API error ({resp.status_code}): {error_text.decode('utf-8', errors='ignore')}")

                buffer = ""
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        json_str = line[6:].strip()
                        if not json_str or json_str == "[DONE]":
                            continue
                        try:
                            parsed = json.loads(json_str)
                            candidates = parsed.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for p in parts:
                                    t = p.get("text", "")
                                    if t:
                                        yield t
                        except Exception as e:
                            logger.debug(f"Gemini SSE parse warning: {e}")

    async def _stream_groq(
        self, question: str, system_prompt: str, history: Optional[List[Dict[str, str]]]
    ) -> AsyncGenerator[str, None]:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        payload = {
            "model": GROQ_GENERATION_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            async with client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    raise RuntimeError(f"Groq API error ({resp.status_code}): {error_text.decode('utf-8', errors='ignore')}")

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        json_str = line[6:].strip()
                        if not json_str or json_str == "[DONE]":
                            continue
                        try:
                            parsed = json.loads(json_str)
                            delta = parsed["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except Exception as e:
                            logger.debug(f"Groq SSE parse warning: {e}")

# Global generator instance
rag_generator = RAGGenerator()
