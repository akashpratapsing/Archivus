import os
import logging
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# Setup structured logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("rag_backend")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Generation & Embedding Settings
DEFAULT_LLM_PROVIDER: Literal["gemini", "groq"] = os.getenv("LLM_PROVIDER", "gemini").lower() # type: ignore
GEMINI_GENERATION_MODEL = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.8-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
GROQ_GENERATION_MODEL = os.getenv("GROQ_GENERATION_MODEL", "llama-3.3-70b-versatile")

# Chunking Settings (~800 tokens ≈ 3200 characters, ~150 tokens overlap ≈ 600 characters)
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "2400"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "450"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))

# Chroma Settings
CHROMA_HOST = os.getenv("CHROMA_HOST", "")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
COLLECTION_NAME = "knowledge_base"
