import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from config import (
    logger,
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_PERSIST_DIRECTORY,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
)
from embedding import embedding_service

class ChromaVectorStore:
    def __init__(self):
        self._client = None
        self._collection = None
        self._init_db()

    def _init_db(self):
        try:
            if CHROMA_HOST:
                logger.info(f"Connecting to remote Chroma server at {CHROMA_HOST}:{CHROMA_PORT}")
                self._client = chromadb.HttpClient(
                    host=CHROMA_HOST,
                    port=CHROMA_PORT,
                    settings=Settings(allow_reset=True, anonymized_telemetry=False)
                )
            else:
                os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
                logger.info(f"Using local persistent Chroma at {CHROMA_PERSIST_DIRECTORY}")
                self._client = chromadb.PersistentClient(
                    path=CHROMA_PERSIST_DIRECTORY,
                    settings=Settings(allow_reset=True, anonymized_telemetry=False)
                )
            
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Chroma collection '{COLLECTION_NAME}' initialized. Item count: {self._collection.count()}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
            raise

    @property
    def collection(self):
        if self._collection is None:
            self._init_db()
        return self._collection

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = embedding_service.get_embeddings_batch(texts)

        logger.info(f"Upserting {len(chunks)} chunks to Chroma vector database...")
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Aggregates chunks into distinct document summaries.
        """
        try:
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas") or []
            
            docs_map: Dict[str, Dict[str, Any]] = {}
            for meta in metadatas:
                source = meta.get("source", "unknown")
                if source not in docs_map:
                    docs_map[source] = {
                        "filename": source,
                        "chunk_count": 0,
                        "total_chars": 0,
                        "estimated_tokens": 0,
                        "sections": set(),
                        "timestamp": meta.get("timestamp", 0),
                    }
                docs_map[source]["chunk_count"] += 1
                docs_map[source]["total_chars"] += meta.get("char_count", 0)
                docs_map[source]["estimated_tokens"] += meta.get("estimated_tokens", 0)
                if meta.get("section"):
                    docs_map[source]["sections"].add(meta["section"])
                docs_map[source]["timestamp"] = max(docs_map[source]["timestamp"], meta.get("timestamp", 0))

            output = []
            for doc in docs_map.values():
                output.append({
                    "filename": doc["filename"],
                    "chunk_count": doc["chunk_count"],
                    "total_chars": doc["total_chars"],
                    "estimated_tokens": doc["estimated_tokens"],
                    "sections_count": len(doc["sections"]),
                    "timestamp": doc["timestamp"],
                })
            output.sort(key=lambda x: x["timestamp"], reverse=True)
            return output
        except Exception as e:
            logger.error(f"Error listing documents from Chroma: {e}")
            return []

    def delete_document(self, filename: str) -> int:
        """
        Deletes all chunks associated with a source filename.
        """
        try:
            logger.info(f"Deleting document '{filename}' from Chroma...")
            # Query IDs first
            existing = self.collection.get(where={"source": filename})
            ids_to_delete = existing.get("ids", [])
            
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for '{filename}'.")
                return len(ids_to_delete)
            return 0
        except Exception as e:
            logger.error(f"Error deleting document '{filename}': {e}")
            raise

    def search_similar_chunks(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """
        Embeds the query and retrieves top-k chunks from Chroma with cosine similarity scores.
        """
        if self.collection.count() == 0:
            logger.warning("Search attempted on empty knowledge base.")
            return []

        query_embedding = embedding_service.get_embedding(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matched_chunks: List[Dict[str, Any]] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(docs)):
            # In Chroma with cosine distance: similarity = 1 - distance
            dist = distances[i] if i < len(distances) else 1.0
            similarity = max(0.0, 1.0 - dist)
            matched_chunks.append({
                "text": docs[i],
                "metadata": metas[i] if i < len(metas) else {},
                "similarity": round(similarity, 4),
            })

        logger.info(f"Retrieved {len(matched_chunks)} chunks for query: '{query[:50]}...'")
        return matched_chunks

# Global vector store instance
vector_store = ChromaVectorStore()
