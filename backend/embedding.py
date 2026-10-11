import time
from typing import List
import httpx
from config import logger, GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

class GeminiEmbeddingService:
    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_EMBEDDING_MODEL):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def get_embedding(self, text: str) -> List[float]:
        """
        Embed a single text string using Gemini text-embedding-004.
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        url = f"{self.base_url}/models/{self.model}:embedContent?key={self.api_key}"
        payload = {
            "model": f"models/{self.model}",
            "content": {
                "parts": [{"text": text}]
            }
        }

        # Retry logic with backoff
        for attempt in range(3):
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(url, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        return data.get("embedding", {}).get("values", [])
                    elif response.status_code == 429:
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        logger.error(f"Gemini embedding API error ({response.status_code}): {response.text}")
                        raise RuntimeError(f"Embedding API error: {response.status_code} - {response.text}")
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Embedding failed after 3 attempts: {e}")
                    raise
                time.sleep(1)

        raise RuntimeError("Failed to generate embedding after retries.")

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """
        Batch embed multiple texts to optimize throughput and comply with rate limits.
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        all_embeddings: List[List[float]] = []
        url = f"{self.base_url}/models/{self.model}:batchEmbedContents?key={self.api_key}"

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            requests = [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]}
                }
                for t in batch
            ]
            
            payload = {"requests": requests}
            
            success = False
            for attempt in range(3):
                try:
                    with httpx.Client(timeout=45.0) as client:
                        response = client.post(url, json=payload)
                        if response.status_code == 200:
                            data = response.json()
                            embeddings_data = data.get("embeddings", [])
                            for item in embeddings_data:
                                all_embeddings.append(item.get("values", []))
                            success = True
                            break
                        elif response.status_code == 429:
                            time.sleep(2 ** attempt)
                        else:
                            logger.warning(f"Batch embed failed with {response.status_code}, falling back to single embeds")
                            break
                except Exception as e:
                    logger.warning(f"Batch request exception: {e}")
                    time.sleep(1)

            if not success:
                # Fallback to item-by-item embedding if batch endpoint fails
                logger.info(f"Processing batch of {len(batch)} items sequentially")
                for text in batch:
                    all_embeddings.append(self.get_embedding(text))

        return all_embeddings


# Global singleton instance
embedding_service = GeminiEmbeddingService()
