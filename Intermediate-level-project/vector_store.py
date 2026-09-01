import time
import threading
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import settings


class VectorStoreError(Exception):
    """Custom exception for vector store initialization and operation errors."""
    pass


class VectorStoreManager:
    """Thread-safe manager for FAISS vector store using Google Gemini Embeddings with retry logic."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.vector_store: Optional[FAISS] = None
        self.sources: Dict[str, int] = {}  # filename -> chunk count
        self._embeddings: Optional[GoogleGenerativeAIEmbeddings] = None

    def _get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        if self._embeddings is None:
            if not settings.GEMINI_API_KEY:
                raise VectorStoreError(
                    "GEMINI_API_KEY environment variable is not configured. "
                    "Please set GEMINI_API_KEY in your .env file or environment."
                )
            try:
                self._embeddings = GoogleGenerativeAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    google_api_key=settings.GEMINI_API_KEY,
                    max_retries=5
                )
            except Exception as e:
                raise VectorStoreError(f"Failed to initialize Gemini Embeddings: {str(e)}")
        return self._embeddings

    def add_documents(self, documents: List[Document], max_retries: int = 3) -> int:
        """Add document chunks to the FAISS vector index with exponential backoff on 429 rate limit errors."""
        if not documents:
            return 0

        embeddings = self._get_embeddings()

        with self._lock:
            for attempt in range(max_retries):
                try:
                    if self.vector_store is None:
                        self.vector_store = FAISS.from_documents(documents, embeddings)
                    else:
                        self.vector_store.add_documents(documents)

                    # Update metadata statistics
                    for doc in documents:
                        source = doc.metadata.get("source", "unknown")
                        self.sources[source] = self.sources.get(source, 0) + 1

                    return len(documents)
                except Exception as e:
                    err_msg = str(e)
                    if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota exceeded" in err_msg) and attempt < max_retries - 1:
                        wait_seconds = (attempt + 1) * 3
                        time.sleep(wait_seconds)
                        continue
                    elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota exceeded" in err_msg:
                        raise VectorStoreError(
                            "Gemini API rate limit (429 Quota Exceeded) hit during document embedding. "
                            "Free tier quotas allow a limited number of requests per minute. Please wait 30 seconds and retry."
                        )
                    else:
                        raise VectorStoreError(f"Error adding documents to FAISS vector index: {err_msg}")

    def similarity_search(self, query: str, k: int = 4, max_retries: int = 3) -> List[Tuple[Document, float]]:
        """Perform similarity search with distance score and exponential backoff retry."""
        with self._lock:
            if self.vector_store is None:
                return []

            for attempt in range(max_retries):
                try:
                    results = self.vector_store.similarity_search_with_score(query, k=k)
                    return results
                except Exception as e:
                    err_msg = str(e)
                    if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota exceeded" in err_msg) and attempt < max_retries - 1:
                        wait_seconds = (attempt + 1) * 3
                        time.sleep(wait_seconds)
                        continue
                    elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota exceeded" in err_msg:
                        raise VectorStoreError(
                            "Gemini Embedding API Rate Limit (429) hit. "
                            "Please wait ~15-30 seconds for your free tier API quota to reset before submitting your query again."
                        )
                    else:
                        raise VectorStoreError(f"Similarity search failed: {err_msg}")

    def clear(self):
        """Reset the vector index and stats."""
        with self._lock:
            self.vector_store = None
            self.sources.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store index statistics."""
        with self._lock:
            total_chunks = sum(self.sources.values())
            return {
                "total_chunks": total_chunks,
                "total_documents": len(self.sources),
                "sources": self.sources,
                "is_empty": self.vector_store is None or total_chunks == 0
            }


# Global singleton instance
vector_store_manager = VectorStoreManager()
