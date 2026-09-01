import logging
from typing import List, Dict, Any, Optional
from app.embeddings.embedding_service import embedding_service
from app.retrieval.vector_store import vector_store
from app.config import settings

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """Retriever layer for fetching relevant document chunks given a user query."""

    def __init__(self, store=None, embedder=None):
        self.store = store or vector_store
        self.embedder = embedder or embedding_service

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        target_filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Embed query, execute vector search, and return matching document chunks.
        Optionally filter by target_filename for document-scoped search.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        k = top_k or settings.TOP_K_RESULTS
        # Fetch extra results if filtering by document
        fetch_k = k * 3 if target_filename else k

        query_vector = self.embedder.embed_query(clean_query)
        raw_results = self.store.similarity_search(query_vector, top_k=fetch_k)

        # Apply document scoping if requested
        if target_filename:
            raw_results = [
                doc for doc in raw_results
                if doc.get("filename") == target_filename
            ]

        return raw_results[:k]

retriever = DocumentRetriever()
