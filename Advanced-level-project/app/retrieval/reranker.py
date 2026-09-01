import logging
from typing import List, Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)

class ScoreThresholdReranker:
    """
    Reranks and filters retrieved chunks to eliminate irrelevant noise.
    Filters out chunks with similarity scores below the configured threshold.
    """

    def __init__(self, threshold: float = None):
        self.threshold = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

    def rerank(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Filter chunks by score threshold and deduplicate identical snippets.

        Returns:
            Dict containing:
            - filtered_chunks: List of high-relevance chunks
            - discarded_chunks_count: int
            - top_score: float
        """
        if not chunks:
            return {
                "filtered_chunks": [],
                "discarded_chunks_count": 0,
                "top_score": 0.0
            }

        filtered = []
        discarded_count = 0
        seen_texts = set()

        # Sort chunks by score descending
        sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)
        top_score = sorted_chunks[0].get("score", 0.0) if sorted_chunks else 0.0

        for chunk in sorted_chunks:
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "").strip()

            # Skip below threshold
            if score < self.threshold:
                discarded_count += 1
                continue

            # Skip duplicate text snippets
            text_snippet = text[:100].lower()
            if text_snippet in seen_texts:
                discarded_count += 1
                continue

            seen_texts.add(text_snippet)
            filtered.append(chunk)

        logger.info(f"Reranker: Kept {len(filtered)} chunks, discarded {discarded_count} low-relevance chunks (Threshold={self.threshold})")

        return {
            "filtered_chunks": filtered,
            "discarded_chunks_count": discarded_count,
            "top_score": top_score
        }

reranker = ScoreThresholdReranker()
