import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Local Embedding Generation Service using SentenceTransformers.
    Runs on CPU with L2 normalization for Cosine Similarity vector search.
    """

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def _get_model(self) -> SentenceTransformer:
        """Lazy load SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
            self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate L2-normalized dense embeddings for a batch of text chunks.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) with float32 type.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate L2-normalized embedding for a single user query.

        Returns:
            np.ndarray of shape (1, embedding_dim) with float32 type.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        model = self._get_model()
        embedding = model.encode([query.strip()], convert_to_numpy=True, normalize_embeddings=True)
        return embedding.astype(np.float32)

embedding_service = EmbeddingService()
