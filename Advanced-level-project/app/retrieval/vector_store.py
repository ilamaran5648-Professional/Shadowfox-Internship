import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import faiss
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    FAISS-backed Vector Storage with metadata persistence and document management.
    Uses Inner Product (IndexFlatIP) on normalized embeddings for Cosine Similarity search.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index_dir = Path(settings.INDEX_DIR)
        self.index_file = self.index_dir / "faiss_index.bin"
        self.metadata_file = self.index_dir / "metadata.json"
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []

        # Load index from disk if it exists
        self.load_index()

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Add new document chunks and vectors to FAISS index.
        """
        if not chunks or embeddings.size == 0:
            return

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: Expected {self.dimension}, got {embeddings.shape[1]}")

        # Add vectors to FAISS index
        self.index.add(embeddings)
        # Store metadata in corresponding order
        self.metadata.extend(chunks)

        # Save state to disk
        self.save_index()
        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total vectors: {self.index.ntotal}")

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search and return top-k chunks with similarity scores.
        """
        if self.index.ntotal == 0:
            return []

        k = top_k or settings.TOP_K_RESULTS
        k = min(k, self.index.ntotal)

        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            chunk_data = dict(self.metadata[idx])
            chunk_data["score"] = float(dist)  # Cosine similarity score [0, 1]
            results.append(chunk_data)

        return results

    def delete_document(self, filename: str) -> int:
        """
        Remove all chunks belonging to a document and rebuild FAISS index.
        """
        if self.index.ntotal == 0:
            return 0

        # Filter out chunks matching filename
        retained_metadata = []
        retained_indices = []

        for idx, item in enumerate(self.metadata):
            if item.get("filename") != filename:
                retained_metadata.append(item)
                retained_indices.append(idx)

        deleted_count = len(self.metadata) - len(retained_metadata)
        if deleted_count == 0:
            return 0

        # Reconstruct FAISS index with retained vectors
        new_index = faiss.IndexFlatIP(self.dimension)
        if retained_indices:
            # Extract retained vectors from existing index
            vectors = np.zeros((len(retained_indices), self.dimension), dtype=np.float32)
            for i, old_idx in enumerate(retained_indices):
                vectors[i] = self.index.reconstruct(old_idx)
            new_index.add(vectors)

        self.index = new_index
        self.metadata = retained_metadata

        # Save rebuilt index
        self.save_index()
        logger.info(f"Deleted '{filename}' ({deleted_count} chunks removed). Remaining vectors: {self.index.ntotal}")
        return deleted_count

    def clear_index(self):
        """Reset the vector store and wipe index files."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()
        logger.info("Cleared FAISS vector store.")

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Return list of distinct indexed files with metadata."""
        docs: Dict[str, Dict[str, Any]] = {}
        for item in self.metadata:
            fn = item["filename"]
            if fn not in docs:
                docs[fn] = {
                    "filename": fn,
                    "file_type": item.get("file_type", "unknown"),
                    "chunks_count": 0,
                    "pages": set()
                }
            docs[fn]["chunks_count"] += 1
            docs[fn]["pages"].add(item.get("page_number", 1))

        output = []
        for fn, info in docs.items():
            output.append({
                "filename": fn,
                "file_type": info["file_type"],
                "chunks_count": info["chunks_count"],
                "pages_count": len(info["pages"])
            })
        return output

    def save_index(self):
        """Persist FAISS index binary and metadata JSON to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_file))
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def load_index(self):
        """Load FAISS index binary and metadata JSON from disk."""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Successfully loaded index from disk. Total vectors: {self.index.ntotal}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {str(e)}. Initializing empty index.")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = []

vector_store = FAISSVectorStore()
