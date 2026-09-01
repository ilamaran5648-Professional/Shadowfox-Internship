import shutil
import logging
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, status

from app.config import settings
from app.ingestion.loaders import DocumentLoader
from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import RecursiveTextChunker
from app.embeddings.embedding_service import embedding_service
from app.retrieval.vector_store import vector_store
from app.pipeline.rag_pipeline import rag_pipeline
from app.api.schemas import (
    UploadResponse, IngestResponse, QueryRequest, QueryResponse,
    DocumentListResponse, DocumentItem, HealthResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    """Verify application health and vector index status."""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.VERSION,
        total_vectors=vector_store.index.ntotal,
        indexed_documents_count=len(vector_store.get_indexed_documents())
    )

@router.post("/upload", response_model=UploadResponse, summary="Upload Document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, MD) to the server staging directory.
    """
    if not DocumentLoader.is_supported(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.filename}'. Allowed extensions: .pdf, .txt, .md"
        )

    upload_path = Path(settings.UPLOADS_DIR) / file.filename

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = upload_path.stat().st_size

        return UploadResponse(
            message=f"File '{file.filename}' uploaded successfully.",
            filename=file.filename,
            file_type=upload_path.suffix.lower().replace(".", ""),
            file_size_bytes=file_size,
            saved_path=str(upload_path)
        )
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )

@router.post("/ingest", response_model=IngestResponse, summary="Ingest & Index Document")
def ingest_document(filename: str):
    """
    Parse, chunk, embed, and index an uploaded document in FAISS.
    """
    file_path = Path(settings.UPLOADS_DIR) / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Uploaded file '{filename}' not found in uploads directory."
        )

    try:
        # 1. Load document
        loaded_doc = DocumentLoader.load_document(file_path)

        # 2. Parse text & metadata
        parsed_sections = DocumentParser.parse_loaded_document(loaded_doc)

        # 3. Chunk text
        chunker = RecursiveTextChunker()
        chunks = chunker.create_chunks(parsed_sections)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document '{filename}' yielded no readable text chunks."
            )

        # 4. Generate embeddings
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedding_service.embed_texts(chunk_texts)

        # 5. Store in FAISS
        # Remove any prior indexing of this document first to avoid duplication
        vector_store.delete_document(filename)
        vector_store.add_chunks(chunks, embeddings)

        return IngestResponse(
            message=f"Successfully ingested and indexed '{filename}'.",
            filename=filename,
            total_pages=loaded_doc["total_pages"],
            total_chunks=len(chunks),
            vector_count=vector_store.index.ntotal
        )
    except Exception as e:
        logger.error(f"Ingestion error for '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse, summary="Query RAG Assistant")
def query_rag(request: QueryRequest):
    """
    Execute full LangGraph RAG pipeline: Query -> Retrieval -> Rerank -> LLM -> Validation -> Answer.
    """
    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Please upload and ingest documents first."
        )

    try:
        result = rag_pipeline.run(
            query=request.query,
            top_k=request.top_k,
            target_filename=request.target_filename
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query execution failed: {str(e)}"
        )

@router.get("/documents", response_model=DocumentListResponse, summary="List Indexed Documents")
def list_documents():
    """Get list of indexed documents and overall vector stats."""
    docs = vector_store.get_indexed_documents()
    total_chunks = sum(d["chunks_count"] for d in docs)
    items = [DocumentItem(**d) for d in docs]
    
    return DocumentListResponse(
        documents=items,
        total_documents=len(items),
        total_chunks=total_chunks
    )

@router.delete("/documents/{filename}", summary="Delete Document from Index")
def delete_document(filename: str):
    """Delete a document and its associated vectors from FAISS index."""
    deleted_count = vector_store.delete_document(filename)

    # Delete raw file if present
    raw_path = Path(settings.UPLOADS_DIR) / filename
    if raw_path.exists():
        raw_path.unlink()

    return {
        "message": f"Deleted '{filename}' from index.",
        "removed_chunks_count": deleted_count,
        "remaining_vectors_count": vector_store.index.ntotal
    }

@router.delete("/clear-index", summary="Wipe Vector Index")
def clear_all():
    """Wipe all index data and clear uploads directory."""
    vector_store.clear_index()
    for f in Path(settings.UPLOADS_DIR).glob("*"):
        if f.is_file() and f.name != ".gitkeep":
            f.unlink()
    return {"message": "All indexed data and uploaded files have been cleared."}
