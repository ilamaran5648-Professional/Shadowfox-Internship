import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from config import settings
from document_processor import process_document, DocumentProcessingError
from vector_store import vector_store_manager, VectorStoreError
from rag_pipeline import rag_pipeline, RAGPipelineError

app = FastAPI(
    title="Document QA RAG Backend",
    description="Grounded Question Answering API powered by FastAPI, PyMuPDF, FAISS, and Google Gemini API.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request & Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question text to query against ingested documents")
    top_k: Optional[int] = Field(default=None, ge=1, le=10, description="Number of top document chunks to retrieve")


class SourceItem(BaseModel):
    id: int
    content: str
    source: str
    page: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    retrieved_sources: List[SourceItem]


class UploadResponse(BaseModel):
    message: str
    processed_files: List[str]
    chunks_added: int
    total_chunks: int


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint providing vector store status and configuration details."""
    stats = vector_store_manager.get_stats()
    return {
        "status": "healthy",
        "api_key_configured": bool(settings.GEMINI_API_KEY),
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
        "vector_store_stats": stats
    }


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process PDF or TXT files into the FAISS vector index."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided for upload."
        )

    processed_filenames = []
    all_chunks = []
    errors = []

    for file in files:
        filename = file.filename or "unknown_file"
        
        # Check file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".pdf", ".txt"]:
            errors.append(f"'{filename}': Unsupported file type. Only .pdf and .txt files are accepted.")
            continue

        try:
            content = await file.read()
            if not content or len(content.strip()) == 0:
                errors.append(f"'{filename}': File is empty.")
                continue

            chunks = process_document(file_bytes=content, filename=filename)
            all_chunks.extend(chunks)
            processed_filenames.append(filename)

        except DocumentProcessingError as e:
            errors.append(f"'{filename}': {str(e)}")
        except Exception as e:
            errors.append(f"'{filename}': Unexpected error during processing ({str(e)})")

    if not all_chunks:
        error_msg = "; ".join(errors) if errors else "No valid document content could be extracted."
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

    try:
        chunks_added = vector_store_manager.add_documents(all_chunks)
    except VectorStoreError as e:
        err_msg = str(e)
        if "429" in err_msg or "Rate Limit" in err_msg or "Quota" in err_msg:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=err_msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_msg)

    stats = vector_store_manager.get_stats()

    message = f"Successfully processed {len(processed_filenames)} file(s)."
    if errors:
        message += f" Warnings: {'; '.join(errors)}"

    return UploadResponse(
        message=message,
        processed_files=processed_filenames,
        chunks_added=chunks_added,
        total_chunks=stats["total_chunks"]
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Grounded Question Answering over ingested vector context."""
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty."
        )

    try:
        result = rag_pipeline.answer_question(question=request.question, top_k=request.top_k)
        return QueryResponse(**result)
    except (VectorStoreError, RAGPipelineError) as e:
        err_msg = str(e)
        if "429" in err_msg or "Rate Limit" in err_msg or "Quota" in err_msg:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=err_msg)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_msg)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected Error: {str(e)}")


@app.post("/clear")
async def clear_vector_store():
    """Clear all vector index data and uploaded documents."""
    vector_store_manager.clear()
    return {"message": "Vector store index successfully cleared."}


# Mount Static Files & UI Endpoint
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    """Serve the single-page HTML frontend."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        content={"message": "Document QA RAG Backend API is running. Index UI static file missing."},
        status_code=200
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
