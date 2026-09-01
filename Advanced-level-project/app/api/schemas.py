from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    message: str
    filename: str
    file_type: str
    file_size_bytes: int
    saved_path: str

class IngestResponse(BaseModel):
    message: str
    filename: str
    total_pages: int
    total_chunks: int
    vector_count: int

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or query text")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of top chunks to retrieve")
    target_filename: Optional[str] = Field(default=None, description="Optional document name to constrain search")

class SourceCitation(BaseModel):
    filename: str
    page_number: int
    chunk_id: str
    score: float
    snippet: str

class GroundednessInfo(BaseModel):
    is_grounded: bool
    confidence_level: str
    overlap_score: float
    is_refusal: bool
    validation_message: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    groundedness: GroundednessInfo
    sources: List[SourceCitation]
    execution_time_ms: float
    debug_info: Dict[str, Any]

class DocumentItem(BaseModel):
    filename: str
    file_type: str
    chunks_count: int
    pages_count: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentItem]
    total_documents: int
    total_chunks: int

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    total_vectors: int
    indexed_documents_count: int
