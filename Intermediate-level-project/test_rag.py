import os
import io
import fitz
from fastapi.testclient import TestClient

from config import settings
from document_processor import process_document, extract_text_from_pdf, extract_text_from_txt, DocumentProcessingError
from vector_store import vector_store_manager
from main import app

client = TestClient(app)

def create_sample_pdf_bytes(text: str = "This is a sample test PDF document content for RAG testing.") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def test_config_defaults():
    assert settings.EMBEDDING_MODEL == "models/gemini-embedding-001"
    assert settings.LLM_MODEL == "gemini-3.5-flash"
    assert settings.CHUNK_SIZE == 1000
    assert settings.CHUNK_OVERLAP == 200
    assert settings.TOP_K == 4

def test_extract_text_from_txt():
    sample_text = "Hello world! This is a test document."
    txt_bytes = sample_text.encode("utf-8")
    docs = extract_text_from_txt(txt_bytes, "test.txt")
    assert len(docs) == 1
    assert docs[0].page_content == sample_text
    assert docs[0].metadata["source"] == "test.txt"
    assert docs[0].metadata["page"] == 1

def test_extract_text_from_pdf():
    pdf_bytes = create_sample_pdf_bytes("PDF processing test with PyMuPDF.")
    docs = extract_text_from_pdf(pdf_bytes, "sample.pdf")
    assert len(docs) == 1
    assert "PDF processing test" in docs[0].page_content
    assert docs[0].metadata["source"] == "sample.pdf"
    assert docs[0].metadata["page"] == 1

def test_process_document_chunking():
    long_text = "Paragraph test content. " * 100  # Creates longer text
    txt_bytes = long_text.encode("utf-8")
    chunks = process_document(txt_bytes, "long_document.txt")
    assert len(chunks) > 0
    assert all(len(c.page_content) <= 1100 for c in chunks)  # Chunk size ~1000

def test_unsupported_file_extension():
    try:
        process_document(b"some content", "file.png")
        assert False, "Expected DocumentProcessingError for unsupported file type"
    except DocumentProcessingError:
        pass  # Expected behavior

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store_stats" in data

def test_api_clear():
    response = client.post("/clear")
    assert response.status_code == 200
    assert response.json()["message"] == "Vector store index successfully cleared."

if __name__ == "__main__":
    print("Running verification test suite...")
    test_config_defaults()
    print("[OK] Config defaults verified.")
    test_extract_text_from_txt()
    print("[OK] TXT extraction verified.")
    test_extract_text_from_pdf()
    print("[OK] PyMuPDF PDF extraction verified.")
    test_process_document_chunking()
    print("[OK] Text chunking verified.")
    test_api_health()
    print("[OK] API /health endpoint verified.")
    test_api_clear()
    print("[OK] API /clear endpoint verified.")
    print("\nALL BACKEND UNIT TESTS PASSED SUCCESSFULLY!")
