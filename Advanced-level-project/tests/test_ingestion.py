import tempfile
from pathlib import Path
import pytest
from app.ingestion.loaders import DocumentLoader
from app.ingestion.parser import DocumentParser
from app.ingestion.chunker import RecursiveTextChunker

def test_document_loader_txt():
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
        f.write("Line 1 text.\nLine 2 text.")
        temp_path = Path(f.name)

    try:
        loaded = DocumentLoader.load_document(temp_path)
        assert loaded["filename"] == temp_path.name
        assert loaded["file_type"] == "txt"
        assert len(loaded["pages"]) == 1
        assert "Line 1 text." in loaded["pages"][0][1]
    finally:
        temp_path.unlink()

def test_document_parser_clean():
    raw = "Hello   world!\n\n\n\nThis   is a test."
    cleaned = DocumentParser.clean_text(raw)
    assert "Hello world!" in cleaned
    assert "\n\n\n" not in cleaned

def test_recursive_chunker():
    sections = [{
        "text": "Paragraph 1 is here. " * 20 + "\n\n" + "Paragraph 2 is here. " * 20,
        "filename": "sample.md",
        "file_type": "markdown",
        "page_number": 1
    }]
    chunker = RecursiveTextChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.create_chunks(sections)
    
    assert len(chunks) > 1
    assert chunks[0]["filename"] == "sample.md"
    assert chunks[0]["chunk_id"].startswith("sample.md_p1_c0")
