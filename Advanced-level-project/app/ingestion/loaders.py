import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pypdf

class DocumentLoader:
    """Multi-format document loader supporting PDF, TXT, and Markdown files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    @classmethod
    def is_supported(cls, file_path: str | Path) -> bool:
        """Check if file extension is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def load_document(cls, file_path: str | Path) -> Dict[str, Any]:
        """
        Load a document and extract text along with page/source metadata.

        Returns:
            Dict containing:
            - filename: name of file
            - file_type: pdf / txt / md
            - pages: List[Tuple[int, str]] where int is 1-based page number
            - total_pages: int
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Supported: {cls.SUPPORTED_EXTENSIONS}")

        if ext == ".pdf":
            return cls._load_pdf(path)
        elif ext == ".txt":
            return cls._load_txt(path, file_type="txt")
        elif ext == ".md":
            return cls._load_txt(path, file_type="markdown")
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    @classmethod
    def _load_pdf(cls, path: Path) -> Dict[str, Any]:
        """Extract text page-by-page from PDF."""
        pages: List[Tuple[int, str]] = []
        try:
            reader = pypdf.PdfReader(str(path))
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text() or ""
                cleaned = extracted.strip()
                if cleaned:
                    pages.append((i + 1, cleaned))
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF file '{path.name}': {str(e)}")

        if not pages:
            raise ValueError(f"PDF file '{path.name}' contains no readable text.")

        return {
            "filename": path.name,
            "file_type": "pdf",
            "pages": pages,
            "total_pages": len(reader.pages),
            "file_path": str(path)
        }

    @classmethod
    def _load_txt(cls, path: Path, file_type: str) -> Dict[str, Any]:
        """Read plain text or markdown file."""
        content = ""
        encodings = ["utf-8", "latin-1", "cp1252"]
        
        for enc in encodings:
            try:
                content = path.read_text(encoding=enc).strip()
                break
            except UnicodeDecodeError:
                continue

        if not content:
            raise ValueError(f"File '{path.name}' is empty or could not be decoded.")

        return {
            "filename": path.name,
            "file_type": file_type,
            "pages": [(1, content)],
            "total_pages": 1,
            "file_path": str(path)
        }
