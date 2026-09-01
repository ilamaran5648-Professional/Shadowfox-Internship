import fitz  # PyMuPDF
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings


class DocumentProcessingError(Exception):
    """Custom exception raised when document processing fails."""
    pass


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    """Extract page-by-page text from PDF file bytes using PyMuPDF (fitz)."""
    documents: List[Document] = []
    try:
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
        if pdf_doc.page_count == 0:
            raise DocumentProcessingError(f"PDF file '{filename}' contains no pages.")
        
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc.load_page(page_num)
            text = page.get_text("text").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                            "total_pages": pdf_doc.page_count
                        }
                    )
                )
        pdf_doc.close()
    except fitz.FileDataError as e:
        raise DocumentProcessingError(f"Corrupt or invalid PDF file '{filename}': {str(e)}")
    except Exception as e:
        if isinstance(e, DocumentProcessingError):
            raise e
        raise DocumentProcessingError(f"Failed to extract text from PDF '{filename}': {str(e)}")
        
    return documents


def extract_text_from_txt(file_bytes: bytes, filename: str) -> List[Document]:
    """Extract text from TXT file bytes supporting UTF-8 and latin-1 fallback."""
    try:
        try:
            text = file_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1").strip()
            
        if not text:
            raise DocumentProcessingError(f"TXT file '{filename}' is empty.")
            
        return [
            Document(
                page_content=text,
                metadata={
                    "source": filename,
                    "page": 1,
                    "total_pages": 1
                }
            )
        ]
    except Exception as e:
        if isinstance(e, DocumentProcessingError):
            raise e
        raise DocumentProcessingError(f"Failed to read text file '{filename}': {str(e)}")


def process_document(file_bytes: bytes, filename: str) -> List[Document]:
    """Validate file, extract raw documents, and split text into chunked Documents."""
    filename_lower = filename.lower()
    
    if filename_lower.endswith(".pdf"):
        raw_docs = extract_text_from_pdf(file_bytes, filename)
    elif filename_lower.endswith(".txt"):
        raw_docs = extract_text_from_txt(file_bytes, filename)
    else:
        raise DocumentProcessingError(f"Unsupported file type for '{filename}'. Only .pdf and .txt are supported.")
        
    if not raw_docs:
        raise DocumentProcessingError(f"No extractable text found in '{filename}'.")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunked_docs = text_splitter.split_documents(raw_docs)
    return chunked_docs
