import hashlib
from typing import Dict, List, Any
from app.config import settings

class RecursiveTextChunker:
    """
    Splits text recursively based on natural separators (paragraphs, sentences, words)
    while maintaining overlap to preserve contextual continuity across chunk boundaries.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using hierarchy of separators."""
        final_chunks = []
        
        # Base condition
        if len(text) <= self.chunk_size:
            return [text]

        # Find best separator
        separator = separators[-1]
        for s in separators:
            if s == "":
                separator = ""
                break
            if s in text:
                separator = s
                break

        # Split text by separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Merge splits up to chunk_size
        current_chunk = []
        current_length = 0

        for split in splits:
            item = split + (separator if separator != "" else "")
            item_len = len(item)

            if current_length + item_len > self.chunk_size:
                if current_chunk:
                    joined = "".join(current_chunk).strip()
                    if joined:
                        final_chunks.append(joined)

                # Maintain overlap
                overlap_buffer = []
                overlap_len = 0
                for prev_item in reversed(current_chunk):
                    if overlap_len + len(prev_item) <= self.chunk_overlap:
                        overlap_buffer.insert(0, prev_item)
                        overlap_len += len(prev_item)
                    else:
                        break

                current_chunk = overlap_buffer + [item]
                current_length = sum(len(x) for x in current_chunk)
            else:
                current_chunk.append(item)
                current_length += item_len

        if current_chunk:
            joined = "".join(current_chunk).strip()
            if joined:
                final_chunks.append(joined)

        return final_chunks

    def create_chunks(self, parsed_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert parsed document sections into metadata-enriched text chunks.

        Returns:
            List of chunk dictionaries containing:
            - chunk_id: unique ID string
            - text: chunk text content
            - filename: source document name
            - page_number: page number
            - file_type: pdf/txt/markdown
            - chunk_index: index of chunk within page
        """
        all_chunks = []

        for section in parsed_sections:
            text = section["text"]
            filename = section["filename"]
            page_number = section["page_number"]
            file_type = section["file_type"]

            raw_chunks = self._split_text(text, self.separators)

            for idx, raw_chunk in enumerate(raw_chunks):
                cleaned_chunk = raw_chunk.strip()
                if not cleaned_chunk:
                    continue

                # Hash chunk content for unique identification
                content_hash = hashlib.md5(f"{filename}_{page_number}_{idx}_{cleaned_chunk[:50]}".encode()).hexdigest()[:8]
                chunk_id = f"{filename}_p{page_number}_c{idx}_{content_hash}"

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": cleaned_chunk,
                    "filename": filename,
                    "page_number": page_number,
                    "file_type": file_type,
                    "chunk_index": idx
                })

        return all_chunks
