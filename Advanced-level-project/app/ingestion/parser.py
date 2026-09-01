import re
from typing import Dict, List, Any

class DocumentParser:
    """Cleans raw text and standardizes metadata across different file types."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Sanitize raw text by normalizing whitespace and removing control characters."""
        if not text:
            return ""
        # Replace multiple newlines or tabs with standard whitespace
        text = re.sub(r'[\r\t\f\v]', ' ', text)
        text = re.sub(r' \n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    @classmethod
    def parse_loaded_document(cls, loaded_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process loaded document pages into parsed sections with enriched metadata.

        Returns:
            List of parsed page dicts with fields:
            - text: cleaned text
            - filename: source file name
            - file_type: pdf/txt/markdown
            - page_number: int
        """
        parsed_sections = []
        filename = loaded_doc["filename"]
        file_type = loaded_doc["file_type"]

        for page_num, raw_text in loaded_doc["pages"]:
            cleaned = cls.clean_text(raw_text)
            if cleaned:
                parsed_sections.append({
                    "text": cleaned,
                    "filename": filename,
                    "file_type": file_type,
                    "page_number": page_num
                })

        return parsed_sections
