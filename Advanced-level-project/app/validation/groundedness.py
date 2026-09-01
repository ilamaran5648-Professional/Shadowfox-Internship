import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GroundednessValidator:
    """
    Validates whether the generated answer is strictly grounded in the retrieved document context.
    Prevents hallucination by calculating term overlap and checking for ungrounded claims.
    """

    REFUSAL_PHRASES = [
        "information not found",
        "not mentioned in the provided",
        "no relevant document context",
        "does not contain enough information",
        "cannot answer based on the provided context"
    ]

    def validate(
        self,
        query: str,
        answer: str,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate answer groundedness against retrieved context.

        Returns:
            Dict containing:
            - is_grounded: bool
            - confidence_level: "HIGH" | "MEDIUM" | "LOW" | "N/A (Refusal)"
            - overlap_score: float (0.0 to 1.0)
            - is_refusal: bool
            - validation_message: str
        """
        if not answer or not answer.strip():
            return {
                "is_grounded": False,
                "confidence_level": "LOW",
                "overlap_score": 0.0,
                "is_refusal": False,
                "validation_message": "Empty answer generated."
            }

        answer_lower = answer.lower()

        # Check if the model explicitly refused to answer (which is valid grounded behavior)
        for phrase in self.REFUSAL_PHRASES:
            if phrase in answer_lower:
                return {
                    "is_grounded": True,
                    "confidence_level": "N/A (Refusal)",
                    "overlap_score": 1.0,
                    "is_refusal": True,
                    "validation_message": "Grounded refusal: Model correctly identified missing information."
                }

        if not context_chunks:
            return {
                "is_grounded": False,
                "confidence_level": "LOW",
                "overlap_score": 0.0,
                "is_refusal": False,
                "validation_message": "No context available to ground answer."
            }

        # Combine all retrieved context text
        full_context_text = " ".join([c.get("text", "") for c in context_chunks]).lower()

        # Extract words (length > 3) from generated answer, excluding common stop words
        stop_words = {
            "the", "and", "is", "in", "to", "of", "it", "that", "you", "he", "was", "for",
            "on", "are", "as", "with", "his", "they", "at", "be", "this", "from", "or",
            "had", "by", "not", "word", "but", "what", "some", "we", "can", "out", "other",
            "were", "all", "there", "when", "up", "use", "your", "how", "said", "an", "each",
            "she", "which", "do", "their", "time", "if", "will", "way", "about", "many",
            "then", "them", "would", "like", "so", "these", "her", "long", "make", "thing",
            "see", "him", "two", "has", "look", "more", "day", "could", "go", "come", "did",
            "my", "sound", "no", "most", "number", "who", "over", "know", "than", "call"
        }

        answer_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', answer_lower))
        content_words = [w for w in answer_words if w not in stop_words]

        if not content_words:
            return {
                "is_grounded": True,
                "confidence_level": "HIGH",
                "overlap_score": 1.0,
                "is_refusal": False,
                "validation_message": "Answer contains general phrasing."
            }

        # Count how many content words appear in context text
        matched_words = [w for w in content_words if w in full_context_text]
        overlap_score = len(matched_words) / len(content_words)

        if overlap_score >= 0.55:
            confidence = "HIGH"
            is_grounded = True
            msg = f"High grounding confidence ({len(matched_words)}/{len(content_words)} key terms verified in context)."
        elif overlap_score >= 0.30:
            confidence = "MEDIUM"
            is_grounded = True
            msg = f"Moderate grounding confidence ({len(matched_words)}/{len(content_words)} key terms verified in context)."
        else:
            confidence = "LOW"
            is_grounded = False
            msg = f"Low grounding confidence ({len(matched_words)}/{len(content_words)} key terms verified in context). Potential ungrounded content."

        return {
            "is_grounded": is_grounded,
            "confidence_level": confidence,
            "overlap_score": round(overlap_score, 2),
            "is_refusal": False,
            "validation_message": msg
        }

groundedness_validator = GroundednessValidator()
