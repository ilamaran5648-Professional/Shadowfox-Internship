import os
import logging
from typing import List, Dict, Any
from app.config import settings

logger = logging.getLogger(__name__)

# These are CONFIRMED available models from your Gemini API account (ordered by preference)
# The google-genai SDK v2.x requires the SHORT name (without the "models/" prefix)
FALLBACK_MODELS = [
    "gemini-3.5-flash",       # Preferred - confirmed available
    "gemini-3.6-flash",       # Also confirmed available
    "gemini-2.5-flash",       # Stable fallback
    "gemini-flash-latest",    # Latest flash alias
    "gemini-2.5-flash-lite",  # Lightweight fallback
]

class GeminiLLMService:
    """
    LLM generation client powered by Google Gemini API (google-genai SDK v2.x).
    Enforces grounded generation instructions to prevent hallucination.
    Uses a confirmed-available model fallback chain for resilient generation.
    """

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        # Primary model from config; SDK v2.x expects short name WITHOUT "models/" prefix
        self.model_name = settings.LLM_MODEL_NAME.replace("models/", "").strip()
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize Google GenAI client (SDK v2.x)."""
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set. LLM service will operate in offline fallback mode.")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            logger.info(f"Initialized Google GenAI SDK client (v{genai.__version__}).")
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI SDK: {str(e)}")
            self._client = None

    def build_context_string(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved context chunks into a structured prompt section."""
        if not chunks:
            return "No relevant context found in documents."

        formatted_parts = []
        for idx, chunk in enumerate(chunks, 1):
            fn = chunk.get("filename", "Unknown")
            page = chunk.get("page_number", 1)
            cid = chunk.get("chunk_id", f"c{idx}")
            text = chunk.get("text", "").strip()
            part = f"--- [Source #{idx} | File: {fn} | Page: {page} | Chunk ID: {cid}] ---\n{text}"
            formatted_parts.append(part)

        return "\n\n".join(formatted_parts)

    def _build_prompt(self, query: str, context_str: str) -> str:
        """Construct grounded RAG prompt with strict context-only instruction."""
        return f"""You are a strict, objective Retrieval-Augmented Generation (RAG) assistant.
Your task is to answer the user's question ONLY using the factual context snippets provided below.

STRICT GROUNDING RULES:
1. Base your answer ONLY on facts directly stated in the Context below.
2. Do NOT assume, extrapolate, or use outside knowledge.
3. If the provided Context does NOT contain sufficient factual evidence to answer the question, respond EXACTLY with:
   "Information not found in the provided documents."
4. Do NOT make up answers or hallucinate.
5. Whenever stating facts, reference the source document and page number if available.

=== CONTEXT START ===
{context_str}
=== CONTEXT END ===

USER QUESTION: {query}

GROUNDED ANSWER:"""

    def _try_generate(self, model_id: str, prompt: str) -> str:
        """Attempt a single generation call with a specific model ID."""
        response = self._client.models.generate_content(
            model=model_id,
            contents=prompt,
        )
        return response.text.strip()

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a grounded answer strictly using the provided context chunks.
        Tries the configured model first, then falls back through confirmed-available models.
        """
        if not context_chunks:
            return (
                "Information not found in the provided documents.\n\n"
                "*Reason*: No relevant document context matched your query."
            )

        context_str = self.build_context_string(context_chunks)
        prompt = self._build_prompt(query, context_str)

        if not self.api_key or not self._client:
            return self._generate_offline_fallback(query, context_chunks)

        # Build deduplicated list: configured model first, then standard fallbacks
        configured = self.model_name
        candidates = [configured] + [m for m in FALLBACK_MODELS if m != configured]

        last_error = None
        for model_id in candidates:
            try:
                logger.info(f"Trying Gemini model: '{model_id}'")
                answer = self._try_generate(model_id, prompt)
                if answer:
                    logger.info(f"Gemini response received from model: '{model_id}'")
                    return answer
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Model '{model_id}' failed: {last_error[:120]}")
                continue

        logger.error(f"All Gemini model attempts exhausted. Last error: {last_error}")
        return (
            f"Gemini API Error: All model endpoints failed.\n\n"
            f"Last Error: {last_error}\n\n"
            f"Possible Fixes:\n"
            f"- Verify your GEMINI_API_KEY in .env is valid and active.\n"
            f"- Try again in a few seconds (temporary API congestion).\n"
            f"- Get a free key at: https://aistudio.google.com/"
        )

    def _generate_offline_fallback(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Returns a structured offline response when API key is not available."""
        first_chunk = context_chunks[0]
        fn = first_chunk.get("filename", "Unknown")
        page = first_chunk.get("page_number", 1)
        text = first_chunk.get("text", "")[:300]

        return (
            f"[Offline Mode - Set GEMINI_API_KEY in .env to enable LLM generation]\n\n"
            f"Based on {fn} (Page {page}):\n\n"
            f'"{text}..."\n\n'
            f"(Grounded answer sourced from retrieved chunk '{first_chunk.get('chunk_id')}')."
        )

llm_service = GeminiLLMService()
