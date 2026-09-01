"""Quick smoke test: verify Gemini API + RAG LLM generation works end-to-end."""
import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from app.generation.llm import llm_service

# Fake context chunk simulating a retrieved RAG document chunk
test_chunks = [{
    "filename": "AI Engineer Internship Task List.pdf",
    "page_number": 1,
    "chunk_id": "test_chunk_001",
    "text": "The SHADOWFOX AI Engineer Internship runs from August 1, 2026 to September 2, 2026. The intern is required to build a production-style RAG Assistant using FastAPI, Streamlit, FAISS, LangGraph, and Google Gemini.",
    "score": 0.85
}]

print("Testing Gemini API generation with model:", llm_service.model_name)
print("-" * 60)

answer = llm_service.generate_answer(
    query="What is the internship duration and what technologies are required?",
    context_chunks=test_chunks
)

print("ANSWER:\n")
print(answer)
print("\n" + "-" * 60)
print("SUCCESS: Gemini LLM generation working correctly!")
