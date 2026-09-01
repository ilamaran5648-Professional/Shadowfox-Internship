import time
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from config import settings
from vector_store import vector_store_manager, VectorStoreError


class RAGPipelineError(Exception):
    """Custom exception raised during RAG pipeline execution."""
    pass


RAG_PROMPT_TEMPLATE = """You are a senior AI assistant specializing in document-based Question Answering (RAG).

CRITICAL RULES:
1. Answer the question relying ONLY and EXCLUSIVELY on the provided Document Context below.
2. Do NOT use outside knowledge, prior assumptions, or extrapolate beyond the explicit facts in the context.
3. If the provided context does NOT contain sufficient information to answer the question, state clearly:
   "I cannot answer this question based on the provided document context."
4. Keep your answer accurate, structured, concise, and formatted with clean Markdown.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

GROUNDED ANSWER:"""


class RAGPipeline:
    """Executes grounded Question Answering over FAISS vector store using Google Gemini LLM with rate limit retries."""
    
    def __init__(self):
        self._llm = None

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            if not settings.GEMINI_API_KEY:
                raise RAGPipelineError(
                    "GEMINI_API_KEY environment variable is not configured."
                )
            try:
                self._llm = ChatGoogleGenerativeAI(
                    model=settings.LLM_MODEL,
                    google_api_key=settings.GEMINI_API_KEY,
                    temperature=0.2,
                    max_output_tokens=1024,
                    max_retries=5
                )
            except Exception as e:
                raise RAGPipelineError(f"Failed to initialize ChatGoogleGenerativeAI model: {str(e)}")
        return self._llm

    def answer_question(self, question: str, top_k: int = None, max_retries: int = 3) -> Dict[str, Any]:
        """Retrieve relevant document chunks and generate a grounded answer."""
        k = top_k or settings.TOP_K
        stats = vector_store_manager.get_stats()
        
        if stats["is_empty"]:
            return {
                "answer": "No documents have been ingested into the system. Please upload a `.pdf` or `.txt` file first.",
                "retrieved_sources": []
            }

        # 1. Similarity Retrieval (handles retries internally)
        search_results = vector_store_manager.similarity_search(query=question, k=k)
        
        if not search_results:
            return {
                "answer": "I cannot answer this question based on the provided document context.",
                "retrieved_sources": []
            }

        # 2. Format Context & Sources
        context_snippets = []
        retrieved_sources = []

        for idx, (doc, score) in enumerate(search_results, start=1):
            source_name = doc.metadata.get("source", "unknown")
            page_num = doc.metadata.get("page", 1)
            snippet = f"--- Snippet {idx} [Source: {source_name} | Page: {page_num}] ---\n{doc.page_content}"
            context_snippets.append(snippet)
            
            retrieved_sources.append({
                "id": idx,
                "content": doc.page_content,
                "source": source_name,
                "page": page_num,
                "score": round(float(score), 4)
            })

        formatted_context = "\n\n".join(context_snippets)

        # 3. Construct Prompt & Generate Response with Retries
        prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        full_prompt_text = prompt.format(context=formatted_context, question=question)

        llm = self._get_llm()

        for attempt in range(max_retries):
            try:
                response = llm.invoke(full_prompt_text)
                
                # Safely parse response content whether it is a string or list
                if isinstance(response.content, str):
                    answer_text = response.content.strip()
                elif isinstance(response.content, list):
                    parts = []
                    for item in response.content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                        elif hasattr(item, "text"):
                            parts.append(item.text)
                        else:
                            parts.append(str(item))
                    answer_text = "\n".join(parts).strip()
                else:
                    answer_text = str(response.content).strip()
                
                return {
                    "answer": answer_text,
                    "retrieved_sources": retrieved_sources
                }
            except Exception as e:
                err_msg = str(e)
                if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota exceeded" in err_msg) and attempt < max_retries - 1:
                    wait_seconds = (attempt + 1) * 3
                    time.sleep(wait_seconds)
                    continue
                elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "Quota exceeded" in err_msg:
                    raise RAGPipelineError(
                        "Gemini LLM API Rate Limit (429) hit. "
                        "Please wait ~15-30 seconds for your free tier quota window to reset before asking again."
                    )
                else:
                    raise RAGPipelineError(f"Failed to generate answer from Gemini API: {err_msg}")


rag_pipeline = RAGPipeline()
