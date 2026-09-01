import time
import logging
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END

from app.retrieval.retriever import retriever
from app.retrieval.reranker import reranker
from app.generation.llm import llm_service
from app.validation.groundedness import groundedness_validator
from app.config import settings

logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    query: str
    target_filename: Optional[str]
    top_k: int
    raw_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    answer: str
    groundedness: Dict[str, Any]
    sources: List[Dict[str, Any]]
    execution_time_ms: float
    debug_info: Dict[str, Any]

class RAGPipeline:
    """
    LangGraph-powered stateful RAG pipeline connecting Query Processing, Retrieval,
    Reranking, LLM Generation, Groundedness Validation, and Citation Assembly.
    """

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RAGState)

        # Add Nodes
        workflow.add_node("preprocess_query", self._node_preprocess_query)
        workflow.add_node("retrieve_documents", self._node_retrieve_documents)
        workflow.add_node("rerank_chunks", self._node_rerank_chunks)
        workflow.add_node("generate_answer", self._node_generate_answer)
        workflow.add_node("validate_groundedness", self._node_validate_groundedness)

        # Connect Nodes sequentially
        workflow.set_entry_point("preprocess_query")
        workflow.add_edge("preprocess_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "rerank_chunks")
        workflow.add_edge("rerank_chunks", "generate_answer")
        workflow.add_edge("generate_answer", "validate_groundedness")
        workflow.add_edge("validate_groundedness", END)

        return workflow.compile()

    def _node_preprocess_query(self, state: RAGState) -> Dict[str, Any]:
        """Node 1: Preprocess and sanitize user query."""
        clean_q = state["query"].strip()
        logger.info(f"LangGraph Node [preprocess_query]: {clean_q}")
        return {"query": clean_q}

    def _node_retrieve_documents(self, state: RAGState) -> Dict[str, Any]:
        """Node 2: Similarity search in FAISS vector store."""
        q = state["query"]
        top_k = state.get("top_k", settings.TOP_K_RESULTS)
        target_fn = state.get("target_filename")

        raw_chunks = retriever.retrieve(query=q, top_k=top_k, target_filename=target_fn)
        logger.info(f"LangGraph Node [retrieve_documents]: Retrieved {len(raw_chunks)} raw chunks.")
        return {"raw_chunks": raw_chunks}

    def _node_rerank_chunks(self, state: RAGState) -> Dict[str, Any]:
        """Node 3: Rerank and filter chunks by relevance threshold."""
        raw_chunks = state.get("raw_chunks", [])
        q = state["query"]

        rerank_result = reranker.rerank(query=q, chunks=raw_chunks)
        filtered = rerank_result["filtered_chunks"]

        logger.info(f"LangGraph Node [rerank_chunks]: {len(filtered)} chunks passed score threshold.")
        return {"reranked_chunks": filtered}

    def _node_generate_answer(self, state: RAGState) -> Dict[str, Any]:
        """Node 4: Grounded generation via Gemini LLM."""
        q = state["query"]
        chunks = state.get("reranked_chunks", [])

        answer = llm_service.generate_answer(query=q, context_chunks=chunks)
        logger.info("LangGraph Node [generate_answer]: Answer generated.")
        return {"answer": answer}

    def _node_validate_groundedness(self, state: RAGState) -> Dict[str, Any]:
        """Node 5: Groundedness check and citation extraction."""
        q = state["query"]
        ans = state["answer"]
        chunks = state.get("reranked_chunks", [])

        val_result = groundedness_validator.validate(query=q, answer=ans, context_chunks=chunks)

        # Build clean citation list
        sources = []
        for c in chunks:
            sources.append({
                "filename": c.get("filename"),
                "page_number": c.get("page_number", 1),
                "chunk_id": c.get("chunk_id"),
                "score": round(c.get("score", 0.0), 3),
                "snippet": c.get("text", "")[:200] + "..."
            })

        logger.info(f"LangGraph Node [validate_groundedness]: Status={val_result['confidence_level']}")
        return {
            "groundedness": val_result,
            "sources": sources
        }

    def run(self, query: str, top_k: int = 5, target_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the full RAG workflow pipeline.

        Returns:
            Dict containing answer, sources, groundedness evaluation, and debug stats.
        """
        start_time = time.time()

        initial_state: RAGState = {
            "query": query,
            "target_filename": target_filename,
            "top_k": top_k,
            "raw_chunks": [],
            "reranked_chunks": [],
            "answer": "",
            "groundedness": {},
            "sources": [],
            "execution_time_ms": 0.0,
            "debug_info": {}
        }

        final_state = self.graph.invoke(initial_state)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "query": final_state["query"],
            "answer": final_state["answer"],
            "groundedness": final_state["groundedness"],
            "sources": final_state["sources"],
            "execution_time_ms": elapsed_ms,
            "debug_info": {
                "raw_chunks_retrieved": len(final_state.get("raw_chunks", [])),
                "reranked_chunks_used": len(final_state.get("reranked_chunks", [])),
                "target_filename": target_filename,
                "top_k": top_k
            }
        }

rag_pipeline = RAGPipeline()
