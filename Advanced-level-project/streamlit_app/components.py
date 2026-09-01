import streamlit as st
from typing import Dict, List, Any

def render_header():
    """Render top header banner with modern dark glassmorphism style."""
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1e1e2f 0%, #0f172a 100%);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #334155;
            margin-bottom: 24px;
            color: #f8fafc;
        ">
            <h1 style="margin: 0; font-size: 2.2rem; color: #38bdf8; display: flex; align-items: center; gap: 12px;">
                🧠 DocuMind RAG Assistant
            </h1>
            <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
                Production-Style Multi-Format Document Intelligence with Strict Groundedness Guardrails
            </p>
            <div style="margin-top: 12px; display: flex; gap: 16px; font-size: 0.85rem; color: #cbd5e1;">
                <span>📄 PDF / TXT / MD Support</span>
                <span>•</span>
                <span>⚡ FAISS Vector Search</span>
                <span>•</span>
                <span>🛡️ Anti-Hallucination Checks</span>
                <span>•</span>
                <span>🤖 Google Gemini LLM</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_groundedness_badge(groundedness: Dict[str, Any]):
    """Render a color-coded status badge for groundedness confidence."""
    level = groundedness.get("confidence_level", "LOW")
    msg = groundedness.get("validation_message", "")
    score = groundedness.get("overlap_score", 0.0)

    if level == "HIGH":
        color = "#10b981"  # emerald green
        bg = "#064e3b"
        icon = "✅"
    elif level == "MEDIUM":
        color = "#f59e0b"  # amber
        bg = "#78350f"
        icon = "⚠️"
    elif "Refusal" in str(level):
        color = "#60a5fa"  # blue
        bg = "#1e3a8a"
        icon = "ℹ️"
    else:
        color = "#ef4444"  # red
        bg = "#7f1d1d"
        icon = "🚨"

    st.markdown(
        f"""
        <div style="
            background-color: {bg};
            border: 1px solid {color};
            color: {color};
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 0.88rem;
            margin: 10px 0 16px 0;
            display: inline-block;
        ">
            <strong>{icon} Groundedness Check ({level})</strong>: {msg} (Overlap: {int(score*100)}%)
        </div>
        """,
        unsafe_allow_html=True
    )

def render_citations(sources: List[Dict[str, Any]]):
    """Render structured source citations in expandable accordions."""
    if not sources:
        return

    st.markdown("#### 📚 Verified Source Citations")
    for idx, src in enumerate(sources, 1):
        filename = src.get("filename", "Unknown")
        page = src.get("page_number", 1)
        chunk_id = src.get("chunk_id", "N/A")
        score = src.get("score", 0.0)
        snippet = src.get("snippet", "")

        with st.expander(f"Source #{idx}: {filename} (Page {page}) — Similarity Score: {score:.3f}"):
            st.markdown(f"**Chunk ID**: `{chunk_id}`")
            st.markdown(f"**Excerpt**:\n> {snippet}")
