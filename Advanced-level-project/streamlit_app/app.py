import os
import sys
from pathlib import Path

# Add parent project path to sys.path so python imports work smoothly
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import requests
from streamlit_app.components import render_header, render_groundedness_badge, render_citations

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="DocuMind RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render main header
render_header()

# Initialize session chat messages if missing
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to communicate with FastAPI backend
def call_api(method: str, endpoint: str, **kwargs):
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            res = requests.get(url, timeout=10, **kwargs)
        elif method == "POST":
            res = requests.post(url, timeout=30, **kwargs)
        elif method == "DELETE":
            res = requests.delete(url, timeout=10, **kwargs)
        else:
            return None, "Invalid HTTP method"
        
        if res.status_code == 200:
            return res.json(), None
        else:
            detail = res.json().get("detail", res.text)
            return None, f"API Error ({res.status_code}): {detail}"
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to FastAPI server at {API_BASE_URL}. Make sure FastAPI server is running."
    except Exception as e:
        return None, str(e)

# Sidebar Document Management & Observability
with st.sidebar:
    st.header("📂 Document Manager")

    # Upload Section
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Process & Ingest Files", use_container_width=True):
            with st.spinner("Ingesting and indexing documents..."):
                success_count = 0
                for uf in uploaded_files:
                    # Step 1: Upload file
                    files = {"file": (uf.name, uf.getvalue(), uf.type)}
                    up_data, up_err = call_api("POST", "/upload", files=files)

                    if up_err:
                        st.error(f"Failed to upload '{uf.name}': {up_err}")
                        continue

                    # Step 2: Ingest file
                    ing_data, ing_err = call_api("POST", f"/ingest?filename={uf.name}")

                    if ing_err:
                        st.error(f"Failed to index '{uf.name}': {ing_err}")
                    else:
                        success_count += 1
                        st.success(f"Indexed '{uf.name}' ({ing_data['total_chunks']} chunks).")

                if success_count > 0:
                    st.rerun()

    st.markdown("---")
    st.subheader("📑 Indexed Documents")

    # Fetch document list from API
    docs_data, docs_err = call_api("GET", "/documents")

    indexed_docs = []
    if docs_data and "documents" in docs_data:
        indexed_docs = docs_data["documents"]
        for d in indexed_docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{d['filename']}**  \n`{d['chunks_count']} chunks | {d['file_type'].upper()}`")
            with col2:
                if st.button("🗑️", key=f"del_{d['filename']}"):
                    call_api("DELETE", f"/documents/{d['filename']}")
                    st.rerun()
    else:
        st.info("No documents indexed yet.")

    st.markdown("---")
    st.subheader("🛠️ Retrieval Settings")
    
    top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=10, value=5)

    doc_options = ["All Documents"] + [d["filename"] for d in indexed_docs]
    selected_doc = st.selectbox("Document Filter (Scope)", options=doc_options)
    target_filter = None if selected_doc == "All Documents" else selected_doc

    if st.button("🧹 Clear All Indexed Data", use_container_width=True):
        call_api("DELETE", "/clear-index")
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Observability Stats")
    health_data, _ = call_api("GET", "/health")
    if health_data:
        st.metric("Total Indexed Vectors", health_data.get("total_vectors", 0))
        st.metric("Indexed Files", health_data.get("indexed_documents_count", 0))

# Main Chat Interface
st.subheader("💬 Chat with Document Assistant")

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "groundedness" in msg and msg["groundedness"]:
            render_groundedness_badge(msg["groundedness"])
        if "sources" in msg and msg["sources"]:
            render_citations(msg["sources"])

# User Chat Input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query via RAG backend API
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating grounded response..."):
            req_body = {
                "query": prompt,
                "top_k": top_k,
                "target_filename": target_filter
            }

            res_data, res_err = call_api("POST", "/query", json=req_body)

            if res_err:
                st.error(f"Query Error: {res_err}")
            else:
                answer = res_data["answer"]
                groundedness = res_data.get("groundedness", {})
                sources = res_data.get("sources", [])
                exec_time = res_data.get("execution_time_ms", 0)

                st.markdown(answer)
                render_groundedness_badge(groundedness)
                render_citations(sources)

                st.caption(f"⚡ Execution latency: {exec_time} ms | Chunks used: {len(sources)}")

                # Save assistant response to state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "groundedness": groundedness,
                    "sources": sources
                })
