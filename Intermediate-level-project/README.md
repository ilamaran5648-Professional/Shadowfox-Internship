# Document-Based Question Answering (RAG) Backend

A complete, modular, and production-ready RAG backend application built with **FastAPI**, **PyMuPDF**, **LangChain**, **FAISS**, and **Google Gemini API** (`text-embedding-004` & `gemini-3.5-flash`), paired with a sleek glassmorphic Web UI.

---

## 📁 Project Structure

```text
.
├── main.py                # FastAPI REST API endpoints, routing & CORS
├── config.py              # Configuration manager & environment variable loader
├── document_processor.py   # PDF (PyMuPDF) / TXT parsing & LangChain chunking
├── vector_store.py        # Thread-safe FAISS vector index & Gemini embeddings
├── rag_pipeline.py        # Strict grounded prompt template & Gemini QA chain
├── static/                # Web UI static assets
│   ├── index.html         # Modern responsive web layout
│   ├── styles.css         # Custom CSS design system (glassmorphism)
│   └── app.js             # Drag-drop upload, querying & API communication
├── requirements.txt       # Pinned Python package dependencies
├── .env.example           # Environment template file
└── README.md              # Documentation & usage instructions
```

---

## ⚡ Quick Start Instructions

### 1. Prerequisites
- Python 3.10 or higher
- A Google Gemini API Key (Get a free key from [Google AI Studio](https://aistudio.google.com/))

### 2. Environment Setup & Dependency Installation

Create a virtual environment and install the required dependencies:

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Gemini API Key

Copy the `.env.example` file to `.env` and set your `GEMINI_API_KEY`:

```bash
# Windows PowerShell
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env` to include your actual API key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
EMBEDDING_MODEL=models/text-embedding-004
LLM_MODEL=gemini-2.5-flash
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
```

Alternatively, set the environment variable directly in your terminal:

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_actual_gemini_api_key_here"

# Linux / macOS
export GEMINI_API_KEY="your_actual_gemini_api_key_here"
```

---

##  Running the Application

Launch the server using Uvicorn:

```bash
uvicorn main:app --reload
```

Once running:
- **Web User Interface**: Open [http://localhost:8000](http://localhost:8000) in your web browser.
- **Interactive OpenAPI Specs**: Open [http://localhost:8000/docs](http://localhost:8000/docs).

---

##  API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive Web Client |
| `GET` | `/health` | Server health check & vector store status |
| `POST` | `/upload` | Upload `.pdf` or `.txt` files to extract text & add to FAISS vector index |
| `POST` | `/query` | Execute grounded question answering over ingested vector context |
| `POST` | `/clear` | Clear the in-memory FAISS vector index |

### Sample Query API Payload (`POST /query`):

```json
{
  "question": "What are the main findings in section 2?",
  "top_k": 4
}
```

### Sample Query API Response:

```json
{
  "answer": "According to section 2 of the document, the main findings demonstrate...",
  "retrieved_sources": [
    {
      "id": 1,
      "content": "Section 2: Results... ",
      "source": "annual_report.pdf",
      "page": 3,
      "score": 0.412
    }
  ]
}

---

## 🧪 Testing & Code Verification

You can verify the backend without running the Web UI using Python:

```bash
python -c "import main; print('FastAPI App initialized successfully!')"
```
