document.addEventListener("DOMContentLoaded", () => {
  initDropzone();
  checkHealth();
  
  // Enter key shortcut in textarea (Shift+Enter for newline, Enter to submit)
  document.getElementById("query-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  });
});

// Toast notification helper
function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  
  const icon = type === "success" 
    ? `<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>`
    : `<svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`;

  toast.innerHTML = `${icon} <span>${escapeHtml(message)}</span>`;
  container.appendChild(toast);
  
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// System Health & Stats Checker
async function checkHealth() {
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");

  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("Backend server error");
    const data = await res.json();

    if (!data.api_key_configured) {
      statusDot.className = "status-dot";
      statusText.textContent = "Missing GEMINI_API_KEY (.env)";
      showToast("GEMINI_API_KEY is not configured in .env file", "error");
    } else {
      statusDot.className = "status-dot active";
      statusText.textContent = "API Ready & Connected";
    }

    // Update Model Labels & Stats
    if (data.embedding_model) document.getElementById("stat-embed-model").textContent = data.embedding_model;
    if (data.llm_model) document.getElementById("stat-llm-model").textContent = data.llm_model;
    
    if (data.vector_store_stats) {
      document.getElementById("stat-docs").textContent = data.vector_store_stats.total_documents || 0;
      document.getElementById("stat-chunks").textContent = data.vector_store_stats.total_chunks || 0;
    }
  } catch (err) {
    statusDot.className = "status-dot";
    statusText.textContent = "Offline / Connection Failed";
  }
}

// Dropzone Initialization
function initDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length) uploadFiles(files);
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) uploadFiles(e.target.files);
  });
}

// Upload Files API
async function uploadFiles(files) {
  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }

  showToast(`Uploading ${files.length} document(s)...`, "success");

  try {
    const res = await fetch("/upload", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Failed to process files");
    }

    showToast(data.message || "File(s) uploaded successfully!", "success");
    checkHealth();
  } catch (err) {
    showToast(err.message, "error");
  }
}

// Submit Question Query
async function submitQuery() {
  const input = document.getElementById("query-input");
  const question = input.value.trim();
  if (!question) {
    showToast("Please enter a question.", "error");
    return;
  }

  const btnAsk = document.getElementById("btn-ask");
  const btnText = document.getElementById("btn-text");
  const btnSpinner = document.getElementById("btn-spinner");
  const answerSection = document.getElementById("answer-section");
  const answerContent = document.getElementById("answer-content");
  const sourcesContainer = document.getElementById("sources-container");
  const sourcesCount = document.getElementById("sources-count");

  btnAsk.disabled = true;
  btnText.style.display = "none";
  btnSpinner.style.display = "block";

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Query execution failed.");

    // Render Answer
    answerSection.style.display = "block";
    answerContent.innerHTML = formatMarkdown(data.answer);

    // Render Context Sources
    sourcesContainer.innerHTML = "";
    const sources = data.retrieved_sources || [];
    sourcesCount.textContent = sources.length;

    sources.forEach((src) => {
      const card = document.createElement("div");
      card.className = "source-card";
      card.innerHTML = `
        <div class="source-card-header" onclick="toggleSourceSnippet(this)">
          <div class="source-meta">
            <span class="source-name">📄 ${escapeHtml(src.source)}</span>
            <span class="source-page">Page ${src.page}</span>
          </div>
          <div class="source-score">Relevance Distance: ${src.score} ▾</div>
        </div>
        <div class="source-snippet" style="display: none;">${escapeHtml(src.content)}</div>
      `;
      sourcesContainer.appendChild(card);
    });

  } catch (err) {
    showToast(err.message, "error");
  } finally {
    btnAsk.disabled = false;
    btnText.style.display = "inline";
    btnSpinner.style.display = "none";
  }
}

// Toggle accordion for retrieved sources
function toggleSourceSnippet(headerEl) {
  const snippet = headerEl.nextElementSibling;
  if (snippet.style.display === "none") {
    snippet.style.display = "block";
  } else {
    snippet.style.display = "none";
  }
}

// Clear Index API
async function clearIndex() {
  if (!confirm("Are you sure you want to clear the FAISS vector index and all uploaded document chunks?")) {
    return;
  }

  try {
    const res = await fetch("/clear", { method: "POST" });
    const data = await res.json();
    showToast(data.message || "Vector index cleared.", "success");
    
    document.getElementById("answer-section").style.display = "none";
    document.getElementById("query-input").value = "";
    checkHealth();
  } catch (err) {
    showToast("Failed to clear index.", "error");
  }
}

// Simple Helper utilities
function escapeHtml(text) {
  if (!text) return "";
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatMarkdown(text) {
  if (!text) return "";
  let html = escapeHtml(text);
  
  // Bold formatting **text**
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Italic *text*
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  // Line breaks
  html = html.replace(/\n/g, '<br>');
  
  return html;
}
