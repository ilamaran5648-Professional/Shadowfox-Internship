const header = document.getElementById("siteHeader");
if (header) {
  const onScroll = () => {
    header.classList.toggle("is-scrolled", window.scrollY > 8);
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

const notes = document.getElementById("notes");
const wordCount = document.getElementById("word-count");
const form = document.getElementById("studyForm");

function countWords(text) {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

function updateWordCount() {
  if (!notes || !wordCount) return;
  const n = countWords(notes.value);
  wordCount.textContent = `${n} word${n === 1 ? "" : "s"}`;

  const hint = document.getElementById("notesHint");
  if (hint) {
    if (n > 2000) {
      hint.textContent = "That's a lot of text — consider summarizing in smaller chunks for best results.";
      hint.classList.add("is-warning");
    } else {
      hint.textContent = "Works best with 50–2,000 words at a time.";
      hint.classList.remove("is-warning");
    }
  }
}

if (notes) {
  notes.addEventListener("input", updateWordCount);
  updateWordCount();
}

if (form) {
  form.addEventListener("submit", (event) => {
    if (notes && !notes.value.trim()) {
      event.preventDefault();
      notes.focus();
      return;
    }

    const clicked = event.submitter || document.activeElement;

    setTimeout(() => {
      if (clicked && clicked.classList && clicked.classList.contains("btn")) {
        clicked.classList.add("is-loading");
      }
      form.querySelectorAll(".btn").forEach((btn) => {
        btn.setAttribute("disabled", "true");
      });
    }, 0);
  });
}

const copyBtn = document.getElementById("copyBtn");
const resultContent = document.getElementById("resultContent");

if (copyBtn && resultContent) {
  copyBtn.addEventListener("click", async () => {
    const label = copyBtn.querySelector(".btn-label");
    
    try {
      await navigator.clipboard.writeText(resultContent.innerText);
      if (label) {
        const original = label.textContent;
        label.textContent = "Copied!";
        setTimeout(() => { label.textContent = original; }, 1600);
      }
    } catch (err) {
      if (label) label.textContent = "Couldn't copy";
    }
  });
}