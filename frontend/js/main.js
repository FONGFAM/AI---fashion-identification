/**
 * AI Fashion Identification - main.js
 * Xử lý: Upload ảnh, Drag & Drop, Gọi API, Hiển thị kết quả
 */

const API_URL = "http://localhost:5000/api/predict";

// ── DOM Elements ─────────────────────────────────────────────────
const dropArea         = document.getElementById("drop-area");
const fileInput        = document.getElementById("fileElem");
const previewContainer = document.getElementById("preview-container");
const imagePreview     = document.getElementById("image-preview");
const btnPredict       = document.getElementById("btn-predict");
const btnReset         = document.getElementById("btn-reset");
const resultSection    = document.getElementById("result-section");
const resultName       = document.getElementById("result-name");
const resultConfidence = document.getElementById("result-confidence");
const mockWarning      = document.getElementById("mock-warning");
const loadingOverlay   = document.getElementById("loading");
const confidenceBar    = document.getElementById("confidence-bar");
const confidenceValue  = document.getElementById("confidence-value");
const allClassesList   = document.getElementById("all-classes-list");
const aiProcessedPreview = document.getElementById("ai-processed-preview");

// ── State ─────────────────────────────────────────────────────────
let currentFile = null;

// ── Fashion MNIST Class Icons ─────────────────────────────────────
const CLASS_META = {
  "Áo thun / Áo phông":          { icon: "👕", color: "#6366f1" },
  "Quần dài":                    { icon: "👖", color: "#8b5cf6" },
  "Áo len":                      { icon: "🧥", color: "#ec4899" },
  "Đầm / Váy liền":              { icon: "👗", color: "#f43f5e" },
  "Áo khoác":                    { icon: "🥼", color: "#f97316" },
  "Sandal / Dép quai hậu":       { icon: "👡", color: "#eab308" },
  "Áo sơ mi":                    { icon: "👔", color: "#22c55e" },
  "Giày thể thao":               { icon: "👟", color: "#06b6d4" },
  "Túi xách":                    { icon: "👜", color: "#3b82f6" },
  "Bốt cổ thấp":                 { icon: "👢", color: "#a855f7" },
};

// ── Utility ───────────────────────────────────────────────────────
function showEl(el)  { el.classList.remove("hidden"); }
function hideEl(el)  { el.classList.add("hidden"); }

function setLoading(active) {
  if (active) {
    showEl(loadingOverlay);
    btnPredict.disabled = true;
    btnPredict.textContent = "Đang phân tích...";
  } else {
    hideEl(loadingOverlay);
    btnPredict.disabled = false;
    btnPredict.textContent = "Bắt đầu phân tích";
  }
}

// ── Drag & Drop ───────────────────────────────────────────────────
["dragenter", "dragover", "dragleave", "drop"].forEach(evt => {
  dropArea.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); });
  document.body.addEventListener(evt, e => { e.preventDefault(); });
});

["dragenter", "dragover"].forEach(evt =>
  dropArea.addEventListener(evt, () => dropArea.classList.add("highlight"))
);

["dragleave", "drop"].forEach(evt =>
  dropArea.addEventListener(evt, () => dropArea.classList.remove("highlight"))
);

dropArea.addEventListener("drop", e => {
  const dt = e.dataTransfer;
  const file = dt.files[0];
  if (file && file.type.startsWith("image/")) handleFile(file);
});

// ── File Input Change ─────────────────────────────────────────────
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

// ── Handle File ───────────────────────────────────────────────────
function handleFile(file) {
  currentFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    imagePreview.src = e.target.result;
    hideEl(dropArea);
    showEl(previewContainer);
    hideEl(resultSection);
  };
  reader.readAsDataURL(file);
}

// ── Reset ─────────────────────────────────────────────────────────
function resetApp() {
  currentFile = null;
  fileInput.value = "";
  imagePreview.src = "";
  if (aiProcessedPreview) aiProcessedPreview.src = "";
  hideEl(previewContainer);
  hideEl(resultSection);
  hideEl(mockWarning);
  showEl(dropArea);
  if (confidenceBar) confidenceBar.style.width = "0%";
}

btnReset.addEventListener("click", resetApp);

// ── Predict ───────────────────────────────────────────────────────
btnPredict.addEventListener("click", async () => {
  if (!currentFile) return;

  setLoading(true);
  hideEl(resultSection);

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const res = await fetch(API_URL, { method: "POST", body: formData });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      throw new Error(err.error || `Lỗi server: ${res.status}`);
    }

    const data = await res.json();

    if (!data.success) throw new Error(data.error || "Dự đoán thất bại!");

    displayResult(data);

  } catch (err) {
    displayError(err.message);
  } finally {
    setLoading(false);
  }
});

// ── Display Result ────────────────────────────────────────────────
function displayResult(data) {
  const { result, confidence, is_mock, processed_image } = data;
  const meta = CLASS_META[result] || { icon: "🏷️", color: "#6366f1" };

  // Set AI processed image view
  if (aiProcessedPreview && processed_image) {
    aiProcessedPreview.src = processed_image;
  }

  // Result icon & name
  const iconEl = document.getElementById("result-icon");
  if (iconEl) iconEl.textContent = meta.icon;

  resultName.textContent = result;
  resultName.style.color = meta.color;

  // Confidence bar
  const pct = Math.round(confidence);
  if (confidenceBar) {
    confidenceBar.style.background = meta.color;
    // Animate bar
    setTimeout(() => { confidenceBar.style.width = pct + "%"; }, 50);
  }
  if (confidenceValue) confidenceValue.textContent = pct + "%";
  resultConfidence.textContent = `${confidence.toFixed(1)}%`;

  // Confidence color hint
  const confEl = document.getElementById("result-confidence");
  if (confidence >= 85) confEl.style.color = "#22c55e";
  else if (confidence >= 65) confEl.style.color = "#f97316";
  else confEl.style.color = "#ef4444";

  // Mock warning
  if (is_mock) showEl(mockWarning);
  else hideEl(mockWarning);

  showEl(resultSection);
  resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Display Error ─────────────────────────────────────────────────
function displayError(message) {
  const iconEl = document.getElementById("result-icon");
  if (iconEl) iconEl.textContent = "❌";

  resultName.textContent = "Đã xảy ra lỗi";
  resultName.style.color = "#ef4444";
  resultConfidence.textContent = "—";

  if (confidenceBar) { confidenceBar.style.width = "0%"; }
  if (confidenceValue) confidenceValue.textContent = "—";

  mockWarning.textContent = `⚠️ ${message}`;
  mockWarning.style.background = "#fef2f2";
  mockWarning.style.color = "#dc2626";
  mockWarning.style.borderColor = "#fecaca";
  showEl(mockWarning);
  showEl(resultSection);
}

// ── Paste from clipboard ──────────────────────────────────────────
document.addEventListener("paste", e => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) { handleFile(file); break; }
    }
  }
});
