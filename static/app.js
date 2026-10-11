// Frontend logic: tabs, inputs, API calls and rendering of the report.

const $ = (id) => document.getElementById(id);
const RING = 2 * Math.PI * 52;

const METHODS = [
  ["lexical", "N-gram overlap", "Exact 3-word sequences: copy-paste"],
  ["tfidf", "TF-IDF cosine", "Shared distinctive keywords"],
  ["semantic", "Sentence-BERT", "Same meaning, even in different words"],
  ["fused", "Combined score", "0.25 n-gram + 0.25 TF-IDF + 0.5 SBERT"],
];

const EXPLAIN = {
  Original: "Little or no overlap with the sources. The text appears to be original work.",
  Suspicious: "Part of the text closely matches the sources. Review the highlighted sentences.",
  Plagiarized: "A large share of the text is copied or paraphrased from the sources.",
};

// ---------- helpers ----------
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function toast(msg, isError = false) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 4500);
}

async function api(url, options = {}, message = "Analysing sentences…") {
  $("overlay-msg").textContent = message;
  $("overlay").hidden = false;
  try {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({ error: "Unexpected server response." }));
    if (!res.ok) throw new Error(data.error || "Something went wrong.");
    return data;
  } finally {
    $("overlay").hidden = true;
  }
}

function wordCount(text) {
  return (text.trim().match(/\S+/g) || []).length;
}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + tab.dataset.tab));
    $("results").hidden = true;
  })
);

// ---------- file inputs ----------
function bindFile(inputId, textareaId) {
  const input = $(inputId);
  input.addEventListener("change", () => {
    const f = input.files[0];
    $(inputId + "-name").textContent = f ? f.name : "";
    if (f && textareaId) {
      $(textareaId).value = "";
      $(textareaId).placeholder = `Using uploaded file: ${f.name}`;
      updateCount();
    }
  });
  if (textareaId) {
    $(textareaId).addEventListener("input", () => {
      if ($(textareaId).value && input.files.length) {
        input.value = "";
        $(inputId + "-name").textContent = "";
      }
    });
  }
}
bindFile("check-file", "check-text");
bindFile("cmp-a-file", "cmp-a-text");
bindFile("cmp-b-file", "cmp-b-text");
bindFile("src-file");

function updateCount() {
  const n = wordCount($("check-text").value);
  $("check-count").textContent = `${n} word${n === 1 ? "" : "s"}`;
}
$("check-text").addEventListener("input", updateCount);

function addInput(form, prefix, textareaId, fileId) {
  const file = $(fileId).files[0];
  if (file) form.append(prefix + "_file", file);
  else form.append(prefix + "_text", $(textareaId).value);
  return Boolean(file) || $(textareaId).value.trim().length > 0;
}

// ---------- samples ----------
async function loadSamples() {
  const names = await fetch("/api/samples").then((r) => r.json());
  $("samples").innerHTML = "";
  names.forEach((name) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = name.replace(/^\d+_/, "").replace(/_/g, " ");
    chip.addEventListener("click", async () => {
      const { text } = await fetch(`/api/samples/${encodeURIComponent(name)}`).then((r) => r.json());
      $("check-file").value = "";
      $("check-file-name").textContent = "";
      $("check-text").value = text;
      updateCount();
    });
    $("samples").append(chip);
  });
}

// ---------- source library ----------
let libraryNames = [];
function renderLibrary(corpus, highlight) {
  libraryNames = corpus.documents;
  $("corpus-status").textContent = `${corpus.documents.length} sources · ${corpus.sentences.toLocaleString()} sentences`;
  $("library-title").textContent = `Library: ${corpus.documents.length} sources`;
  const filter = $("library-filter").value.toLowerCase();
  $("library").innerHTML = corpus.documents
    .filter((d) => d.name.toLowerCase().includes(filter))
    .map((d) => `<li class="${d.name === highlight ? "new" : ""}"><span>${escapeHtml(d.name)}</span><span>${d.sentences} sent.</span></li>`)
    .join("");
}
$("library-filter").addEventListener("input", () =>
  renderLibrary({ documents: libraryNames, sentences: libraryNames.reduce((a, d) => a + d.sentences, 0) })
);

async function loadCorpus() {
  renderLibrary(await fetch("/api/corpus").then((r) => r.json()));
}

$("src-upload-btn").addEventListener("click", async () => {
  const file = $("src-file").files[0];
  if (!file) return toast("Choose a file first.", true);
  const form = new FormData();
  form.append("file", file);
  try {
    const data = await api("/api/corpus/upload", { method: "POST", body: form }, `Indexing ${file.name}…`);
    renderLibrary(data, data.name);
    $("src-file").value = "";
    $("src-file-name").textContent = "";
    toast(`Added "${data.name}" (${data.added_sentences} sentences).`);
  } catch (e) {
    toast(e.message, true);
  }
});

async function addWikipedia() {
  const topic = $("wiki-topic").value.trim();
  if (!topic) return toast("Type a topic first.", true);
  try {
    const data = await api("/api/corpus/wikipedia", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic }),
    }, `Fetching "${topic}" from Wikipedia…`);
    renderLibrary(data, data.name);
    $("wiki-topic").value = "";
    toast(`Added "${data.name}" (${data.added_sentences} sentences).`);
  } catch (e) {
    toast(e.message, true);
  }
}
$("wiki-btn").addEventListener("click", addWikipedia);
$("wiki-topic").addEventListener("keydown", (e) => e.key === "Enter" && addWikipedia());

// ---------- checks ----------
$("check-btn").addEventListener("click", async () => {
  const form = new FormData();
  if (!addInput(form, "suspicious", "check-text", "check-file")) return toast("Paste some text or upload a file.", true);
  try {
    renderReport(await api("/api/check", { method: "POST", body: form }), "the source library");
  } catch (e) {
    toast(e.message, true);
  }
});

$("web-btn").addEventListener("click", async () => {
  const form = new FormData();
  if (!addInput(form, "suspicious", "check-text", "check-file")) return toast("Paste some text or upload a file.", true);
  try {
    const data = await api("/api/check_web", { method: "POST", body: form },
      "Searching the web and reading matching pages… this can take up to a minute.");
    renderReport(data, `${data.pages} web page${data.pages === 1 ? "" : "s"} found with ${data.searches} searches`);
    if (data.notice) toast(data.notice);
  } catch (e) {
    toast(e.message, true);
  }
});

$("compare-btn").addEventListener("click", async () => {
  const form = new FormData();
  const a = addInput(form, "suspicious", "cmp-a-text", "cmp-a-file");
  const b = addInput(form, "source", "cmp-b-text", "cmp-b-file");
  if (!a || !b) return toast("Provide both documents.", true);
  try {
    renderReport(await api("/api/compare", { method: "POST", body: form }), "the source document");
  } catch (e) {
    toast(e.message, true);
  }
});

// ---------- report ----------
let sentences = [];

function renderReport(r, against) {
  sentences = r.sentences;
  $("results").hidden = false;

  if (!r.sentences.length) {
    $("res-percent").textContent = "–";
    $("res-verdict").textContent = "No sentences";
    $("res-verdict").className = "verdict";
    $("res-explain").textContent = "No sentences with 5 or more words were found in the input.";
  } else {
    $("res-percent").textContent = `${Math.round(r.percent)}%`;
    $("res-verdict").textContent = r.verdict;
    $("res-verdict").className = `verdict ${r.verdict}`;
    $("res-explain").textContent = `${EXPLAIN[r.verdict]} Compared against ${against}.`;
  }
  $("res-copied").textContent = `${Math.round(r.copied_percent)}%`;
  $("res-para").textContent = `${Math.round(r.paraphrased_percent)}%`;
  $("res-orig").textContent = `${Math.round(100 - r.copied_percent - r.paraphrased_percent)}%`;

  const c = (r.copied_percent / 100) * RING;
  const p = (r.paraphrased_percent / 100) * RING;
  $("ring-copied").style.strokeDasharray = `0 ${RING}`;
  $("ring-para").style.strokeDasharray = `0 ${RING}`;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    $("ring-copied").style.strokeDasharray = `${c} ${RING}`;
    $("ring-para").style.strokeDasharray = `${p} ${RING}`;
    $("ring-para").style.strokeDashoffset = `${-c}`;
  }));

  $("res-doc").innerHTML = r.sentences
    .map((s, i) => `<span class="s ${s.label}" data-i="${i}" tabindex="${s.label === "Original" ? -1 : 0}">${escapeHtml(s.text)}</span>`)
    .join(" ");

  $("res-sources").innerHTML = r.sources.length
    ? r.sources.map((s) => bar(s.name, s.percent, `${Math.round(s.percent)}%`)).join("")
    : `<li class="empty">No matching sources found.</li>`;

  $("res-methods").innerHTML = METHODS
    .map(([key, name, note]) => bar(name, r.method_scores[key] * 100, (r.method_scores[key] ?? 0).toFixed(2), note))
    .join("");
  requestAnimationFrame(() => requestAnimationFrame(() =>
    document.querySelectorAll("#results .fill").forEach((f) => (f.style.width = f.dataset.w + "%"))
  ));

  $("res-table").innerHTML = r.sentences.map((s, i) => `
    <tr><td class="num">${i + 1}</td><td>${escapeHtml(s.text)}</td>
    <td><span class="pill ${s.label}">${s.label}</span></td>
    <td class="num">${s.lexical.toFixed(2)}</td><td class="num">${s.tfidf.toFixed(2)}</td><td class="num">${s.semantic.toFixed(2)}</td></tr>`).join("");

  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function bar(name, pct, value, note, color) {
  const w = Math.max(0, Math.min(100, pct || 0));
  return `<li><div class="bar-head"><span title="${escapeHtml(name)}">${escapeHtml(name)}</span><b>${value}</b></div>
    <div class="track"><div class="fill" data-w="${w}" ${color ? `style="background:${color}"` : ""}></div></div>
    ${note ? `<span class="note">${escapeHtml(note)}</span>` : ""}</li>`;
}

// ---------- tooltip on highlighted sentences ----------
function showTip(el) {
  const s = sentences[el.dataset.i];
  if (!s || s.label === "Original") return;
  const tip = $("tooltip");
  tip.innerHTML = `<div class="t-head">${s.label} from ${escapeHtml(s.source)}</div>
    <div class="t-src">“${escapeHtml(s.source_sentence)}”</div>
    <div class="t-scores">N-gram ${s.lexical.toFixed(2)} · TF-IDF ${s.tfidf.toFixed(2)} · SBERT ${s.semantic.toFixed(2)}</div>`;
  tip.hidden = false;
  const r = el.getBoundingClientRect();
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = Math.min(Math.max(8, r.left), window.innerWidth - tw - 8);
  let top = r.bottom + 8;
  if (top + th > window.innerHeight - 8) top = Math.max(8, r.top - th - 8);
  tip.style.left = left + "px";
  tip.style.top = top + "px";
}
function hideTip() {
  $("tooltip").hidden = true;
  document.querySelectorAll(".doc .s.focus").forEach((e) => e.classList.remove("focus"));
}
$("res-doc").addEventListener("mouseover", (e) => e.target.matches(".s") && showTip(e.target));
$("res-doc").addEventListener("mouseout", (e) => e.target.matches(".s") && hideTip());
$("res-doc").addEventListener("focusin", (e) => e.target.matches(".s") && showTip(e.target));
$("res-doc").addEventListener("focusout", hideTip);
$("res-doc").addEventListener("click", (e) => {
  if (!e.target.matches(".s")) return;
  hideTip();
  e.target.classList.add("focus");
  showTip(e.target);
});
window.addEventListener("scroll", hideTip, { passive: true });

loadSamples();
loadCorpus();
