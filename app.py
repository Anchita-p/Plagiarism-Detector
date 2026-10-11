"""Flask web server: serves the HTML/CSS/JS frontend and a small JSON API.

Run with:  python app.py   then open http://127.0.0.1:5000
"""
import threading
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request
from werkzeug.utils import secure_filename

from scripts.build_corpus import fetch as fetch_wikipedia, safe_name
from src.config import CORPUS_DIR, INDEX_PATH, SAMPLES_DIR
from src import web_search
from src.detector import COPIED, PARAPHRASED, Reference, Report, compare_documents, detect
from src.file_reader import read_text

ALLOWED = {".txt", ".pdf", ".docx"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB uploads
lock = threading.Lock()   # the corpus index is shared between requests
reference = None


def source_name(filename):
    return Path(filename).stem.replace("_", " ")


def get_text(prefix):
    """Text from an uploaded file (<prefix>_file) or a text field (<prefix>_text)."""
    upload = request.files.get(f"{prefix}_file")
    if upload and upload.filename:
        if Path(upload.filename).suffix.lower() not in ALLOWED:
            raise ValueError("Only .txt, .pdf and .docx files are supported.")
        return read_text(upload.read(), upload.filename)
    return request.form.get(f"{prefix}_text", "")


def report_json(report):
    words = sum(s.words for s in report.sentences) or 1

    def share(label):
        return 100 * sum(s.words for s in report.sentences if s.label == label) / words

    return {
        "percent": round(report.plagiarism_percent, 1),
        "verdict": report.verdict,
        "copied_percent": round(share(COPIED), 1),
        "paraphrased_percent": round(share(PARAPHRASED), 1),
        "method_scores": {k: round(v, 3) for k, v in report.method_scores.items()},
        "sources": [{"name": k, "percent": round(v, 1)} for k, v in report.sources.items()],
        "sentences": [{
            "text": s.sentence, "label": s.label, "source": s.source,
            "source_sentence": s.source_sentence,
            "lexical": round(s.lexical, 3), "tfidf": round(s.tfidf, 3), "semantic": round(s.semantic, 3),
        } for s in report.sentences],
    }


def corpus_json():
    if reference is None:
        return {"documents": [], "sentences": 0}
    counts = reference.document_counts()
    return {"documents": [{"name": k, "sentences": v} for k, v in counts.items()],
            "sentences": len(reference)}


def add_to_corpus(name, text, filename):
    """Save a new source into data/corpus and index it."""
    with lock:
        global reference
        documents = {name: text}
        if reference is None:
            reference = Reference.from_documents(documents)
            added = len(reference)
        else:
            added = reference.add_documents(documents)
        (CORPUS_DIR / filename).write_text(text, encoding="utf-8")
        reference.save(INDEX_PATH)
    return {"name": name, "added_sentences": added, **corpus_json()}


@app.errorhandler(ValueError)
def bad_request(err):
    return jsonify(error=str(err)), 400


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/corpus")
def corpus():
    return jsonify(corpus_json())


@app.get("/api/samples")
def samples():
    return jsonify([p.stem for p in sorted(SAMPLES_DIR.glob("*.txt"))])


@app.get("/api/samples/<name>")
def sample(name):
    path = SAMPLES_DIR / f"{secure_filename(name)}.txt"
    if not path.exists():
        abort(404)
    return jsonify(text=path.read_text(encoding="utf-8"))


@app.post("/api/check")
def check():
    text = get_text("suspicious")
    if not text.strip():
        raise ValueError("Please paste some text or upload a file.")
    if reference is None:
        raise ValueError("The source library is empty. Add sources in the Sources tab first.")
    with lock:
        report = detect(text, reference)
    return jsonify(report_json(report))


@app.get("/api/web/status")
def web_status():
    return jsonify(enabled=web_search.is_configured())


@app.post("/api/check_web")
def check_web():
    """Search the whole web (Google via Serper) for sources, then run the normal detector."""
    text = get_text("suspicious")
    if not text.strip():
        raise ValueError("Please paste some text or upload a file.")
    try:
        documents, searches = web_search.find_web_sources(text)
    except web_search.WebSearchError as err:
        raise ValueError(str(err))
    if not documents:
        data = report_json(Report([], 0.0, "Original"))
        data.update(searches=searches, pages=0, notice="No web pages containing these sentences were found.")
        return jsonify(data)
    with lock:   # the SBERT model is shared between requests
        report = detect(text, Reference.from_documents(documents))
    return jsonify({**report_json(report), "searches": searches, "pages": len(documents)})


@app.post("/api/compare")
def compare():
    suspicious, source = get_text("suspicious"), get_text("source")
    if not suspicious.strip() or not source.strip():
        raise ValueError("Please provide both documents.")
    return jsonify(report_json(compare_documents(suspicious, source, "Source document")))


@app.post("/api/corpus/upload")
def upload_source():
    upload = request.files.get("file")
    if not upload or not upload.filename:
        raise ValueError("Choose a file to add.")
    filename = secure_filename(upload.filename)
    if Path(filename).suffix.lower() not in ALLOWED:
        raise ValueError("Only .txt, .pdf and .docx files are supported.")
    text = read_text(upload.read(), filename)
    return jsonify(add_to_corpus(source_name(filename), text, Path(filename).stem + ".txt"))


@app.post("/api/corpus/wikipedia")
def add_wikipedia():
    topic = (request.json or {}).get("topic", "").strip()
    if not topic:
        raise ValueError("Type a Wikipedia topic.")
    try:
        title, text = fetch_wikipedia(topic)
    except Exception:
        raise ValueError("Could not reach Wikipedia. Check your internet connection.")
    if len(text.split()) < 50:
        raise ValueError(f'No Wikipedia article found for "{topic}".')
    filename = safe_name(title)
    return jsonify(add_to_corpus(source_name(filename), text, filename))


def startup():
    global reference
    if INDEX_PATH.exists():
        print("Loading source index ...")
        reference = Reference.load()
    print("Loading Sentence-BERT model and NLTK data ...")
    # Run one tiny check now, in the main thread: the models and NLTK's lazily
    # loaded data (WordNet, Punkt) must not be first loaded inside a request thread.
    warm = "This short sentence warms up every part of the detector."
    detect(warm, reference) if reference is not None else compare_documents(warm, warm)


if __name__ == "__main__":
    startup()
    print("\nReady: open http://127.0.0.1:5000 in your browser\n")
    app.run(host="127.0.0.1", port=5000, threaded=True)
