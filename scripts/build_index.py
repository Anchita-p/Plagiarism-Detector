"""Pre-compute the searchable index for the reference corpus.

Splits every corpus document into sentences, encodes them with SBERT,
builds n-gram sets and fits TF-IDF, then saves everything to models/.
Run again whenever you add or remove files in data/corpus/.
"""
import time

from src.config import CORPUS_DIR, INDEX_PATH
from src.detector import Reference
from src.file_reader import read_path


def main():
    files = sorted(p for p in CORPUS_DIR.iterdir() if p.suffix.lower() in {".txt", ".pdf", ".docx"})
    if not files:
        raise SystemExit(f"No documents in {CORPUS_DIR}. Run: python -m scripts.build_corpus")
    documents = {p.stem.replace("_", " "): read_path(p) for p in files}
    print(f"Indexing {len(documents)} documents ...")
    start = time.time()
    reference = Reference.from_documents(documents, show_progress=True)
    reference.save(INDEX_PATH)
    print(f"Indexed {len(reference)} sentences in {time.time() - start:.0f}s -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
