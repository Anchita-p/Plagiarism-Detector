"""Command-line plagiarism checker.

  python check.py essay.txt                     # check against the reference corpus
  python check.py essay.txt --against source.pdf   # compare two documents
"""
import argparse
import textwrap

from src.config import INDEX_PATH
from src.detector import ORIGINAL, Reference, compare_documents, detect
from src.file_reader import read_path


def print_report(report):
    print(f"\nPlagiarism: {report.plagiarism_percent:.1f}%   Verdict: {report.verdict}")
    print("Document-level scores: " + ", ".join(f"{k}={v:.2f}" for k, v in report.method_scores.items()))
    if report.sources:
        print("\nMatched sources:")
        for name, pct in report.sources.items():
            print(f"  {pct:5.1f}%  {name}")
    print("\nSentence analysis:")
    for i, r in enumerate(report.sentences, 1):
        print(f"\n[{i}] {r.label.upper()}  (n-gram {r.lexical:.2f} | tf-idf {r.tfidf:.2f} | sbert {r.semantic:.2f})")
        print(textwrap.indent(textwrap.fill(r.sentence, 90), "    "))
        if r.label != ORIGINAL:
            print(textwrap.indent(textwrap.fill(f"<- {r.source}: {r.source_sentence}", 90), "    "))


def main():
    parser = argparse.ArgumentParser(description="NLP plagiarism detector")
    parser.add_argument("file", help="suspicious document (.txt, .pdf, .docx)")
    parser.add_argument("--against", help="compare with this single document instead of the corpus")
    args = parser.parse_args()

    text = read_path(args.file)
    if args.against:
        report = compare_documents(text, read_path(args.against), args.against)
    else:
        if not INDEX_PATH.exists():
            raise SystemExit("Corpus index missing. Run: python -m scripts.build_index")
        report = detect(text, Reference.load())
    print_report(report)


if __name__ == "__main__":
    main()
