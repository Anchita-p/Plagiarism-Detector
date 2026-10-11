"""Extract plain text from uploaded .txt, .pdf and .docx files."""
import io
from pathlib import Path


def read_text(data, filename):
    """data: raw bytes of the file; filename decides the parser."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if suffix == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")


def read_path(path):
    path = Path(path)
    return read_text(path.read_bytes(), path.name)
