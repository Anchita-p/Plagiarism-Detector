"""Central configuration: paths, model name and detection thresholds."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"          # reference documents (one .txt per source)
SAMPLES_DIR = DATA_DIR / "samples"        # demo suspicious documents
EVAL_DIR = DATA_DIR / "eval"              # labeled evaluation data
MODELS_DIR = ROOT / "models"              # downloaded SBERT model + saved index
INDEX_PATH = MODELS_DIR / "corpus_index.pkl"

# Sentence-BERT model: small (80 MB), fast on CPU, 384-dim embeddings
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Word n-gram size used for lexical (copy-paste) overlap
NGRAM_N = 3

# How many semantically closest corpus sentences to re-score per query sentence
TOP_K_CANDIDATES = 5

# Weights for fusing the three similarity scores into one
FUSION_WEIGHTS = {"lexical": 0.25, "tfidf": 0.25, "semantic": 0.50}

# Sentence-level decision thresholds (fixed defaults; see results/evaluation.md for how they perform)
COPY_THRESHOLD = 0.50        # n-gram containment >= this  -> "Copied"
PARAPHRASE_THRESHOLD = 0.75  # SBERT cosine >= this         -> "Paraphrased"

# Document verdict from the share of words that sit in flagged sentences
SUSPICIOUS_PERCENT = 15
PLAGIARIZED_PERCENT = 40


def load_env_file(path=ROOT / ".env"):
    """Read KEY=value lines from .env into the environment (real environment variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env_file()
