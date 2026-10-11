"""Semantic similarity with Sentence-BERT (catches paraphrasing)."""
from functools import lru_cache

import numpy as np

from .config import MODELS_DIR, SBERT_MODEL


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(SBERT_MODEL, cache_folder=str(MODELS_DIR))


def embed(sentences, show_progress=False):
    """Encode raw sentences into unit-length 384-dim vectors.
    Raw text is used on purpose: transformers rely on stopwords, word order
    and inflection, so classic preprocessing would hurt them."""
    return get_model().encode(
        list(sentences), batch_size=64, normalize_embeddings=True,
        show_progress_bar=show_progress, convert_to_numpy=True,
    ).astype(np.float32)


def cosine_matrix(a, b):
    """Vectors are normalised, so the dot product equals cosine similarity."""
    return a @ b.T
