"""Lexical similarity using word n-grams (catches copy-paste and light edits)."""
from .config import NGRAM_N
from .preprocess import normalize


def ngrams(tokens, n=NGRAM_N):
    """Set of word n-grams, e.g. n=3: ('neural', 'network', 'learn')."""
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a, b):
    """|A ∩ B| / |A ∪ B| — symmetric overlap of two n-gram sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a, b):
    """|A ∩ B| / |A| — how much of the suspicious text A appears inside source B.
    Better than Jaccard when the suspicious text is shorter than the source."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def text_ngrams(text, n=NGRAM_N):
    # Stopwords are kept so that n-grams preserve the original word sequence
    return ngrams(normalize(text, remove_stopwords=False), n)
