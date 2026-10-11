"""Plagiarism detection engine.

For every sentence of the suspicious document we
  1. retrieve candidate source sentences (closest by SBERT and by TF-IDF),
  2. score each candidate with three methods: n-gram containment,
     TF-IDF cosine and SBERT cosine,
  3. fuse the scores and label the sentence Copied / Paraphrased / Original.
The document score is the share of words that sit in flagged sentences.
"""
import pickle
from dataclasses import dataclass, field

import numpy as np

from . import config
from .lexical import containment, text_ngrams
from .preprocess import split_sentences
from .semantic import cosine_matrix, embed
from .tfidf import TfidfModel, features as tfidf_features

COPIED, PARAPHRASED, ORIGINAL = "Copied", "Paraphrased", "Original"


@dataclass
class SentenceResult:
    sentence: str
    label: str
    source: str = None
    source_sentence: str = None
    lexical: float = 0.0
    tfidf: float = 0.0
    semantic: float = 0.0
    fused: float = 0.0

    @property
    def words(self):
        return len(self.sentence.split())


@dataclass
class Report:
    sentences: list
    plagiarism_percent: float
    verdict: str
    method_scores: dict = field(default_factory=dict)  # document-level score per method
    sources: dict = field(default_factory=dict)        # source name -> % of words matched


class Reference:
    """Searchable collection of source sentences with precomputed features."""

    def __init__(self, sentences, sources, embeddings, ngram_sets, features):
        self.sentences = sentences
        self.sources = sources
        self.embeddings = embeddings
        self.ngram_sets = ngram_sets
        self.features = features
        self.tfidf = TfidfModel(features)

    @staticmethod
    def _encode(documents, show_progress=False):
        """Split documents into sentences and compute every feature the detector needs."""
        sentences, sources = [], []
        for name, text in documents.items():
            for sent in split_sentences(text):
                sentences.append(sent)
                sources.append(name)
        if not sentences:
            raise ValueError("The document contains no usable sentences (5+ words).")
        embeddings = embed(sentences, show_progress=show_progress)
        return (sentences, sources, embeddings,
                [text_ngrams(s) for s in sentences], [tfidf_features(s) for s in sentences])

    @classmethod
    def from_documents(cls, documents, show_progress=False):
        """documents: {source_name: full_text}"""
        return cls(*cls._encode(documents, show_progress))

    def add_documents(self, documents):
        """Add sources to the index; only the new sentences are encoded.
        A document with an existing name replaces the old version."""
        sentences, sources, embeddings, ngram_sets, features = self._encode(documents)
        keep = [i for i, src in enumerate(self.sources) if src not in documents]
        self.sentences = [self.sentences[i] for i in keep] + sentences
        self.sources = [self.sources[i] for i in keep] + sources
        self.embeddings = np.vstack([self.embeddings[keep], embeddings])
        self.ngram_sets = [self.ngram_sets[i] for i in keep] + ngram_sets
        self.features = [self.features[i] for i in keep] + features
        self.tfidf = TfidfModel(self.features)
        return len(sentences)

    def document_counts(self):
        """{source name: number of indexed sentences}"""
        counts = {}
        for src in self.sources:
            counts[src] = counts.get(src, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[0].lower()))

    def save(self, path=config.INDEX_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path=config.INDEX_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.sentences)


def fuse(lexical, tfidf, semantic):
    w = config.FUSION_WEIGHTS
    return w["lexical"] * lexical + w["tfidf"] * tfidf + w["semantic"] * semantic


def label_sentence(lexical, semantic):
    if lexical >= config.COPY_THRESHOLD:
        return COPIED
    if semantic >= config.PARAPHRASE_THRESHOLD:
        return PARAPHRASED
    return ORIGINAL


def verdict_for(percent):
    if percent >= config.PLAGIARIZED_PERCENT:
        return "Plagiarized"
    if percent >= config.SUSPICIOUS_PERCENT:
        return "Suspicious"
    return "Original"


def detect(text, reference, top_k=config.TOP_K_CANDIDATES):
    """Check a suspicious text against a Reference and return a Report."""
    query = split_sentences(text)
    if not query:
        return Report([], 0.0, "Original")

    sem_matrix = cosine_matrix(embed(query), reference.embeddings)
    tfidf_matrix = reference.tfidf.similarity(query)
    k = min(top_k, len(reference))

    results = []
    for i, sent in enumerate(query):
        # Candidate retrieval: union of top-k by meaning and top-k by keywords
        cands = set(np.argpartition(-sem_matrix[i], k - 1)[:k])
        cands |= set(np.argpartition(-tfidf_matrix[i], k - 1)[:k])
        q_ngrams = text_ngrams(sent)

        best, best_lex, best_tfidf, best_sem = None, 0.0, 0.0, 0.0
        for j in cands:
            lex = containment(q_ngrams, reference.ngram_sets[j])
            tf, sem = float(tfidf_matrix[i, j]), float(sem_matrix[i, j])
            score = fuse(lex, tf, sem)
            if best is None or score > best[0]:
                best = (score, j, lex, tf, sem)
            best_lex, best_tfidf, best_sem = max(best_lex, lex), max(best_tfidf, tf), max(best_sem, sem)

        fused, j, lex, tf, sem = best
        label = label_sentence(best_lex, sem)
        results.append(SentenceResult(
            sentence=sent, label=label,
            source=reference.sources[j] if label != ORIGINAL else None,
            source_sentence=reference.sentences[j],
            lexical=best_lex, tfidf=best_tfidf, semantic=sem, fused=fused,
        ))

    total_words = sum(r.words for r in results)
    flagged = [r for r in results if r.label != ORIGINAL]
    percent = 100 * sum(r.words for r in flagged) / total_words

    sources = {}
    for r in flagged:
        sources[r.source] = sources.get(r.source, 0) + 100 * r.words / total_words

    def weighted(attr):
        return sum(getattr(r, attr) * r.words for r in results) / total_words

    method_scores = {m: weighted(m) for m in ("lexical", "tfidf", "semantic", "fused")}
    return Report(results, percent, verdict_for(percent), method_scores,
                  dict(sorted(sources.items(), key=lambda x: -x[1])))


def compare_documents(suspicious_text, source_text, source_name="Source document"):
    """Document-vs-document mode."""
    return detect(suspicious_text, Reference.from_documents({source_name: source_text}))
