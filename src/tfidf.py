"""TF-IDF vector space model with cosine similarity."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import normalize


def features(text):
    """Lemmatized content words plus word bigrams as TF-IDF features."""
    tokens = normalize(text, remove_stopwords=True)
    return tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def _identity(feature_list):
    # Features are computed once per sentence and stored, so refitting is fast
    return feature_list


class TfidfModel:
    def __init__(self, feature_lists):
        """Learn vocabulary and IDF weights from the reference sentences' features."""
        self.vectorizer = TfidfVectorizer(analyzer=_identity, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(feature_lists)

    def similarity(self, queries):
        """Cosine similarity matrix: rows = query sentences, columns = reference sentences."""
        return cosine_similarity(self.vectorizer.transform([features(q) for q in queries]), self.matrix)
