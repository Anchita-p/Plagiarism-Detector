"""Text preprocessing: cleaning, sentence segmentation, tokenization,
stopword removal and lemmatization."""
import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "“”‘’—–")
MIN_SENTENCE_WORDS = 5


def clean_text(text):
    """Normalise unicode, quotes and whitespace; drop citation marks like [12]."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = re.sub(r"\[\d+\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    """Sentence segmentation with NLTK's Punkt model; very short fragments are dropped."""
    sentences = nltk.sent_tokenize(clean_text(text))
    return [s for s in sentences if len(s.split()) >= MIN_SENTENCE_WORDS]


def tokenize(text):
    """Lowercase word tokens with punctuation and numbers removed."""
    text = text.lower().translate(_PUNCT_TABLE)
    return [t for t in nltk.word_tokenize(text) if t.isalpha()]


def lemmatize(tokens):
    """Reduce words to their dictionary form: 'studies' -> 'study', 'algorithms' -> 'algorithm'."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


def normalize(text, remove_stopwords=True):
    """Full pipeline used by the lexical and TF-IDF methods:
    tokenize -> (stopword removal) -> lemmatize."""
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return lemmatize(tokens)
