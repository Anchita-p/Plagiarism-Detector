"""Download the NLTK data packages the project needs (run once)."""
import ssl

import certifi
import nltk

PACKAGES = [
    "punkt", "punkt_tab",                                            # sentence + word tokenizer
    "stopwords",                                                     # English stopword list
    "wordnet", "omw-1.4",                                            # lemmatizer dictionary
]

if __name__ == "__main__":
    # python.org builds on macOS ship without CA certificates; use certifi's bundle
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    failed = [pkg for pkg in PACKAGES if not nltk.download(pkg, quiet=True)]
    if failed:
        raise SystemExit(f"Failed to download NLTK packages: {failed}")
    print("NLTK data ready.")
