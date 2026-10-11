#!/usr/bin/env bash
# One-time setup: virtual environment, Python packages, NLTK data, corpus index.
set -e
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m scripts.setup_nltk

# The Wikipedia corpus is already included in data/corpus.
# To re-download it:  python -m scripts.build_corpus
python -m scripts.build_index

echo
echo "Setup complete. Start the app with:"
echo "  source .venv/bin/activate && python app.py"
echo "then open http://127.0.0.1:5000"
