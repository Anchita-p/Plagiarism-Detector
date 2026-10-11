# NLP Plagiarism Detector

Detects **copy-paste, lightly edited and paraphrased** plagiarism in text documents by
combining three NLP techniques:

| Method | Catches | Technique |
|---|---|---|
| Word n-gram containment | Copy-paste, light edits | Tokenization, lemmatization, n-grams |
| TF-IDF cosine similarity | Reworded text with the same keywords | Vector space model |
| Sentence-BERT similarity | Paraphrasing / synonyms | Transformer sentence embeddings |

Comparison is done **sentence by sentence**, so the app shows exactly which sentences were
copied or paraphrased and from which source.

For a full explanation of how everything works and how it maps to the NLP syllabus, read
**[PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md)**.

## Features

1. **Check document**: any pasted text or uploaded file is compared with every sentence in the
   source library (102 Wikipedia articles on common assignment topics to start with).
2. **Compare two documents**: e.g. two students' assignments, or an essay and a suspected source.
3. **Source library**: grow the library from the browser. Upload your own source documents
   (notes, textbook chapters, previous submissions) or fetch any Wikipedia article by topic.
   New sources are indexed in a few seconds.

4. **Search the web** (optional): look for sources anywhere on the internet instead of only the
   library. See [Web search setup](#web-search-setup-optional).

Input can be pasted text or an uploaded `.txt`, `.pdf` or `.docx` file.

## Setup (one time)

```bash
cd ~/Desktop/Plagiarism-Detector
./setup.sh
```

This creates a virtual environment, installs the packages, downloads NLTK data and builds the
corpus index. The first run also downloads the SBERT model (~90 MB) into `models/`.

**On Windows** (PowerShell, Python 3.10+ installed):

```powershell
cd Plagiarism-Detector
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.setup_nltk
python -m scripts.build_index
python app.py
```

**Moving to another laptop:** copy the folder *without* `.venv/` and `models/` (they are
machine-specific and get recreated by the setup), then run the setup again.

## Run

```bash
source .venv/bin/activate

# Web app: then open http://127.0.0.1:5000 in your browser
python app.py

# Command line
python check.py data/samples/3_paraphrase.txt
python check.py my_essay.pdf --against source.docx

# Evaluation on the labeled Clough & Stevenson corpus (dataset not included, see below)
python -m scripts.evaluate
```

### Evaluation dataset (not included)

The evaluation uses the Clough & Stevenson plagiarism corpus (University of Sheffield). It is not
redistributed in this repository. To run `scripts.evaluate`, download the corpus from its
official page (search for "Clough Stevenson plagiarised short answers corpus") and place its
files in `data/eval/clough/`. Everything else in the app works without it. The results of our
run are in `results/evaluation.md`.

## Web search setup (optional)

The **Search the web** button on the Check tab picks the 8 longest sentences of your document,
searches Google for each one as an exact phrase (through the [Serper](https://serper.dev) API),
downloads the top result pages and runs the normal detector against them. Matched web pages
appear as sources by URL. It is a bit slower than a library check (roughly 10 to 60 seconds).

1. Sign up at [serper.dev](https://serper.dev) and copy your API key (new accounts get free credits).
2. Create your own `.env` file from the template and paste the key in:

   ```bash
   cp .env.example .env
   # then edit .env so it reads:  SERPER_API_KEY=your_key_here
   ```

   (Or run `export SERPER_API_KEY=your_key_here` in the terminal before starting the app.)
3. Restart `python app.py`.

Notes:
- Each check uses at most 8 Serper credits. Results are cached in `data/web_cache/`, so checking
  the same text again is free.
- `.env` is git-ignored: never commit your key. If it leaks, create a new one on serper.dev.
- It only finds pages that Google has indexed. Paywalled papers and private submissions are not
  searchable.
- Without a key the rest of the app works normally; the button just shows a setup message.

## Adding your own reference documents

Easiest: use the **Source library** tab in the web app. Alternatively, put `.txt` / `.pdf` /
`.docx` files in `data/corpus/` and rebuild the whole index:

```bash
python -m scripts.build_index
```

More Wikipedia articles can be fetched with
`python -m scripts.build_corpus "Topic one" "Topic two"`.

## Project structure

```
Plagiarism-Detector/
├── app.py                    Flask web server + JSON API
├── templates/index.html      Web page (HTML)
├── static/style.css          Styling (light + dark mode)
├── static/app.js             Frontend logic (calls the API, renders the report)
├── check.py                  Command-line checker
├── setup.sh                  One-time setup
├── requirements.txt
├── src/
│   ├── config.py             Paths, model name, thresholds
│   ├── preprocess.py         Cleaning, sentence split, tokenize, stopwords, lemmatization
│   ├── lexical.py            Word n-grams, Jaccard and containment
│   ├── tfidf.py              TF-IDF vectors and cosine similarity
│   ├── semantic.py           Sentence-BERT embeddings
│   ├── detector.py           Candidate retrieval, score fusion, labels, report
│   ├── file_reader.py        .txt / .pdf / .docx text extraction
│   └── web_search.py         Serper web search + page download (optional feature)
├── scripts/
│   ├── setup_nltk.py         Downloads NLTK data
│   ├── build_corpus.py       Downloads Wikipedia articles
│   ├── build_index.py        Pre-computes embeddings / n-grams / TF-IDF for the corpus
│   └── evaluate.py           Precision / recall / F1 on the Clough corpus
├── data/
│   ├── corpus/               Reference documents (Wikipedia)
│   ├── samples/              Demo inputs (copy, light edit, paraphrase, original, mixed)
│   └── eval/clough/          Clough & Stevenson corpus (download separately, git-ignored)
├── results/                  Evaluation output (evaluation.md, scores.csv)
├── .env.example              Template for your SERPER_API_KEY (copy to .env)
└── models/                   SBERT model + corpus index (generated)
```

## Data credits

- Reference corpus: Wikipedia articles (CC BY-SA), fetched through the MediaWiki API.
- Evaluation: P. Clough and M. Stevenson, *Developing a corpus of plagiarised short answers*,
  Language Resources and Evaluation, 45(1), 2011.
- Model: `sentence-transformers/all-MiniLM-L6-v2` (Reimers & Gurevych, Sentence-BERT, 2019).
