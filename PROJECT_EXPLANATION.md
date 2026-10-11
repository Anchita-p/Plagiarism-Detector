# Plagiarism Detector: How It Works and Why It Is an NLP Project

## 1. Problem statement

Plagiarism means presenting someone else's text as your own. It comes in levels of difficulty:

| Type | Example | What's needed to detect it |
|---|---|---|
| **Copy-paste** | Text copied word for word | Simple string or word matching |
| **Light edit** | A few words changed, sentence order kept | Fuzzy word matching (n-grams) |
| **Heavy paraphrase** | Same meaning, different words and structure | Understanding *meaning* (semantics) |
| **Not plagiarism** | Original text on the *same topic* | Must **not** be flagged |

The last row is the hardest part. A naive system that only checks "is this about the same
topic?" flags every honest essay on machine learning as plagiarized. This project detects
real copying, including paraphrase, while leaving original same-topic writing alone.

## 2. What happens when you press "Check"

```
 Suspicious text                         Reference corpus (built once, saved in models/)
       │                                         │
       ▼                                         ▼
 ┌──────────────┐                        ┌────────────────────────────┐
 │ Preprocessing │                       │ 102 Wikipedia articles     │
 │ (src/preprocess.py)                   │  → ~6.5k sentences           │
 └──────┬───────┘                        │  → n-gram sets              │
        ▼                                │  → TF-IDF matrix            │
  Sentence 1, Sentence 2, ...            │  → SBERT embeddings         │
        │                                └─────────────┬──────────────┘
        ▼                                              │
 ┌───────────────────────────────────────────────────────────────────┐
 │ For EACH suspicious sentence:                                      │
 │  1. Candidate retrieval: top-5 corpus sentences by SBERT           │
 │     + top-5 by TF-IDF                                              │
 │  2. Score every candidate with three methods:                     │
 │       • n-gram containment  (lexical)                             │
 │       • TF-IDF cosine       (statistical)                         │
 │       • SBERT cosine        (semantic)                            │
 │  3. Fuse: 0.25·ngram + 0.25·tfidf + 0.50·sbert → best candidate    │
 │  4. Label:  n-gram ≥ 0.50 → COPIED                                 │
 │             SBERT  ≥ 0.75 → PARAPHRASED                            │
 │             otherwise     → ORIGINAL                              │
 └───────────────────────────────────────────────────────────────────┘
        │
        ▼
 Report: % of words in flagged sentences → verdict
         (<15% Original, 15–40% Suspicious, ≥40% Plagiarized),
         highlighted text, matched sources, per-sentence scores
```

## 3. Each module explained

### 3.1 Preprocessing (`src/preprocess.py`)

| Step | Function | Example | NLP concept |
|---|---|---|---|
| Cleaning | `clean_text` | Unicode normalisation, curly quotes → straight, removes `[12]` citations | Text normalisation |
| Sentence segmentation | `split_sentences` | Paragraph → list of sentences (NLTK **Punkt**, an unsupervised model that learns abbreviations, so "Dr. Smith" is not split) | Sentence boundary detection |
| Tokenization | `tokenize` | "Machine-learning isn't new!" → `machinelearning`, `isnt`, `new` (punctuation, including hyphens, is removed before splitting) | Word tokenization |
| Lemmatization | `lemmatize` | studies → study, algorithms → algorithm | Morphology, WordNet |
| Stopword removal | `normalize` | drops *the, is, of, and...* | Stopword filtering |

**Note:** the WordNet lemmatizer treats every word as a noun unless told otherwise, so
"studies" → "study" works but "running" stays "running". Adding a POS tagger would fix verbs,
but in testing it made preprocessing about 7× slower for little gain, so it was left out to keep things simple.

**Lemmatization vs stemming:** a stemmer (Porter) chops suffixes ("studies" → "studi") and can
create non-words. A lemmatizer returns a dictionary word ("study"). Lemmatization is slower but
more accurate.

### 3.2 Lexical similarity: word n-grams (`src/lexical.py`)

An **n-gram** is a sequence of *n* consecutive words. With n = 3 (trigrams):

> "machine learning is a field of study" → (machine, learning, is), (learning, is, a),
> (is, a, field), (a, field, of), (field, of, study)

We compare the trigram **sets** of two sentences:

- **Jaccard similarity** = |A ∩ B| / |A ∪ B|
- **Containment** = |A ∩ B| / |A|: what fraction of the suspicious sentence's trigrams appear
  in the source. We use containment because a copied sentence may be a *part* of a longer
  source sentence. Jaccard would penalise that and containment does not.

Stopwords are **kept** here, because word order matters for copy detection and removing words
would create fake n-grams that never existed in the text. Text is lemmatized, so changing
"studies" to "study" does not break a match.

**Strength:** very precise for copy-paste. **Weakness:** one synonym per three words destroys
almost all trigrams, so it fails on paraphrase.

### 3.3 Statistical similarity: TF-IDF (`src/tfidf.py`)

Each sentence becomes a vector with one dimension per vocabulary term (lemmatized content words
plus word bigrams).

- **TF (term frequency):** how often the term occurs in this sentence (sublinear: 1 + log tf)
- **IDF (inverse document frequency):** log(N / df). Rare words like "photosynthesis" get high
  weight. Common words like "important" get low weight.
- **Cosine similarity** = (A · B) / (|A| |B|), the angle between two vectors, from 0 to 1.

IDF is learned from the whole corpus, so the system knows which words are distinctive. Two
sentences sharing "cyanobacteria" and "photopigment" are far more suspicious than two sharing
"system" and "process".

**Strength:** handles reordering and small edits, and is fast. **Weakness:** synonyms are
different dimensions ("car" vs "automobile" = 0 overlap), so heavy paraphrase is still missed.

### 3.4 Semantic similarity: Sentence-BERT (`src/semantic.py`)

**Sentence-BERT** (Reimers & Gurevych, 2019) is a transformer (BERT-style) network fine-tuned
on sentence pairs. It maps a sentence to a 384-dimensional vector so that sentences with the
**same meaning land close together**, even with completely different words.

- Model: `all-MiniLM-L6-v2`: 6 transformer layers, 22M parameters, runs on CPU.
- Inside: WordPiece **subword tokenization** → token embeddings + positional encoding →
  **self-attention** layers (each word's representation depends on its context) → **mean
  pooling** of token vectors → one sentence vector.
- Similarity: cosine between the two vectors.

Example from the demo sample `3_paraphrase.txt`:
- Source: *"The term machine learning was coined in 1959 by Arthur Samuel, an IBM employee and
  pioneer in the field of computer gaming and artificial intelligence."*
- Paraphrase: *"Arthur Samuel, who worked at IBM and was an early pioneer of computer games and
  AI, introduced the phrase "machine learning" in 1959."*
- N-gram containment = **0.05** (almost no shared 3-word sequences), SBERT = **0.93**.

Another measured example: "The cat sat on the mat" vs "A feline was sitting on the rug" scores
0.58, while "Stock prices fell sharply today" scores only 0.06 against either one.

**Important design choice:** SBERT gets the **raw sentence, not the preprocessed one**.
Transformers were trained on natural text. Removing stopwords or lemmatizing ("was sitting"
→ "sit") destroys grammar and context the model relies on and makes its embeddings worse.
Classic preprocessing helps n-grams and TF-IDF but hurts transformers. That's why the project
has two preprocessing paths.

### 3.5 Detection engine (`src/detector.py`)

1. **Candidate retrieval.** Scoring every suspicious sentence against all ~6.5k corpus sentences
   with n-grams would be slow, so the matrix multiplications for SBERT and TF-IDF pick the
   top 5 + 5 candidates. This is the same "retrieve then re-rank" idea used in search engines.
2. **Score fusion.** `0.25·ngram + 0.25·tfidf + 0.50·sbert` chooses the best-matching candidate.
3. **Labelling.**
   - `COPIED` if n-gram containment ≥ 0.50 (half of the trigrams match exactly)
   - `PARAPHRASED` if SBERT ≥ 0.75 (same meaning)
   - `ORIGINAL` otherwise
4. **Document score.** The percentage of words in flagged sentences (word-weighted, so a short
   flagged sentence counts less than a long one), plus a per-source breakdown.

**Why sentence level and not whole document?** A whole-document embedding only captures what
a document is *about*. Every essay on climate change is "about climate change", so a
whole-document approach gives high similarity to every one of them. Comparing sentence by
sentence asks a much stricter question: *does this specific statement have a near-identical
counterpart in the source?* Section 5 measures this difference.

### 3.6 Corpus and index (`scripts/build_corpus.py`, `scripts/build_index.py`)

- `build_corpus.py` downloads 103 Wikipedia topics that students commonly write about,
  through the MediaWiki API ("Global warming" redirects to "Climate change", so 102 files
  are saved). It strips references and "See also" sections and keeps the first ~1500 words
  of each article.
- `build_index.py` splits the corpus into sentences, then computes and saves all SBERT
  embeddings, n-gram sets and the TF-IDF model to `models/corpus_index.pkl`.
  This **offline indexing** step is what makes checking fast: at query time only the
  suspicious document needs encoding.

### 3.7 Interfaces

- **Web app**: a Flask backend (`app.py`) with an HTML/CSS/JavaScript frontend
  (`templates/index.html`, `static/style.css`, `static/app.js`). The browser sends the text to a
  small JSON API and draws the report:
  - `POST /api/check`: check a document against the source library
  - `POST /api/compare`: compare two documents
  - `POST /api/corpus/upload` and `POST /api/corpus/wikipedia`: add a source to the library
  - `GET /api/corpus`: list the sources

  The report shows a similarity ring, the verdict, the highlighted text (red = copied,
  orange = paraphrased, hover for the matching source sentence and its scores), matched
  sources, scores per method, and a sentence-by-sentence table.
- **Adding sources is incremental**: only the new document's sentences are encoded with SBERT
  and appended to the index. The TF-IDF model is re-fitted on stored per-sentence features,
  which takes about 3 seconds on an old laptop, so there's no full rebuild.
- `check.py`: the same report on the command line.

## 4. Evaluation method (`scripts/evaluate.py`)

**Dataset:** Clough & Stevenson (2011) corpus of plagiarised short answers. Students answered
5 computer-science questions (e.g. "What is Bayes' theorem?"). Each answer was written under
one of four instructions: `cut` (copy-paste from Wikipedia), `light` (minor edits), `heavy`
(rewrite/paraphrase) or `non` (write it yourself from other material). There are 95 answers
plus the 5 Wikipedia sources. The `non` answers are on the **same topic** as the source, which
makes this a fair test of the topic-vs-plagiarism problem.

**Procedure:** each answer is compared with its task's source document (compare mode).

**Metrics:**
- **Precision**: of the answers we flagged, how many were really plagiarized?
- **Recall**: of the plagiarized answers, how many did we catch?
- **F1**: harmonic mean of precision and recall.
- **ROC-AUC**: probability that a random plagiarized answer scores higher than a random
  original one (threshold-free; 1.0 = perfect ranking, 0.5 = random).

To compare the methods fairly, each method's decision threshold is chosen with **5-fold
cross-validation**: picked on 4/5 of the data and tested on the remaining 1/5, so no method is
tuned on the answers it is scored on. A **whole-document embedding baseline** (one averaged
vector per document, the approach used by many simple projects online) is included for
comparison.

## 5. Results

All numbers below come from running `python -m scripts.evaluate`. The full output is in
`results/evaluation.md`, and per-document scores are in `results/scores.csv`.

### 5.1 Average score per category

| Category | Whole-doc embedding (baseline) | N-gram | TF-IDF | SBERT (sentence) | Fused |
|---|---|---|---|---|---|
| cut (copy-paste) | 0.868 | 0.642 | 0.746 | 0.831 | 0.759 |
| light edit | 0.898 | 0.486 | 0.719 | 0.849 | 0.723 |
| heavy paraphrase | 0.915 | 0.286 | 0.566 | 0.776 | 0.593 |
| **non (original)** | **0.768** | **0.025** | **0.290** | **0.557** | **0.343** |

What this shows:
- **The whole-document baseline gives honest, original answers 0.77 similarity**, and it rates
  heavy paraphrases (0.915) *higher* than copy-paste (0.868). It measures topic, not copying.
  This is the main flaw of simple "one vector per document" projects.
- **N-gram** scores fall steadily from cut → light → heavy → non, as expected for a method that
  sees only exact word sequences. It nearly fails on heavy paraphrase (0.286).
- **SBERT** stays high for heavy paraphrase (0.776) but is lower for original answers (0.557).
  That's the semantic signal n-grams miss.

### 5.2 Each method as a classifier (5-fold cross-validated threshold)

| Method | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Whole-document embedding (baseline) | 0.705 | 0.746 | 0.772 | 0.759 | 0.872 |
| N-gram containment | 0.905 | 0.929 | 0.912 | 0.920 | 0.971 |
| TF-IDF cosine | 0.895 | 0.912 | 0.912 | 0.912 | 0.970 |
| SBERT (sentence-level) | 0.874 | 0.895 | 0.895 | 0.895 | 0.947 |
| **Fused score (document-level)** | **0.947** | **0.964** | **0.947** | **0.956** | **0.979** |

| Full system (% plagiarized text) | 0.853 | 0.891 | 0.860 | 0.875 | 0.965 |

**Combining the three scores beats every single method** (F1 0.956), and it improves on the
whole-document baseline by about 20 F1 points. The last row is the app's own measure (the share
of words in flagged sentences). It ranks documents well (AUC 0.965) but as a single-threshold
classifier it is slightly below the fused score. That's a possible improvement: base the verdict
on the fused score instead.

### 5.3 The app's verdict with its fixed thresholds

With the thresholds in `src/config.py` (Suspicious or Plagiarized = flagged):

| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| 0.863 | 0.824 | 0.982 | 0.896 |

| Truth \ Verdict | Original | Suspicious | Plagiarized |
|---|---|---|---|
| cut | 1 | 4 | 14 |
| light | 0 | 0 | 19 |
| heavy | 0 | 5 | 14 |
| non | 26 | 10 | 2 |

The app is tuned to **not miss plagiarism** (56 of 57 caught). The cost is that 10 original
answers land in "Suspicious". These are short answers on a narrow topic (e.g. defining
Bayes' theorem), where an honest definition naturally resembles the Wikipedia one. That's why
"Suspicious" means *needs a human look* rather than *guilty*. Only 2 of 38 original answers
were labeled "Plagiarized".

**Note:** the app's thresholds (n-gram 0.50, SBERT 0.75, verdict bands 15% / 40%) are fixed
defaults set *before* the evaluation was run and were **not tuned on this data**, so these
numbers are an honest out-of-the-box result. The dataset is small (95 answers), so treat
the numbers as indicative.

### 5.4 Demo samples (corpus mode, `data/samples/`)

| Sample | Result |
|---|---|
| `1_exact_copy.txt` | 100%, Plagiarized: all sentences "Copied" from *Machine learning* |
| `2_light_edit.txt` | 100%, Plagiarized |
| `3_paraphrase.txt` | 77%, Plagiarized: n-gram ≈ 0, but SBERT 0.77–0.93 catches it |
| `4_original_same_topic.txt` | **0%, Original**: an ML essay, but not copied |
| `5_mixed_sources.txt` | 37%, Suspicious: *Climate change* 25% (copied), *Photosynthesis* 12% (paraphrased); the student's own sentences are not flagged |

## 6. How this maps to the NLP syllabus

| Syllabus topic | Where it appears in the project |
|---|---|
| Text preprocessing / normalisation | `preprocess.clean_text`: unicode, quotes, citations |
| Sentence segmentation | NLTK Punkt in `split_sentences` |
| Tokenization (word and subword) | NLTK `word_tokenize`; WordPiece inside SBERT |
| Stopwords | `normalize(remove_stopwords=True)` for TF-IDF |
| Morphology: stemming vs lemmatization | WordNet lemmatizer; comparison explained in 3.1 |
| N-gram models | Word trigrams, Jaccard and containment in `lexical.py` |
| Vector space model, TF-IDF | `tfidf.py` with unigrams + bigrams, sublinear TF |
| Similarity measures | Jaccard, containment, cosine |
| Word / sentence embeddings, distributional semantics | Sentence-BERT in `semantic.py` |
| Transformers, attention, BERT | `all-MiniLM-L6-v2` (6-layer transformer, mean pooling) |
| Semantic textual similarity / paraphrase identification | Core task of the project |
| Information retrieval | Candidate retrieval + re-ranking, offline index |
| Evaluation metrics | Precision, recall, F1, ROC-AUC, cross-validation, confusion matrix |
| Applications of NLP | Plagiarism detection, academic integrity |

## 7. Limitations and future work

- **Corpus coverage.** Only text that exists in `data/corpus` can be found. Real systems
  (Turnitin) index billions of pages. Future work: a web-search mode using a search API.
- **Cross-language plagiarism.** Text translated from Hindi to English is not detected.
  Swapping in the multilingual model `paraphrase-multilingual-MiniLM-L12-v2` would add this
  with a one-line config change (and an index rebuild).
- **Hyphenated words** are merged ("machine-learning" → "machinelearning"), so they don't
  match the unhyphenated form in n-grams/TF-IDF. SBERT is unaffected.
- **Common knowledge.** Very generic sentences ("Machine learning is a part of AI") can match a
  source even when written honestly. A minimum-length or "common phrase" filter could reduce
  this.
- **Idea plagiarism.** Copying the structure or argument of a work in completely new words is
  beyond sentence-level similarity.
- **AI-generated text.** Text written by ChatGPT is new text, not copied, so this system will
  not flag it. Detecting it is a separate problem.
- **Scale.** NumPy brute-force search works for ~10^4–10^5 sentences. For millions, use an
  approximate nearest-neighbour index such as FAISS.

## 8. Likely viva questions

- **Why not just use SBERT alone?** SBERT is weaker at telling exact copying from a close
  paraphrase, and it can give fairly high scores to generic same-topic sentences. N-grams give
  precise, explainable evidence for copy-paste ("these exact phrases match"). Combining the
  methods makes the system more robust and the output easier to explain.
- **Why containment instead of Jaccard?** The suspicious sentence can be a fragment of a longer
  source sentence. Containment measures how much of the suspicious text is inside the source.
- **Why don't you remove stopwords for SBERT?** See 3.4: transformers need natural sentences.
- **What does cosine similarity measure?** The angle between vectors, independent of length,
  so a long and a short sentence can still be similar.
- **How did you pick the thresholds?** The app uses fixed, reasonable defaults (half of the
  trigrams matching = copied; SBERT ≥ 0.75 = same meaning) chosen before evaluation, so the
  Clough results are not tuned to the test data. For the method comparison, each method's
  threshold is chosen by 5-fold cross-validation.
- **What's the time complexity?** Indexing is O(N) encodings (done once). Each query sentence
  costs one encoding plus a matrix-vector product over N corpus sentences.
