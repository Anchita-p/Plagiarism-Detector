"""Download Wikipedia articles as the reference corpus (data/corpus/*.txt).

Usage:  python -m scripts.build_corpus            # default topic list
        python -m scripts.build_corpus "Topic A" "Topic B"
"""
import re
import sys
import time

import requests

from src.config import CORPUS_DIR

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "NLP-Plagiarism-Detector/1.0 (student project; educational use)"}

# Topics students commonly write assignments about
TOPICS = [
    "Artificial intelligence", "Machine learning", "Deep learning", "Natural language processing",
    "Artificial neural network", "Computer vision", "Data science", "Big data", "Cloud computing",
    "Internet of things", "Blockchain", "Cryptocurrency", "Cybersecurity", "Computer virus",
    "Internet", "World Wide Web", "Social media", "Smartphone", "Operating system", "Database",
    "Python (programming language)", "Java (programming language)", "Algorithm", "Quantum computing",
    "Robotics", "Self-driving car", "5G", "Virtual reality", "Augmented reality", "ChatGPT",
    "Climate change", "Global warming", "Greenhouse effect", "Renewable energy", "Solar power",
    "Wind power", "Nuclear power", "Fossil fuel", "Air pollution", "Water pollution",
    "Plastic pollution", "Deforestation", "Biodiversity", "Ozone depletion", "Sustainable development",
    "Photosynthesis", "Cell (biology)", "DNA", "Evolution", "Natural selection", "Genetics",
    "Human brain", "Immune system", "Vaccine", "COVID-19 pandemic", "Cancer", "Diabetes",
    "Antibiotic", "Mental health", "Nutrition", "Black hole", "Solar System", "Big Bang",
    "Gravity", "Theory of relativity", "Quantum mechanics", "Electricity", "Periodic table",
    "Water cycle", "Earthquake", "Volcano", "Moon landing", "International Space Station",
    "World War I", "World War II", "French Revolution", "Industrial Revolution", "Cold War",
    "Indian independence movement", "Mahatma Gandhi", "Renaissance", "Roman Empire",
    "Ancient Egypt", "Mughal Empire", "Democracy", "United Nations", "Globalization",
    "Economics", "Inflation", "Stock market", "Capitalism", "Socialism", "Education",
    "Poverty", "Human rights", "Constitution of India", "Indian economy", "Taj Mahal",
    "Olympic Games", "Cricket", "Football", "Population growth", "Urbanization",
]

MAX_WORDS = 1500  # keep the first part of long articles so indexing stays fast on a laptop

SKIP_SECTIONS = re.compile(r"^==+\s*(See also|References|Notes|Further reading|External links|"
                           r"Bibliography|Sources|Citations|Footnotes)\s*==+", re.M)


def fetch(title, retries=5):
    params = {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
              "titles": title, "format": "json", "formatversion": 2}
    for attempt in range(retries):
        resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            break
        time.sleep(5 * (attempt + 1))  # rate limited: back off and retry
    resp.raise_for_status()
    page = resp.json()["query"]["pages"][0]
    text = page.get("extract", "")
    cut = SKIP_SECTIONS.search(text)
    if cut:
        text = text[:cut.start()]
    text = re.sub(r"^==+.*?==+\s*$", "", text, flags=re.M)  # drop section headings
    return page.get("title", title), truncate(re.sub(r"\n{2,}", "\n\n", text).strip())


def truncate(text, max_words=MAX_WORDS):
    """Cut long articles at a paragraph boundary near max_words."""
    words = text.split(" ")
    if len(words) <= max_words:
        return text
    cut = text.rfind("\n", 0, len(" ".join(words[:max_words])))
    return text[:cut].strip() if cut > 0 else " ".join(words[:max_words])


def safe_name(title):
    return re.sub(r"[^\w]+", "_", title).strip("_") + ".txt"


def main(topics):
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for i, topic in enumerate(topics, 1):
        if (CORPUS_DIR / safe_name(topic)).exists():
            continue
        try:
            title, text = fetch(topic)
        except Exception as exc:  # network errors etc.
            print(f"[{i}/{len(topics)}] FAILED {topic}: {exc}")
            continue
        if len(text.split()) < 200:
            print(f"[{i}/{len(topics)}] skipped {topic} (too short)")
            continue
        (CORPUS_DIR / safe_name(title)).write_text(text, encoding="utf-8")
        print(f"[{i}/{len(topics)}] {title}: {len(text.split())} words")
        time.sleep(1)


if __name__ == "__main__":
    main(sys.argv[1:] or TOPICS)
