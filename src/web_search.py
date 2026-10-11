"""Web source discovery: find pages that contain a document's sentences.

  1. pick the most distinctive sentences of the suspicious text,
  2. search Google (through the Serper API) for each one as an exact phrase,
  3. download the top result pages and extract their text,
  4. return {url: text}, ready for Reference.from_documents().

Needs the SERPER_API_KEY environment variable (free key at https://serper.dev).
"""
import hashlib
import ipaddress
import json
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests

from . import config
from .file_reader import read_text
from .preprocess import split_sentences

SERPER_URL = "https://google.serper.dev/search"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NLP-Plagiarism-Detector/1.0; educational use)"}

MAX_QUERIES = 8          # searches per document (each one uses 1 Serper credit)
RESULTS_PER_QUERY = 3    # result pages kept per search
MAX_PAGES = 15           # pages downloaded per document
MAX_PAGE_BYTES = 3_000_000
MIN_PAGE_WORDS = 50
QUERY_WORDS = 14         # words of a sentence used in the exact-phrase query
CACHE_DIR = config.DATA_DIR / "web_cache"
SKIP_DOMAINS = ("youtube.com", "facebook.com", "instagram.com", "twitter.com", "x.com", "pinterest.com")


class WebSearchError(ValueError):
    """User-facing problem with the web search (missing key, quota, network)."""


def is_configured():
    return bool(os.environ.get("SERPER_API_KEY"))


def pick_sentences(text, n=MAX_QUERIES):
    """The n longest sentences, in document order: long sentences are the most distinctive."""
    sentences = split_sentences(text)
    longest = sorted(range(len(sentences)), key=lambda i: -len(sentences[i].split()))[:n]
    return [sentences[i] for i in sorted(longest)]


def _cache_path(kind, key):
    return CACHE_DIR / f"{kind}_{hashlib.sha1(key.encode()).hexdigest()}.json"


def search(sentence):
    """Exact-phrase Google search for one sentence -> [{url, title, snippet}]. Cached on disk."""
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise WebSearchError("Web search is not set up. Set the SERPER_API_KEY environment variable "
                             "(free key at serper.dev) and restart the app.")
    phrase = " ".join(sentence.split()[:QUERY_WORDS])
    cache = _cache_path("search", phrase)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        resp = requests.post(SERPER_URL, headers={"X-API-KEY": key, "Content-Type": "application/json"},
                             json={"q": f'"{phrase}"', "num": RESULTS_PER_QUERY}, timeout=20)
    except requests.RequestException:
        raise WebSearchError("Could not reach the Serper search API. Check your internet connection.")
    if resp.status_code in (401, 403):
        raise WebSearchError("Serper rejected the API key. Check SERPER_API_KEY.")
    if resp.status_code == 429 or (resp.status_code == 400 and "credit" in resp.text.lower()):
        raise WebSearchError("The Serper search quota is used up.")
    resp.raise_for_status()
    results = [{"url": r["link"], "title": r.get("title", ""), "snippet": r.get("snippet", "")}
               for r in resp.json().get("organic", []) if r.get("link")][:RESULTS_PER_QUERY]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(results), encoding="utf-8")
    return results


def _is_public_url(url):
    """Refuse non-http(s) URLs and hosts that resolve to private/local addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if any(host == d or host.endswith("." + d) for d in SKIP_DOMAINS):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


def _html_to_text(html_bytes):
    from lxml import html as lxml_html
    tree = lxml_html.fromstring(html_bytes)
    for bad in tree.xpath("//script|//style|//noscript|//nav|//header|//footer|//aside|//form|//svg"):
        bad.getparent().remove(bad)
    blocks = [re.sub(r"\s+", " ", el.text_content()).strip() for el in tree.xpath("//p|//li|//h1|//h2|//h3|//blockquote")]
    return "\n".join(b for b in blocks if len(b.split()) >= 4)


def fetch_page(url):
    """Download a page and return its text ('' if it cannot be used)."""
    cache = _cache_path("page", url)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["text"]
    text = ""
    try:
        current = url
        for _ in range(4):   # follow redirects by hand so every hop is checked
            if not _is_public_url(current):
                break
            resp = requests.get(current, headers=HEADERS, timeout=15, stream=True, allow_redirects=False)
            if resp.is_redirect:
                current = requests.compat.urljoin(current, resp.headers.get("Location", ""))
                continue
            if resp.status_code != 200:
                break
            data = resp.raw.read(MAX_PAGE_BYTES, decode_content=True)
            ctype = resp.headers.get("Content-Type", "").lower()
            if "pdf" in ctype or current.lower().split("?")[0].endswith(".pdf"):
                text = read_text(data, "page.pdf")
            elif "html" in ctype or "xml" in ctype:
                text = _html_to_text(data)
            break
    except Exception:
        text = ""
    if len(text.split()) >= MIN_PAGE_WORDS:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"text": text}), encoding="utf-8")
        return text
    return ""


def find_web_sources(text):
    """Search the web for a document's sentences.
    Returns ({url: page_text}, number_of_searches)."""
    sentences = pick_sentences(text)
    if not sentences:
        return {}, 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        result_lists = list(pool.map(search, sentences))   # raises WebSearchError for the caller

    urls = []
    for result in _interleave(result_lists):
        if result["url"] not in urls:
            urls.append(result["url"])
    urls = urls[:MAX_PAGES]
    with ThreadPoolExecutor(max_workers=6) as pool:
        pages = list(pool.map(fetch_page, urls))
    return {u: t for u, t in zip(urls, pages) if t}, len(sentences)


def _interleave(lists):
    """Best result of every query first, then the second best, ... so every query gets pages."""
    for rank in range(RESULTS_PER_QUERY):
        for results in lists:
            if rank < len(results):
                yield results[rank]
