"""Evaluate the detector on the Clough & Stevenson (2011) plagiarism corpus.

95 short answers to 5 questions, each labeled:
  cut   - copy-pasted from the Wikipedia source
  light - copied with minor edits
  heavy - rewritten / paraphrased
  non   - written independently (same topic, NOT plagiarized)
Every answer is compared with the source text of its task (document-vs-document mode).

Outputs results/evaluation.md and results/scores.csv.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from src.config import EVAL_DIR, ROOT
from src.detector import Reference, detect
from src.preprocess import split_sentences
from src.semantic import embed

DATA = EVAL_DIR / "clough"
RESULTS = ROOT / "results"
CATEGORIES = ["cut", "light", "heavy", "non"]
METHODS = {
    "Whole-document embedding (baseline)": "baseline",
    "N-gram containment": "lexical",
    "TF-IDF cosine": "tfidf",
    "SBERT (sentence-level)": "semantic",
    "Fused score": "fused",
    "Full system (% plagiarized text)": "percent",
}


def read(name):
    return (DATA / name).read_text(encoding="utf-8", errors="ignore")


def doc_vector(text):
    """Baseline used by many simple projects: one averaged vector per document."""
    v = embed(split_sentences(text)).mean(axis=0)
    return v / np.linalg.norm(v)


def score_corpus():
    info = pd.read_csv(DATA / "file_information.csv")
    answers = info[info.Category != "orig"]
    refs, src_vecs = {}, {}
    for task in sorted(answers.Task.unique()):
        source = read(f"orig_task{task}.txt")
        refs[task] = Reference.from_documents({f"orig_task{task}": source})
        src_vecs[task] = doc_vector(source)

    rows = []
    for _, row in answers.iterrows():
        text = read(row.File)
        report = detect(text, refs[row.Task])
        rows.append({
            "file": row.File, "task": row.Task, "category": row.Category,
            "plagiarized": int(row.Category != "non"),
            "baseline": float(doc_vector(text) @ src_vecs[row.Task]),
            **report.method_scores,
            "percent": report.plagiarism_percent / 100,
            "verdict": report.verdict,
        })
        print(f"{row.File:18} {row.Category:6} -> {report.plagiarism_percent:5.1f}% {report.verdict}")
    return pd.DataFrame(rows)


def cv_threshold_predictions(scores, labels, folds=5, seed=0):
    """Pick the F1-best threshold on 4 folds, apply it to the 5th (no test-set peeking)."""
    preds = np.zeros_like(labels)
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(scores, labels):
        candidates = np.unique(scores[train])
        best = max(candidates, key=lambda t: f1_score(labels[train], scores[train] >= t))
        preds[test] = scores[test] >= best
    return preds


def metrics_row(name, labels, preds, scores=None):
    return {
        "Method": name,
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall": recall_score(labels, preds),
        "F1": f1_score(labels, preds),
        "ROC-AUC": roc_auc_score(labels, scores) if scores is not None else np.nan,
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    df = score_corpus()
    df.to_csv(RESULTS / "scores.csv", index=False)
    labels = df.plagiarized.to_numpy()

    by_category = df.groupby("category")[list(METHODS.values())].mean().loc[CATEGORIES]
    by_category.columns = list(METHODS)

    method_table = pd.DataFrame([
        metrics_row(name, labels, cv_threshold_predictions(df[col].to_numpy(), labels), df[col])
        for name, col in METHODS.items()
    ])

    system_preds = (df.verdict != "Original").astype(int).to_numpy()
    system_table = pd.DataFrame([metrics_row("Full system, fixed thresholds from config.py",
                                             labels, system_preds, df.percent)])
    verdicts = pd.crosstab(df.category, df.verdict).reindex(CATEGORIES).fillna(0).astype(int)
    cm = confusion_matrix(labels, system_preds)

    fmt = lambda t: t.to_markdown(index=False, floatfmt=".3f")
    report = f"""# Evaluation on the Clough & Stevenson plagiarism corpus

{len(df)} answers ({', '.join(f'{c}: {(df.category == c).sum()}' for c in CATEGORIES)}).
Binary task: plagiarized (cut / light / heavy) vs. not plagiarized (non).

## 1. Average score per category

Higher is more similar to the source. A useful method must separate **heavy** from **non**.

{by_category.to_markdown(floatfmt=".3f")}

## 2. Each method as a classifier

The threshold is chosen by 5-fold cross-validation (picked on 4 folds, tested on the 5th).
ROC-AUC is threshold-free.

{fmt(method_table)}

## 3. Full system with its fixed thresholds

Uses the thresholds in `src/config.py` (a document is flagged if its verdict is not "Original").

{fmt(system_table)}

Confusion matrix (rows = truth [non, plagiarized], columns = predicted):

| | Pred. non | Pred. plagiarized |
|---|---|---|
| **non** | {cm[0, 0]} | {cm[0, 1]} |
| **plagiarized** | {cm[1, 0]} | {cm[1, 1]} |

Verdict counts per category:

{verdicts.to_markdown()}
"""
    (RESULTS / "evaluation.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
