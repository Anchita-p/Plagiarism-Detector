# Evaluation on the Clough & Stevenson plagiarism corpus

95 answers (cut: 19, light: 19, heavy: 19, non: 38).
Binary task: plagiarized (cut / light / heavy) vs. not plagiarized (non).

## 1. Average score per category

Higher is more similar to the source. A useful method must separate **heavy** from **non**.

| category   |   Whole-document embedding (baseline) |   N-gram containment |   TF-IDF cosine |   SBERT (sentence-level) |   Fused score |   Full system (% plagiarized text) |
|:-----------|--------------------------------------:|---------------------:|----------------:|-------------------------:|--------------:|-----------------------------------:|
| cut        |                                 0.868 |                0.642 |           0.746 |                    0.831 |         0.759 |                              0.717 |
| light      |                                 0.898 |                0.486 |           0.719 |                    0.849 |         0.723 |                              0.816 |
| heavy      |                                 0.915 |                0.286 |           0.566 |                    0.776 |         0.593 |                              0.606 |
| non        |                                 0.768 |                0.025 |           0.290 |                    0.557 |         0.343 |                              0.134 |

## 2. Each method as a classifier

The threshold is chosen by 5-fold cross-validation (picked on 4 folds, tested on the 5th).
ROC-AUC is threshold-free.

| Method                              |   Accuracy |   Precision |   Recall |    F1 |   ROC-AUC |
|:------------------------------------|-----------:|------------:|---------:|------:|----------:|
| Whole-document embedding (baseline) |      0.705 |       0.746 |    0.772 | 0.759 |     0.872 |
| N-gram containment                  |      0.905 |       0.929 |    0.912 | 0.920 |     0.971 |
| TF-IDF cosine                       |      0.895 |       0.912 |    0.912 | 0.912 |     0.970 |
| SBERT (sentence-level)              |      0.874 |       0.895 |    0.895 | 0.895 |     0.947 |
| Fused score                         |      0.947 |       0.964 |    0.947 | 0.956 |     0.979 |
| Full system (% plagiarized text)    |      0.853 |       0.891 |    0.860 | 0.875 |     0.965 |

## 3. Full system with its fixed thresholds

Uses the thresholds in `src/config.py` (a document is flagged if its verdict is not "Original").

| Method                                       |   Accuracy |   Precision |   Recall |    F1 |   ROC-AUC |
|:---------------------------------------------|-----------:|------------:|---------:|------:|----------:|
| Full system, fixed thresholds from config.py |      0.863 |       0.824 |    0.982 | 0.896 |     0.965 |

Confusion matrix (rows = truth [non, plagiarized], columns = predicted):

| | Pred. non | Pred. plagiarized |
|---|---|---|
| **non** | 26 | 12 |
| **plagiarized** | 1 | 56 |

Verdict counts per category:

| category   |   Original |   Plagiarized |   Suspicious |
|:-----------|-----------:|--------------:|-------------:|
| cut        |          1 |            14 |            4 |
| light      |          0 |            19 |            0 |
| heavy      |          0 |            14 |            5 |
| non        |         26 |             2 |           10 |
