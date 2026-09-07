# Week 8: Day 2 — Text Representation: TF-IDF & Word Embeddings

**Welcome to Day 2 of Week 8 — Sprint 3 (NLP & Computer Vision) in the BinX Tech AI & ML Internship.**

This notebook (`TF-IDF_Embeddings.ipynb`) is **cumulative**: it consumes the task-aware Arabic
cleaned text produced by [Week 8 Day 1](../Day1/Sprint3_NLP-Preprocessing.ipynb)
(`Data/processed/arabic_sentiment_cleaned_20k.csv`) and evaluates every representation on the
**same 3,000-review held-out test set** that the Week 7 AraBERT transformer was evaluated on
(test macro F1 = 0.9000). Nothing here invents results — every number is recomputed from the
actual repository artifacts or quoted from the executed Week 7 notebooks.

---

## 📋 Day Summary

**Focus:** converting cleaned Arabic text into numerical representations (TF-IDF and word
embeddings), benchmarking them with a classical classifier, and making an evidence-based
representation decision for the Arabic sentiment project.

**What was completed today:**

1. **TF-IDF implementation** — `TfidfVectorizer` fitted on the 14,000 training reviews only
   (no leakage); a small validation sweep (3k / 5k / 10k / 20k features) selected
   **max_features = 10,000** (validation macro-F1 0.8503; the curriculum's 5,000 default scored
   0.8493).
2. **Classical baseline** — Logistic Regression on the sparse TF-IDF matrix:
   **test Accuracy 0.8623 · Precision (macro) 0.8624 · Recall (macro) 0.8623 ·
   F1 (macro) 0.8623 · ROC-AUC 0.9417** on the same 3,000-review test set as Week 7.
3. **Carried-forward Sprint-2 improvement** — 5-fold **stratified cross-validation** of the
   full vectorizer+classifier pipeline on train: macro-F1 = **0.8556 ± 0.0033** (stable).
4. **Word embeddings** — two Arabic-compatible sources: a **Word2Vec model trained on the
   train split only** (gensim, skip-gram, 100-d, 8,688 words, ~10 s) and the **pre-trained
   AraBERT v2 embedding matrix** (768-d × 64k, re-used from the Week-7 cached model — no new
   download, language-compatible by construction).
5. **Semantic-geometry demonstration** — nearest neighbours in both spaces recover polarity and
   consumer-domain clusters (e.g. `ممتاز` → `رائع`/`عظيم`/`مذهل`; `سيء` → `سيئ`/`فظيع`/
   `مروع`; `سعر` → `تكلفة`/`صفقة`/`ارخص`; `شراء` → `يشتري`/`تشتري`/`اشترى` in AraBERT space).
6. **Embedding-based document representation** — mean-pooled word vectors + the *same* Logistic
   Regression: Word2Vec-mean **0.8350**, AraBERT-static-mean **0.8313** (test macro-F1).
7. **TF-IDF vs embeddings comparison** on the same test set — TF-IDF is the strongest
   classical representation (idf weighting preserves word importance that mean-pooling loses).
8. **Comparison against Week 7** — contextual AraBERT v2 (macro-F1 **0.9000**, quoted from the
   executed Week-7 Day-4 notebook) still beats every classical Day-2 baseline; the Week-7 Day-3
   LSTM is quoted only as context (it ran on an ECG *signal* task — not a same-task comparison).
9. **Final decision** — TF-IDF is selected as the classical-thread representation (fast,
   interpretable, best classical score) while the **contextual AraBERT v2** model remains the
   project's core model for the integrated pipeline (best measured representation: 0.9000),
   consistent with the Week-7 architecture decision and the Sprint-3 plan.

**Result log:** `day2_results.json` (same folder) records the configurations and metrics.

---

## 📁 Folder Structure

```
BinX_Week_08/
├── Day1/
│   ├── Sprint3_NLP-Preprocessing.ipynb   # Day 1: planning + Arabic preprocessing (cleaner)
│   └── README.md
├── Day2/
│   ├── TF-IDF_Embeddings.ipynb           # ← Today's main notebook
│   ├── day2_results.json                 # Experiment log (configs + metrics)
│   └── README.md                         # ← You are here
└── README.md                             # Week 8 overview
```

---

## 📊 Headline Results (same 3,000-review held-out test set)

| Representation / Model | Type | Test macro-F1 | Notes |
|---|---|---|---|
| **TF-IDF + Logistic Regression** | Sparse statistical | **0.8623** | Classical-thread winner; ROC-AUC 0.9417; CV 0.8556 ± 0.0033 |
| Word2Vec-mean + LR | Dense static (domain) | 0.8350 | Word2Vec trained on train split only |
| AraBERT-static-mean + LR | Dense static (pre-trained Arabic) | 0.8313 | Week-7 embedding matrix, sub-word mean pooling |
| **Week 7: AraBERT v2 (fine-tuned)** | Contextual transformer | **0.9000** | Quoted from executed Week-7 notebook — same test protocol |

---

## 🛠️ Tech Stack

- **Python 3.13** — Pandas 3.0.3, NumPy 2.5.1, scikit-learn 1.9.0, NLTK 3.10.3
- **Representation:** scikit-learn `TfidfVectorizer`; **gensim 4.4.0** `Word2Vec`;
  Hugging Face **AraBERT v2** embedding matrix (cached from Week 7)
- **Environment:** shared `.venv` at the repository root (gensim added for this day)

---

## 🚀 How to Run

```bash
# from BinX_Week_08/Day2 (or anywhere above it — path resolution is cwd-robust)
..\..\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=python3 TF-IDF_Embeddings.ipynb
# or open in Jupyter:
jupyter notebook TF-IDF_Embeddings.ipynb
```

**Prerequisites:** Week 8 Day 1 must have run so that
`Data/processed/arabic_sentiment_cleaned_20k.csv` exists. The AraBERT cells reuse the Week-7
model cache (`local_files_only`); if the cache is absent on a fresh machine those cells degrade
gracefully and the Word2Vec demonstration still runs standalone.

---

## ✅ Status: Day 2 Complete

| Item | Status |
|------|--------|
| TF-IDF vectorization (fit on train only — no leakage) | ✅ |
| Vocabulary size justified by validation sweep | ✅ |
| Classical classifier baseline (Logistic Regression) | ✅ |
| Proper evaluation (accuracy, macro P/R/F1, confusion matrix, ROC-AUC) | ✅ |
| K-fold cross-validation (Sprint-2 retrospective carry-forward) | ✅ |
| Arabic-compatible word embeddings (AraBERT cached + domain Word2Vec) | ✅ |
| Semantic-neighbour demonstration & interpretation | ✅ |
| Embedding-based document representation evaluated | ✅ |
| TF-IDF vs embeddings vs Week-7 AraBERT comparison (same test set) | ✅ |
| Final representation decision justified by evidence | ✅ |
| Notebook executes end-to-end; results logged | ✅ |

*Continue to Day 3: Computer-Vision Preprocessing (OpenCV) + Mentor Review*
