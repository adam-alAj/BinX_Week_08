# Week 8: Deep Learning & Applied Project — Sprint 3 (Integration & Full Evaluation)

**Welcome to Week 8 of the BinX Tech AI & Machine Learning Internship Program.**

This week is **Sprint 3** of **Phase 3 — Deep Learning & Applied Project**. Sprint 2 (Week 7) established the Arabic sentiment classifier (AraBERT v2, F1 = 0.9000) and completed the dense network experiments. Sprint 3 turns the model into a **complete, trustworthy, deployable component** through integration and rigorous evaluation.

---

## 📋 Table of Contents

| Day | Topic | Notebook | Status |
|:---:|-------|----------|:------:|
| 1 | Sprint 3 Planning & NLP Preprocessing | [`Sprint3_NLP-Preprocessing.ipynb`](./Day1/Sprint3_NLP-Preprocessing.ipynb) | ✅ |
| 2 | Text Representation: TF-IDF & Word Embeddings | [`TF-IDF_Embeddings.ipynb`](./Day2/TF-IDF_Embeddings.ipynb) | ✅ |
| 3 | Computer-Vision Preprocessing (OpenCV) + Mentor Review | (Planned) | Later |
| 4 | Model Integration — End-to-End `predict()` Pipeline + Error Analysis | (Planned) | Later |
| 5 | Full Evaluation, SHAP Explainability, Sprint Review & Retrospective | (Planned) | Later |

---

## 📖 Summary by Day

### ✅ Day 1 — Sprint 3 Planning & NLP Preprocessing

**Focus:** Sprint 3 planning (goal + backlog + carry-forward improvements) and building a task-aware Arabic NLP preprocessing pipeline.

**Accomplishments:**
- Completed **Sprint 3 planning** with a 15-item backlog covering integration & full evaluation of the Arabic sentiment classifier
- Defined the **Sprint 3 goal**: turn the Arabic sentiment classifier into a complete, rigorously evaluated and explained pipeline — NLP preprocessing → text representation → model integration → rigorous evaluation → explainability
- Built a **task-aware Arabic NLP preprocessing pipeline**: normalize → tokenize → clean → stop-words → lemmatize
- Implemented **sentiment-signal preservation**: negation & intensifier protection (`ليس`, `لا`, `لم`, `لن`, `ما`, `غير` protected from stop-word removal)
- Created reusable **`preprocess_text()`** function with quality checks and saved cleaned text for Day 2
- Verified environment: Python 3.13, Pandas 3.0.3, NumPy 2.5.1, scikit-learn 1.9.0, NLTK 3.10.3, qalsadi (Arabic lemmatizer)

**Key concepts:**
- Preprocessing choices are task-dependent — removing negation words in sentiment analysis would flip meaning
- Two text pipelines kept separate: Classical NLP preprocessing (this notebook) vs Transformer preprocessing (Week 7, untouched)
- The pipeline runs cleanly end-to-end with documented preprocessing decisions

**Sprint 2 carry-forward:** k-fold cross-validation for classical-model evaluation (from Week 7 retrospective)

**Reference model:** AraBERT v2 (`aubmindlab/bert-base-arabertv2`), fine-tuned in Week 7 Day 4 — test F1 = 0.9000 on 3,000 held-out Arabic reviews

### ✅ Day 2 — Text Representation: TF-IDF & Word Embeddings

**Focus:** converting the Day-1 cleaned Arabic text into numerical representations (TF-IDF and word embeddings), benchmarking them with a classical classifier, and deciding which representation fits the project.

**Accomplishments:**
- **TF-IDF** (`TfidfVectorizer`, fit on the 14k train reviews only) with a validation sweep (3k/5k/10k/20k) that selected `max_features = 10,000` — **Logistic Regression baseline: test Accuracy 0.8623 · macro-F1 0.8623 · ROC-AUC 0.9417** on the same 3,000-review test set as Week 7
- **Carried-forward Sprint-2 improvement applied**: 5-fold stratified CV of the full pipeline on train → macro-F1 **0.8556 ± 0.0033** (stable)
- **Word embeddings**: domain **Word2Vec** trained on the train split only (gensim, 100-d) **+ pre-trained AraBERT v2 embedding matrix** (768-d, cached Week-7 model — Arabic-native, no new download)
- **Semantic-geometry demo**: nearest neighbours recover polarity & consumer-domain clusters in both embedding spaces (`ممتاز`→`رائع`/`عظيم`/`مذهل`, `سيء`→`رديء`/`فظيع`/`مروع`, `سعر`→`صفقة`/`تكلفة`/`ارخص`)
- **Embedding doc representations (mean pooling) + same LR**: Word2Vec-mean macro-F1 0.8350, AraBERT-static-mean 0.8313 → **TF-IDF is the strongest classical representation** (idf weighting beats naive mean-pooling)
- **Week-7 comparison**: contextual AraBERT v2 (macro-F1 0.9000, quoted) still beats all classical baselines → TF-IDF selected for the classical thread, AraBERT remains the core model for the integrated pipeline
- Result log saved to [`Day2/day2_results.json`](./Day2/day2_results.json)

### 📋 Day 3 — Computer-Vision Preprocessing (OpenCV) + Mentor Review *(Planned)*

- CV preprocessing (OpenCV) + augmentation pipeline (project CV thread)
- Mentor code & notebook review via pull request

### 📋 Day 4 — Model Integration: End-to-End `predict()` Pipeline + Error Analysis *(Planned)*

- End-to-end `predict()` integration with verified training/serving consistency
- Error analysis: confusion matrix + ≥3 misclassified examples categorised (data vs model)

### 📋 Day 5 — Full Evaluation, SHAP Explainability, Sprint Review & Retrospective *(Planned)*

- Full task-appropriate evaluation vs baselines; SHAP global + per-prediction explanations
- Sprint Review + Sprint Retrospective (one concrete change for Sprint 4 = deployment)
- Documentation, Git feature-branch workflow, reviewed PRs

---

## 🛠️ Skills & Tools Covered (Sprint 3)

| Skill | Day | Application |
|-------|:---:|-------------|
| **Sprint Planning** | 1 | Backlog, acceptance criteria, Definition of Done, sprint timeline |
| **Sprint Continuity** | 1 | Retrospective follow-through, action items carried forward (k-fold CV) |
| **Arabic NLP Preprocessing** | 1 | Normalization, tokenization, cleaning, stop-words, lemmatization |
| **Sentiment-Signal Preservation** | 1 | Negation & intensifier protection (task-dependent choice) |
| **Reusable Preprocessing Pipeline** | 1 | `preprocess_text()` function with quality checks |
| **TF-IDF Vectorization** | 2 | Fit on train only, no leakage, classical classifier baseline |
| **Word Embeddings** | 2 | Word2Vec/GloVe, semantic-neighbour demo |
| **Classical vs Transformer Comparison** | 2 | TF-IDF baseline vs AraBERT on same test set |
| **K-Fold Cross-Validation** | 2/5 | Stratified k-fold CV for classical-model evaluation |
| **CV Preprocessing (OpenCV)** | 3 | Image preprocessing + augmentation pipeline |
| **Mentor Code Review** | 3 | Pull request review workflow |
| **End-to-End Integration** | 4 | `predict()` function with training/serving consistency |
| **Error Analysis** | 4 | Confusion matrix, misclassified examples categorisation |
| **Full Evaluation** | 5 | Task-appropriate metrics vs baselines |
| **SHAP Explainability** | 5 | Global + per-prediction explanations |
| **Sprint Review & Retrospective** | 5 | Deliverables, metrics, improvement analysis, Sprint 4 planning |

---

## 📁 Folder Structure

```
BinX_Week_08/
├── Day1/
│   ├── Sprint3_NLP-Preprocessing.ipynb   # Sprint 3 planning + NLP preprocessing
│   └── README.md                         # Day 1 summary
├── Day2/
│   ├── TF-IDF_Embeddings.ipynb           # Text representation: TF-IDF & word embeddings
│   ├── day2_results.json                 # Day 2 experiment log (configs + metrics)
│   └── README.md                         # Day 2 summary
└── README.md                             # ← You are here
```

---

## 🔗 Related

- [Root Repository README](../README.md) — Full internship overview and progress tracker
- [Week 7: Sprint 2](../BinX_Week_07/README.md) — Previous sprint: AraBERT fine-tuning (F1 = 0.9000), dense network experiments
- [Week 7, Day 4 — Attention & Transformers](../BinX_Week_07/Day4/README.md) — Reference model for Sprint 3
- [Week 7, Day 5 — Sprint 2 Close-Out & Retrospective](../BinX_Week_07/Day5/README.md) — Sprint 2 retrospective with carry-forward actions

---

## 🚀 How to Run

1. **Navigate to Week 8:**
   ```bash
   cd BinX_ML_Internship/BinX_Week_08
   ```

2. **Activate the virtual environment** (located at the parent root):
   ```bash
   ..\.venv\Scripts\activate        # Windows
   source ../../.venv/bin/activate  # Linux / macOS
   ```

3. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

4. **Launch Jupyter Notebook (Day 1 or Day 2):**
   ```bash
   jupyter notebook Day1/Sprint3_NLP-Preprocessing.ipynb
   jupyter notebook Day2/TF-IDF_Embeddings.ipynb
   ```
