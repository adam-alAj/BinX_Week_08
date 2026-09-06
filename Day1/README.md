# Week 8: Day 1 — Sprint 3 Planning & NLP Preprocessing

**Welcome to Day 1 of Week 8 — the start of Sprint 3 in Phase 3 of the BinX Tech AI & ML Internship Program.**

This notebook (`Sprint3_NLP-Preprocessing.ipynb`) kicks off **Sprint 3 — NLP & Computer Vision: Text Processing, Image Processing, Integration & Full Model Evaluation**.

> This notebook is **cumulative**: it continues the Week 7 project state, mirrors the Week 7 Day 4 sampling/split protocol, and prepares the cleaned text that Day 2 (TF-IDF & embeddings) will consume. Nothing here invents results — every number below is recomputed from the actual repository data or quoted from the executed Week 7 notebooks.

---

## 📋 Day Summary

**Focus:** Sprint 3 planning (goal + backlog + carry-forward improvements) and building a task-aware Arabic NLP preprocessing pipeline.

**What was completed today:**

1. **Sprint 3 planning** — defined the Sprint 3 goal and 15-item backlog covering integration & full evaluation of the Arabic sentiment classifier.
2. **Arabic NLP preprocessing pipeline** — built a normalize → tokenize → clean → stop-words → lemmatize pipeline that preserves sentiment-critical signal.
3. **Sentiment-signal preservation** — handled negations and intensifiers correctly (task-dependent preprocessing choice from the Week 8 curriculum).
4. **Reusable `preprocess_text()` + quality checks + save cleaned text for Day 2** — produced the cleaned text artifact that Day 2's TF-IDF and embeddings will consume.
5. **Environment verification** — confirmed dependencies: `nltk`, `qalsadi` (Arabic lemmatizer), `scikit-learn`, `pandas`, `numpy`, `matplotlib`.

**What's still planned (backlog items for later days):**

- TF-IDF vectorization + classical classifier baseline (Day 2)
- Word embeddings (Word2Vec/GloVe) + semantic-neighbour demo (Day 2)
- TF-IDF baseline vs AraBERT transformer on the same test set (Day 2)
- Carry forward Sprint-2 improvement: **k-fold cross-validation** for classical-model evaluation (Day 2/5)
- CV preprocessing (OpenCV) + augmentation pipeline (Day 3)
- Mentor code & notebook review via pull request (Day 3)
- End-to-end `predict()` integration with verified training/serving consistency (Day 4)
- Error analysis: confusion matrix + ≥3 misclassified examples categorised (Day 4)
- Full task-appropriate evaluation vs baselines; SHAP explainability (Day 5)
- Sprint Review + Retrospective (Day 5)

---

## 📁 Folder Structure

```
BinX_Week_08/
├── Day1/
│   ├── Sprint3_NLP-Preprocessing.ipynb   # ← Today's main notebook
│   └── README.md                         # ← You are here
└── README.md                             # Week 8 overview (to be updated)
```

---

## 📚 Related Resources

- **Week 7, Day 4** — [Attention & Transformers](./../BinX_Week_07/Day4/README.md) — fine-tuned AraBERT v2 on Arabic sentiment (reference model for Sprint 3: F1 = 0.9000 on 3,000 held-out reviews)
- **Week 7, Day 5** — [Sprint 2 Close-Out & Retrospective](../BinX_Week_07/Day5/README.md) — Sprint 2 retrospective with carry-forward actions (k-fold CV)
- **Week 8, README** — [Week 8 Overview](../README.md) — sprint structure and remaining days
- **Notebook** — [`Sprint3_NLP-Preprocessing.ipynb`](./Sprint3_NLP-Preprocessing.ipynb)

---

## 🛠️ Tech Stack

- **Language:** Python 3.13
- **Core:** Pandas, NumPy, scikit-learn, Matplotlib
- **NLP:** NLTK (ISRI Arabic stemmer), qalsadi (dictionary-based Arabic lemmatizer)
- **Environment:** Shared `.venv` at repository root (same environment as Week 7)

---

## 🚀 How to Run

1. **Navigate to Week 8 Day 1:**
   ```bash
   cd BinX_ML_Internship/BinX_Week_08/Day1
   ```

2. **Activate the virtual environment** (located at the parent root):
   ```bash
   ..\.venv\Scripts\activate        # Windows
   source ../../.venv/bin/activate  # Linux / macOS
   ```

3. **Install dependencies** (if needed):
   ```bash
   ..\.venv\Scripts\pip install nltk qalsadi
   # or from project root:
   pip install -r ../../requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Sprint3_NLP-Preprocessing.ipynb
   ```

---

## 📌 Key Concepts Covered

| Concept | Description |
|---------|-------------|
| **Sprint Planning** | Goal, backlog, priorities, Definition of Done |
| **Sprint Continuity** | Carrying forward Sprint 2 retrospective findings (k-fold CV) |
| **Arabic Text Preprocessing** | Normalization (Alif variants, hamza, dots, tashkeel removal) |
| **Tokenization** | Word-level deterministic regex tokenization |
| **Stop-Word Handling** | Task-aware — sentiment-critical negations protected (`ليس`, `لا`, `لم`, `لن`, `ما`, `غير`) |
| **Lemmatization** | Dictionary-based Arabic lemmatization via qalsadi |
| **Negation Protection** | Preserving sentiment signal — a key task-dependent preprocessing choice |
| **Pipeline Design** | Reusable `preprocess_text()` function with quality checks |

---

## 💡 Key Takeaways

1. **Preprocessing choices are task-dependent.** Removing "not" (a stop word in some lists) would be a serious error in sentiment analysis — it flips meaning. The same applies to Arabic negations.

2. **Two text pipelines must be kept separate:**
   - **Classical NLP preprocessing** (this notebook): for TF-IDF, word embeddings, classical ML baseline (Day 2)
   - **Transformer preprocessing** (Week 7, untouched): AraBERT uses its own WordPiece sub-word tokenization — no aggressive cleaning

3. **No classical Arabic text-cleaning pipeline existed before Day 1.** Week 7 Day 4 fed raw review strings directly to the AraBERT tokenizer. This notebook builds the classical pipeline **from scratch for the TF-IDF/embeddings thread**, while the transformer thread keeps using its native tokenizer.

4. **Sprint 3 builds around the Week-7 AraBERT model** (F1 = 0.9000) — the goal is integration + full evaluation, not retraining. Day-2 classical baseline will be compared against this reference on the same test set.

---

## ✅ Status: Day 1 Complete

| Item | Status |
|------|--------|
| Sprint 3 planning (goal + backlog) | ✅ Complete |
| Arabic NLP preprocessing pipeline | ✅ Complete |
| Sentiment-signal preservation (negations) | ✅ Complete |
| Reusable `preprocess_text()` + quality checks | ✅ Complete |
| Save cleaned text for Day 2 | ✅ Complete |

*Continue to Day 2: Text Representation (TF-IDF & Word Embeddings)*
