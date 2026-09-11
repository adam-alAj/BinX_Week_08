# BinX Week 8 — Interactive Showcase Website

An interactive educational website presenting the complete Week 8 (Sprint 3) work from the BinX Tech AI & Machine Learning Internship.

## Architecture

```
explaintion_wep/
├── backend/
│   ├── app.py              # Flask inference API (loads the real ML pipeline)
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Single-page application
│   ├── css/styles.css      # Dark theme design system
│   └── js/
│       ├── app.js          # Navigation, animations, scroll handling
│       ├── charts.js       # Chart.js visualizations (real data)
│       └── inference.js    # Interactive lab API integration
└── README.md
```

## Quick Start

### 1. Start the Backend (ML Inference API)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The Flask server starts on `http://localhost:5000` and loads the trained ML pipeline.

### 2. Open the Frontend

Open `frontend/index.html` in a browser, or serve it:

```bash
cd frontend
python -m http.server 8080
```

Then visit `http://localhost:8080`.

### 3. Use the Interactive Lab

Navigate to the **Interactive Lab** section and enter Arabic text to get real predictions from the trained model.

## Features

- **Week 8 Overview** — Timeline of all 5 days
- **Day 1** — NLP preprocessing pipeline visualization
- **Day 2** — TF-IDF configuration, embeddings comparison
- **Day 3** — OpenCV preprocessing engine
- **Day 4** — Integrated pipeline, confusion matrix, error analysis
- **Day 5** — Full evaluation dashboard, baseline comparison, SHAP explainability
- **Sprint Review** — Acceptance criteria, retrospective
- **Interactive Lab** — Real-time Arabic sentiment prediction

## Data Sources

All metrics and results are extracted from the actual Week 8 notebooks:
- `BinX_Week_08/Day1/Sprint3_NLP-Preprocessing.ipynb`
- `BinX_Week_08/Day2/TF-IDF_Embeddings.ipynb` + `day2_results.json`
- `BinX_Week_08/Day3/OpenCV.ipynb`
- `BinX_Week_08/Day4/Model-Integration_Error-Analysis.ipynb` + `day4_error_analysis.json`
- `BinX_Week_08/Day5/Sprint-Review.ipynb`

No data is fabricated. Every number traces to a notebook output or generated artifact.
