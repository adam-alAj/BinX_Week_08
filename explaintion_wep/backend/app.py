"""
BinX Tech AI & ML Internship — Week 8 Interactive Website
Flask Inference API for Arabic Sentiment Classification
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize

try:
    from qalsadi.lemmatizer import Lemmatizer
    HAS_QALSADI = True
except ImportError:
    HAS_QALSADI = False

app = Flask(__name__)
CORS(app)

# ============================================================
# Reproduce the EXACT pipeline from Day 4
# ============================================================

SEED = 42
np.random.seed(SEED)

LABEL_NAMES = ['Negative (0)', 'Positive (1)']
DIGIT_RE = re.compile(r'^\d+$')
LATIN_RE = re.compile(r'^[a-zA-Z]+$')
NEGATION_WORDS = {'\u0644\u0627', '\u0644\u0645', '\u0644\u0646', '\u0644\u064a\u0633', '\u0645\u0627',
                  '\u063a\u064a\u0631', '\u0628\u0644\u0627', '\u062f\u0648\u0646', '\u062d\u0627\u0634\u0627'}
INTENSIFIER_WORDS = {'\u062c\u062f\u0627\u064b', '\u062c\u062f\u0627', '\u0643\u062b\u064a\u0631\u0627\u064b', '\u0643\u062b\u064a\u0631\u0627'}
PROTECTED = NEGATION_WORDS | INTENSIFIER_WORDS
STOP_WORDS = set(nltk.corpus.stopwords.words('arabic'))

lemmatizer = Lemmatizer() if HAS_QALSADI else None

def normalize_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[\u0617-\u061a\u064b-\u0652]', '', text)
    text = re.sub(r'[\u0622\u0623\u0625]', '\u0627', text)
    text = text.replace('\u0629', '\u0647').replace('\u0649', '\u064a')
    text = re.sub(r'\u0640', '', text)
    text = re.sub(r'[!?.,:;()\[\]{}\"\'\-/]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def unify_alef(token):
    token = re.sub(r'[\u0622\u0623\u0625]', '\u0627', token)
    return token.replace('\u0629', '\u0647').replace('\u0649', '\u064a')

def preprocess_text(raw_text, lemma_table=None):
    if not isinstance(raw_text, str):
        return ''
    if lemma_table is None:
        lemma_table = {}
    tokens = word_tokenize(normalize_text(raw_text))
    out = []
    for tok in tokens:
        if DIGIT_RE.match(tok):
            continue
        if LATIN_RE.match(tok):
            out.append(tok.lower())
            continue
        u = unify_alef(tok)
        if u in PROTECTED:
            out.append(u)
            continue
        l = unify_alef(lemma_table.get(tok, tok))
        if u in STOP_WORDS or l in STOP_WORDS:
            continue
        out.append(l)
    return ' '.join(out)


# ============================================================
# Load data and build pipeline at startup
# ============================================================

DATA_PATH = None
ORIG_PATH = None
MODEL_BUILT = False
final_vec = None
lr_model = None
lemma_table = {}
feature_names = []

def find_data():
    """Find the data files relative to the backend location."""
    global DATA_PATH, ORIG_PATH
    # Walk up from the backend directory to find the project root
    project_root = Path(__file__).parent.parent.parent.parent
    candidates = [
        project_root / "Data" / "processed" / "arabic_sentiment_cleaned_20k.csv",
        project_root / "Data" / "ALP_dataset" / "arabic_sentiment_reviews.csv",
    ]
    # Also try from CWD walking up
    cwd = Path.cwd()
    for _ in range(8):
        candidates.append(cwd / "Data" / "processed" / "arabic_sentiment_cleaned_20k.csv")
        candidates.append(cwd / "Data" / "ALP_dataset" / "arabic_sentiment_reviews.csv")
        cwd = cwd.parent

    for p in candidates:
        if 'processed' in str(p) and p.exists():
            DATA_PATH = p
        if 'ALP_dataset' in str(p) and p.exists():
            ORIG_PATH = p

    return DATA_PATH is not None and ORIG_PATH is not None

def build_model():
    """Build the exact same pipeline from Day 2/4."""
    global final_vec, lr_model, lemma_table, feature_names, MODEL_BUILT

    if not find_data():
        print("WARNING: Data files not found. Inference will use a mock pipeline.")
        # Create a minimal mock for demo purposes
        final_vec = None
        lr_model = None
        MODEL_BUILT = False
        return

    print(f"Loading cleaned data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df['text_clean'] = df['text_clean'].astype(str)
    df['label'] = df['label'].astype(int)

    # Split
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    X_train_text = train_df['text_clean'].tolist()
    y_train = train_df['label'].to_numpy()

    # Build lemma table from original data
    print(f"Building lemma table from: {ORIG_PATH}")
    orig_df = pd.read_csv(ORIG_PATH)
    lc = Counter()
    for text in orig_df['content'].dropna().head(50000):
        if isinstance(text, str):
            for tok in word_tokenize(normalize_text(text)):
                if not DIGIT_RE.match(tok) and not LATIN_RE.match(tok):
                    lc[unify_alef(tok)] += 1

    lemma_table = {}
    if lemmatizer:
        for tok, cnt in lc.items():
            if cnt >= 5:
                try:
                    l = lemmatizer.lemmatize(tok)
                    if l and l != tok:
                        lemma_table[tok] = unify_alef(l)
                except:
                    pass

    print(f"Lemma table: {len(lemma_table):,} entries")

    # Train TF-IDF + LR (exact config from Day 2)
    final_vec = TfidfVectorizer(max_features=10000, min_df=2, sublinear_tf=True)
    Xtr = final_vec.fit_transform(X_train_text)
    feature_names = final_vec.get_feature_names_out().tolist()

    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    lr_model.fit(Xtr, y_train)

    # Verify reproduction
    from sklearn.metrics import accuracy_score, f1_score
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    Xte = final_vec.transform(test_df['text_clean'].tolist())
    y_test = test_df['label'].to_numpy()
    acc = accuracy_score(y_test, lr_model.predict(Xte))
    f1 = f1_score(y_test, lr_model.predict(Xte), average='macro')
    print(f"Reproduction check: Acc={acc:.4f}, F1={f1:.4f}")

    MODEL_BUILT = True
    print("Pipeline ready.")


# ============================================================
# API Endpoints
# ============================================================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL_BUILT,
        'has_lemmatizer': HAS_QALSADI,
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Full prediction endpoint: raw text -> prediction + explanation."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    raw_text = data['text']
    if not raw_text.strip():
        return jsonify({'error': 'Empty text'}), 400

    if not MODEL_BUILT:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 503

    t0 = time.time()

    # Step 1: Preprocess
    cleaned = preprocess_text(raw_text, lemma_table=lemma_table)
    t_preprocess = time.time() - t0

    # Step 2: Vectorize
    t1 = time.time()
    features = final_vec.transform([cleaned])
    t_vectorize = time.time() - t1

    # Step 3: Predict
    t2 = time.time()
    prediction = int(lr_model.predict(features)[0])
    probabilities = lr_model.predict_proba(features)[0]
    t_predict = time.time() - t2

    # Step 4: Get top features contributing to prediction
    t3 = time.time()
    # SHAP-like: feature contribution = feature_value * coefficient
    coefs = lr_model.coef_[0]
    feature_values = features.toarray().flatten()
    contributions = feature_values * coefs

    # Get top positive and negative contributing features
    top_positive_idx = np.argsort(contributions)[-10:][::-1]
    top_negative_idx = np.argsort(contributions)[:10]

    top_features = []
    for idx in top_positive_idx:
        if contributions[idx] > 0:
            top_features.append({
                'feature': feature_names[idx],
                'weight': round(float(contributions[idx]), 6),
                'direction': 'positive'
            })
    for idx in top_negative_idx:
        if contributions[idx] < 0:
            top_features.append({
                'feature': feature_names[idx],
                'weight': round(float(contributions[idx]), 6),
                'direction': 'negative'
            })

    t_explain = time.time() - t3
    t_total = time.time() - t0

    return jsonify({
        'input': raw_text,
        'cleaned': cleaned,
        'prediction': prediction,
        'label': LABEL_NAMES[prediction],
        'confidence': round(float(probabilities[prediction]), 4),
        'probabilities': {
            'Negative': round(float(probabilities[0]), 4),
            'Positive': round(float(probabilities[1]), 4),
        },
        'top_features': top_features,
        'timing': {
            'preprocess_ms': round(t_preprocess * 1000, 1),
            'vectorize_ms': round(t_vectorize * 1000, 1),
            'predict_ms': round(t_predict * 1000, 1),
            'explain_ms': round(t_explain * 1000, 1),
            'total_ms': round(t_total * 1000, 1),
        }
    })


@app.route('/api/preprocess', methods=['POST'])
def preprocess_endpoint():
    """Preprocessing-only endpoint for the preprocessing playground."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    raw_text = data['text']

    # Step-by-step preprocessing
    normalized = normalize_text(raw_text)
    tokens = word_tokenize(normalized)

    token_details = []
    for tok in tokens:
        detail = {'token': tok, 'status': 'kept'}
        if DIGIT_RE.match(tok):
            detail['status'] = 'removed (digit)'
        elif LATIN_RE.match(tok):
            detail['token'] = tok.lower()
            detail['status'] = 'kept (lowercased)'
        else:
            u = unify_alef(tok)
            if u in PROTECTED:
                detail['status'] = 'protected (negation/intensifier)'
            else:
                l = unify_alef(lemma_table.get(tok, tok))
                if u in STOP_WORDS or l in STOP_WORDS:
                    detail['status'] = 'removed (stop word)'
                elif l != tok:
                    detail['token'] = l
                    detail['status'] = f'lemmatized ({tok} -> {l})'
        token_details.append(detail)

    final_tokens = [t['token'] for t in token_details if 'removed' not in t['status']]
    cleaned = ' '.join(final_tokens)

    return jsonify({
        'input': raw_text,
        'normalized': normalized,
        'tokens': token_details,
        'final_tokens': final_tokens,
        'cleaned': cleaned,
        'token_count': {
            'original': len(tokens),
            'final': len(final_tokens),
            'removed': len(tokens) - len(final_tokens),
        }
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model configuration and metadata."""
    return jsonify({
        'model': 'Logistic Regression (C=1.0)',
        'representation': 'TF-IDF (10K features)',
        'tfidf_config': {
            'max_features': 10000,
            'min_df': 2,
            'sublinear_tf': True,
            'sparsity': 0.9966,
        },
        'dataset': {
            'name': '330K Arabic Sentiment Reviews',
            'working_sample': 20000,
            'splits': {'train': 14000, 'val': 3000, 'test': 3000},
            'classes': {'negative': 0, 'positive': 1},
        },
        'metrics': {
            'accuracy': 0.8623,
            'precision_macro': 0.8624,
            'recall_macro': 0.8623,
            'f1_macro': 0.8623,
            'roc_auc': 0.9417,
            'pr_auc': 0.9418,
            'cv5_f1_mean': 0.8556,
            'cv5_f1_std': 0.0033,
        },
        'baseline': {
            'model': 'AraBERT v2 (aubmindlab/bert-base-arabertv2)',
            'f1_macro': 0.9000,
        },
        'confusion_matrix': {
            'tn': 1300, 'fp': 200,
            'fn': 213, 'tp': 1287,
        },
        'word2vec_neighbors': {
            '\u0645\u0645\u062a\u0627\u0632': ['\u0631\u0627\u0639\u0639', '\u0628\u0635\u0631', '\u0645\u0644\u0648\u0646', '\u0645\u0645\u0627\u062b\u0644', '\u0639\u0638\u064a\u0645', '\u0645\u0630\u0647\u0644', '\u0648\u062a\u0633\u062a\u062d\u0642', '\u0648\u064a\u0633\u062a\u062d\u0642'],
            '\u0631\u0627\u0639\u0639': ['\u0645\u0645\u062a\u0627\u0632', '\u0645\u0630\u0647\u0644', '\u0639\u0638\u064a\u0645', '\u062c\u064a\u062f', '\u0645\u062a\u0645\u064a\u0632', '\u0641\u0646\u064a', '\u0648\u062a\u0633\u062a\u062d\u0642', '\u0645\u062f\u0647\u0634'],
            '\u0633\u064a\u0626': ['\u0633\u064a\u0626', '\u0645\u0633\u062a\u0642\u064a\u0645', '\u0631\u062f\u064a\u0621', '\u0641\u0638\u064a\u0639', '\u0627\u0644\u0641\u0642\u0631\u0627\u0621', '\u0628\u0637\u064a\u0626', '\u062e\u0637\u064a\u0631', '\u062c\u0628\u0646\u064a'],
            '\u062c\u0648\u062f\u0629': ['\u0631\u062f\u064a\u0621\u0629', '\u0635\u0648\u0631\u0629', '\u0645\u0631\u0627\u0642\u0628\u0629', '\u064a\u0646\u062a\u062c', '\u0645\u0648\u0627\u062f', '\u0627\u0633\u062a\u0646\u0633\u0627\u062e', '\u0645\u0646\u062e\u0641\u0636', '\u0645\u0648\u0636\u062d'],
        },
    })


if __name__ == '__main__':
    print("Building ML pipeline...")
    build_model()
    print("Starting inference API on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
