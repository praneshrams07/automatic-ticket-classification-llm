import os
import pickle
import numpy as np
import re

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


SEED = 42
TEST_SIZE = 0.2

# ✅ FAST MODE (set None for full test)
EVAL_SAMPLE_SIZE = None

# model folders (matching your structure)
TFIDF_DIR = "models/tfidf_ticket_baseline"
LSTM_DIR = "models/ticket_lstm_model"
BERT_DIR = "models/bert_ticket_classifier"
XLMR_DIR = "models/xlmr_ticket_classifier"


def safe_text(x):
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def clean_text_basic(x: str) -> str:
    x = safe_text(x).lower()
    x = re.sub(r"\s+", " ", x).strip()
    return x


def compute_scores(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision_w": float(p_w),
        "recall_w": float(r_w),
        "f1_w": float(f1_w),
        "precision_macro": float(p_m),
        "recall_macro": float(r_m),
        "f1_macro": float(f1_m),
    }


def load_test_split():
    ds = load_dataset("Tobi-Bueck/customer-support-tickets")["train"]
    texts = [clean_text_basic(t) for t in ds["body"]]
    labels = list(ds["queue"])

    filtered = [(t, l) for t, l in zip(texts, labels) if t.strip() != ""]
    texts, labels = zip(*filtered)
    texts, labels = list(texts), list(labels)

    X_train, X_test, y_train_lbl, y_test_lbl = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
    )

    if EVAL_SAMPLE_SIZE is not None and len(X_test) > EVAL_SAMPLE_SIZE:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(X_test), size=EVAL_SAMPLE_SIZE, replace=False)
        X_test = [X_test[i] for i in idx]
        y_test_lbl = [y_test_lbl[i] for i in idx]
        print(f"⚡ FAST EVAL: using {EVAL_SAMPLE_SIZE} samples from test set")
    else:
        print(f"✅ FULL EVAL: using full test size = {len(X_test)}")

    return X_test, y_test_lbl


def eval_tfidf(X_test, y_test_lbl):
    with open(os.path.join(TFIDF_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vec: TfidfVectorizer = pickle.load(f)
    with open(os.path.join(TFIDF_DIR, "logreg_model.pkl"), "rb") as f:
        clf: LogisticRegression = pickle.load(f)
    with open(os.path.join(TFIDF_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    X = vec.transform(X_test)
    y_true = le.transform(y_test_lbl)
    y_pred = clf.predict(X)

    return compute_scores(y_true, y_pred)


def eval_lstm(X_test, y_test_lbl):
    model = tf.keras.models.load_model(os.path.join(LSTM_DIR, "lstm_ticket_classifier.keras"))
    with open(os.path.join(LSTM_DIR, "tokenizer.pkl"), "rb") as f:
        tok = pickle.load(f)
    with open(os.path.join(LSTM_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(LSTM_DIR, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    max_len = meta.get("max_len", 300)

    seq = tok.texts_to_sequences(X_test)
    X = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    y_true = le.transform(y_test_lbl)

    probs = model.predict(X, batch_size=256, verbose=0)
    y_pred = probs.argmax(axis=1)

    return compute_scores(y_true, y_pred)


def eval_hf(model_dir, X_test, y_test_lbl):
    # ✅ Use Apple MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device)
    model.eval()

    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    max_len = 256
    meta_path = os.path.join(model_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            max_len = meta.get("max_length", max_len)

    y_true = le.transform(y_test_lbl)

    preds = []
    bs = 64
    total = len(X_test)

    for i in range(0, total, bs):
        batch = X_test[i:i + bs]
        enc = tokenizer(
            batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            y_pred_batch = logits.argmax(dim=1).detach().cpu().numpy()

        preds.extend(y_pred_batch.tolist())

    y_pred = np.array(preds)
    return compute_scores(y_true, y_pred)


def print_table(results):
    keys = [
        "accuracy",
        "precision_w", "recall_w", "f1_w",
        "precision_macro", "recall_macro", "f1_macro"
    ]

    print("\n=== Model Metrics Comparison ===")
    header = f"{'model':14s} " + " ".join([f"{k:>14s}" for k in keys])
    print(header)
    print("-" * len(header))

    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        row = f"{model_name:14s} " + " ".join([f"{metrics[k]:14.4f}" for k in keys])
        print(row)


def main():
    X_test, y_test_lbl = load_test_split()
    results = {}

    if os.path.exists(TFIDF_DIR):
        print("\nEvaluating TF-IDF...")
        results["tfidf_logreg"] = eval_tfidf(X_test, y_test_lbl)

    if os.path.exists(LSTM_DIR):
        print("\nEvaluating LSTM...")
        try:
            results["lstm"] = eval_lstm(X_test, y_test_lbl)
        except Exception as e:
            print("⚠️ Skipping LSTM due to load error:")
            print("   ", repr(e))

    if os.path.exists(BERT_DIR):
        print("\nEvaluating mBERT...")
        results["mbert"] = eval_hf(BERT_DIR, X_test, y_test_lbl)

    if os.path.exists(XLMR_DIR):
        print("\nEvaluating XLM-R...")
        results["xlmr"] = eval_hf(XLMR_DIR, X_test, y_test_lbl)

    print_table(results)

    best = max(results, key=lambda k: results[k]["accuracy"])
    print("\n✅ Best model:", best, "| accuracy:", results[best]["accuracy"])

    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(
            {"best_model": best, "eval_sample_size": EVAL_SAMPLE_SIZE, "metrics": results[best]},
            f
        )

    # also save full results
    with open("models/eval_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("✅ Saved best model info to models/best_model.pkl")
    print("✅ Saved all metrics to models/eval_results.pkl")


if __name__ == "__main__":
    main()

