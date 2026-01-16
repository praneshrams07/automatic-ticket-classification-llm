import os
import pickle
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

SEED = 42
TEST_SIZE = 0.2
MAX_FEATURES = 50000
OUT_DIR = "models/tfidf_ticket_baseline"


def safe_text(x) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def clean_text(x: str) -> str:
    x = safe_text(x).lower()
    x = re.sub(r"\s+", " ", x).strip()
    return x


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = load_dataset("Tobi-Bueck/customer-support-tickets")["train"]
    texts = [clean_text(t) for t in ds["body"]]
    labels = list(ds["queue"])

    # remove empties
    filtered = [(t, l) for t, l in zip(texts, labels) if t.strip() != ""]
    texts, labels = zip(*filtered)
    texts, labels = list(texts), list(labels)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print("TF-IDF + Logistic Regression Accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    with open(os.path.join(OUT_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(OUT_DIR, "logreg_model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(OUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    meta = {
        "model_type": "tfidf_logreg",
        "max_features": MAX_FEATURES,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "accuracy": float(acc),
        "num_classes": int(len(le.classes_)),
    }
    with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ Saved TF-IDF baseline to: {OUT_DIR}")


if __name__ == "__main__":
    main()
