import os
import pickle
import re
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

SEED = 42
TEST_SIZE = 0.2

MAX_VOCAB = 20000
MAX_LEN = 300
EMBED_DIM = 128

EPOCHS = 10
BATCH_SIZE = 32

OUT_DIR = "models/ticket_lstm_model"


def safe_text(x) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def clean_text(x: str) -> str:
    x = safe_text(x).lower()
    # keep letters + basic unicode letters, strip weird symbols
    x = re.sub(r"[^\w\säöüß]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def build_lstm(vocab_size: int, num_classes: int):
    model = models.Sequential([
        layers.Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    ds = load_dataset("Tobi-Bueck/customer-support-tickets")["train"]
    texts = [clean_text(t) for t in ds["body"]]
    labels = list(ds["queue"])

    filtered = [(t, l) for t, l in zip(texts, labels) if t.strip() != ""]
    texts, labels = zip(*filtered)
    texts, labels = list(texts), list(labels)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_texts)

    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_texts), maxlen=MAX_LEN, padding="post", truncating="post")
    X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_texts), maxlen=MAX_LEN, padding="post", truncating="post")

    model = build_lstm(MAX_VOCAB, num_classes=len(le.classes_))
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    probs = model.predict(X_test, batch_size=256)
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)

    print("LSTM Accuracy:", acc)
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    model.save(os.path.join(OUT_DIR, "lstm_ticket_classifier.keras"))

    with open(os.path.join(OUT_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(OUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    meta = {
        "model_type": "lstm",
        "max_vocab": MAX_VOCAB,
        "max_len": MAX_LEN,
        "embed_dim": EMBED_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "accuracy": float(acc),
        "num_classes": int(len(le.classes_)),
    }
    with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ Saved LSTM model to: {OUT_DIR}")


if __name__ == "__main__":
    main()
