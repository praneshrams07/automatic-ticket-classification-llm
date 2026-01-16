import os
import pickle
import numpy as np
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

SEED = 42
TEST_SIZE = 0.2

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 256
EPOCHS = 10
BATCH_SIZE = 8
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

OUT_DIR = "models/xlmr_ticket_classifier"
RUN_DIR = "runs/xlmr_results"


def safe_text(x):
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

    ds = load_dataset("Tobi-Bueck/customer-support-tickets")["train"]
    texts = [safe_text(t) for t in list(ds["body"])]
    labels = list(ds["queue"])

    filtered = [(t, l) for t, l in zip(texts, labels) if t.strip() != ""]
    texts, labels = zip(*filtered)
    texts, labels = list(texts), list(labels)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    train_hf = Dataset.from_dict({"text": X_train, "labels": y_train}).map(tok, batched=True)
    test_hf = Dataset.from_dict({"text": X_test, "labels": y_test}).map(tok, batched=True)
    train_hf.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_hf.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)

    def compute_metrics(eval_pred):
        logits, labels_ = eval_pred
        preds = np.argmax(logits, axis=1)
        return {"accuracy": accuracy_score(labels_, preds)}

    args = TrainingArguments(
        output_dir=RUN_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        fp16=True,
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_hf,
        eval_dataset=test_hf,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    pred = trainer.predict(test_hf)
    y_pred = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print("XLM-R Accuracy:", acc)

    # save
    trainer.model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    with open(os.path.join(OUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    meta = {
        "model_type": "xlmr",
        "hf_model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "warmup_ratio": WARMUP_RATIO,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "accuracy": float(acc),
        "num_classes": int(num_classes),
        "classes": list(le.classes_),
    }
    with open(os.path.join(OUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ Saved XLM-R to: {OUT_DIR}")


if __name__ == "__main__":
    main()
