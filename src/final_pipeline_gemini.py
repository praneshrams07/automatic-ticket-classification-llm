import os
import sys
import json
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------- CONFIG ----------------
MODEL_DIR = "models/xlmr_ticket_classifier"   # ✅ best model folder
CONF_THRESHOLD = 0.70                         # ✅ recommended threshold
TOPK = 3
MAX_LENGTH_DEFAULT = 256

# Gemini (new SDK: google-genai)
GEMINI_MODEL_NAME = "gemini-2.0-flash"        # change if needed


@dataclass
class Prediction:
    queue: str
    confidence: float
    top3: List[Dict[str, Any]]
    decision: str  # "AUTO" or "NEEDS_REVIEW"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pipeline(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    le_path = os.path.join(model_dir, "label_encoder.pkl")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"Missing label_encoder.pkl in {model_dir}")
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)

    max_len = MAX_LENGTH_DEFAULT
    meta_path = os.path.join(model_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        max_len = meta.get("max_length", max_len)

    return tokenizer, model, label_encoder, max_len


def predict_ticket(
    text: str,
    tokenizer,
    model,
    label_encoder,
    max_len: int,
    device: torch.device,
    conf_threshold: float = CONF_THRESHOLD,
    topk: int = TOPK,
) -> Prediction:

    if text is None or str(text).strip() == "":
        return Prediction(
            queue="General Inquiry",
            confidence=0.0,
            top3=[{"queue": "General Inquiry", "confidence": 0.0}],
            decision="NEEDS_REVIEW",
        )

    enc = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    top_idx = probs.argsort()[::-1][:topk]
    top_items = []
    for i in top_idx:
        label = label_encoder.inverse_transform([int(i)])[0]
        top_items.append({"queue": label, "confidence": float(probs[i])})

    top1 = top_items[0]
    decision = "AUTO" if top1["confidence"] >= conf_threshold else "NEEDS_REVIEW"

    return Prediction(
        queue=top1["queue"],
        confidence=float(top1["confidence"]),
        top3=top_items,
        decision=decision,
    )


def build_gemini_prompt(ticket_text: str, predicted_queue: str, decision: str, top3: List[Dict[str, Any]]) -> str:
    top3_str = ", ".join([f"{x['queue']} ({x['confidence']:.2f})" for x in top3])

    return f"""
You are a customer support assistant.

Task:
Write a short, polite, empathetic acknowledgement message to a customer.

Context:
- Ticket text: {ticket_text}
- Predicted department/queue: {predicted_queue}
- Routing decision: {decision}
- Top-3 predicted queues: {top3_str}

Rules:
- Do NOT promise refunds, fixes, or timelines.
- Do NOT ask for sensitive data (passwords, OTP, full card numbers).
- Ask for 1-3 helpful details only if needed (order ID, account email, screenshots, device info).
- Keep it concise (5-10 lines).
- End with a reassuring closing.

Return only the reply message.
""".strip()


def fallback_reply(predicted_queue: str) -> str:
    return (
        "Thanks for reaching out — we’ve received your request and we’re here to help.\n\n"
        f"We’ve routed your ticket to **{predicted_queue}** for review. "
        "If you can share any helpful details (for example: order ID/transaction ID, "
        "date & time of the issue, and a screenshot if available), it will help us assist you faster.\n\n"
        "We’ll get back to you as soon as possible."
    )


def generate_gemini_reply(prompt: str, predicted_queue: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback_reply(predicted_queue)

    try:
        # ✅ New Gemini SDK
        # pip install -U google-genai
        from google import genai

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt
        )

        text = getattr(resp, "text", None)
        if text:
            return text.strip()

        return fallback_reply(predicted_queue)

    except Exception as e:
        # quota exceeded / invalid key / network -> fallback
        print("\n⚠️ Gemini unavailable, using fallback reply.")
        print("   Reason:", repr(e))
        return fallback_reply(predicted_queue)


def main():
    device = get_device()
    print("Device:", device)

    tokenizer, model, label_encoder, max_len = load_pipeline(MODEL_DIR)
    model.to(device)

    if len(sys.argv) >= 2:
        ticket_text = " ".join(sys.argv[1:])
    else:
        print("\nPaste ticket text (end with Enter):")
        ticket_text = input("> ").strip()

    pred = predict_ticket(
        ticket_text,
        tokenizer=tokenizer,
        model=model,
        label_encoder=label_encoder,
        max_len=max_len,
        device=device,
        conf_threshold=CONF_THRESHOLD,
        topk=TOPK,
    )

    print("\n--- Prediction ---")
    print("Queue:", pred.queue)
    print("Confidence:", round(pred.confidence, 4))
    print("Decision:", pred.decision)
    print("Top-3:", json.dumps(pred.top3, indent=2))

    prompt = build_gemini_prompt(ticket_text, pred.queue, pred.decision, pred.top3)

    # Optional: only call Gemini if confident
    # if pred.decision == "NEEDS_REVIEW":
    #     reply = fallback_reply(pred.queue)
    # else:
    #     reply = generate_gemini_reply(prompt, pred.queue)

    reply = generate_gemini_reply(prompt, pred.queue)

    print("\n--- Reply ---")
    print(reply)


if __name__ == "__main__":
    main()

