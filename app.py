# app.py — Streamlit UI for Ticket Classification (XLM-R) + Gemini reply
# Run: streamlit run app.py
# Env: export GEMINI_API_KEY="YOUR_KEY"

import os
import json
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------- CONFIG ----------------
MODEL_DIR = "models/xlmr_ticket_classifier"
DEFAULT_CONF_THRESHOLD = 0.70
DEFAULT_TOPK = 3
MAX_LENGTH_DEFAULT = 256

# Gemini (new SDK: google-genai)
GEMINI_MODEL_NAME = "gemini-2.0-flash"


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


@st.cache_resource(show_spinner=False)
def load_pipeline(model_dir: str) -> Tuple[Any, Any, Any, int, torch.device]:
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        # Optional: silences some tokenizer warnings (safe to keep)
        fix_mistral_regex=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    model.to(device)

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

    return tokenizer, model, label_encoder, max_len, device


def predict_ticket(
    text: str,
    tokenizer,
    model,
    label_encoder,
    max_len: int,
    device: torch.device,
    conf_threshold: float,
    topk: int,
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
        st.warning("Gemini unavailable (using fallback reply).")
        st.caption(f"Reason: {type(e).__name__}")
        return fallback_reply(predicted_queue)


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Ticket Classifier + Gemini Reply", layout="centered")
st.title("🎫 Automatic Ticket Classification + Gemini Reply")
st.caption("Model: XLM-R (fine-tuned) • Confidence routing • Gemini reply with safe fallback")

with st.sidebar:
    st.subheader("Settings")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, float(DEFAULT_CONF_THRESHOLD), 0.01)
    topk = st.selectbox("Top-K predictions", [3, 5], index=0)
    only_gemini_if_auto = st.checkbox("Call Gemini only when decision is AUTO", value=False)

    st.markdown("---")
    st.write("Gemini key status:")
    if os.getenv("GEMINI_API_KEY"):
        st.success("GEMINI_API_KEY detected")
    else:
        st.warning("GEMINI_API_KEY not set (fallback reply will be used)")

ticket_text = st.text_area(
    "Paste customer ticket text:",
    height=180,
    placeholder="Example: My payment failed but money was deducted. Please help."
)

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("🔍 Classify & Draft Reply", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.experimental_rerun()

if run_btn:
    if not ticket_text.strip():
        st.error("Please paste some ticket text.")
        st.stop()

    with st.spinner("Loading model..."):
        tokenizer, model, label_encoder, max_len, device = load_pipeline(MODEL_DIR)

    with st.spinner("Predicting queue..."):
        pred = predict_ticket(
            ticket_text,
            tokenizer=tokenizer,
            model=model,
            label_encoder=label_encoder,
            max_len=max_len,
            device=device,
            conf_threshold=float(conf_threshold),
            topk=int(topk),
        )

    st.subheader("✅ Prediction")
    st.write(f"**Queue:** {pred.queue}")
    st.write(f"**Confidence:** {pred.confidence:.4f}")
    st.write(f"**Decision:** {pred.decision}")

    st.markdown("**Top predictions:**")
    st.json(pred.top3)

    prompt = build_gemini_prompt(ticket_text, pred.queue, pred.decision, pred.top3)

    with st.spinner("Drafting reply..."):
        if only_gemini_if_auto and pred.decision != "AUTO":
            reply = fallback_reply(pred.queue)
            st.info("Decision is NEEDS_REVIEW → using fallback reply (Gemini not called).")
        else:
            reply = generate_gemini_reply(prompt, pred.queue)

    st.subheader("💬 Reply")
    st.text_area("Copy reply:", value=reply, height=170)

    with st.expander("Show Gemini prompt (debug)"):
        st.code(prompt)
