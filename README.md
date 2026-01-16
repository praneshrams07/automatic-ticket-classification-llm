# Automatic Ticket Classification + LLM Reply (Gemini)

An end-to-end **customer support automation pipeline** that:
1) **Classifies** incoming ticket text into the correct **support queue/department** using a fine-tuned **XLM-R (Transformer)** model, and  
2) **Drafts a polite acknowledgement reply** using **Google Gemini**, with a safe **fallback template** when Gemini is unavailable.

> Dataset: `Tobi-Bueck/customer-support-tickets` (Hugging Face)

---

## ✨ What this project does

✅ **Ticket Routing (Multi-class Classification)**  
- Input: ticket `body` text  
- Output: predicted `queue` (52 categories)  
- Model: **XLM-R (xlm-roberta-base)** fine-tuned for sequence classification  
- Also returns **Top-K predictions** + **confidence score**

✅ **Confidence-based Triage**  
- If confidence ≥ threshold → **AUTO route**  
- Else → **NEEDS_REVIEW** (human verification)

✅ **Auto Reply Generation (Gemini)**  
- Sends ticket text + predicted queue to Gemini  
- Generates a short, professional acknowledgement  
- If Gemini quota/key is missing → **fallback reply** (no crash)

✅ **Streamlit UI**  
- Paste ticket text  
- Get queue + confidence + top predictions  
- Get a drafted response (Gemini / fallback)

---

## 🧠 Why XLM-R?
The dataset includes multiple languages (e.g., **English + German**).  
**XLM-R** handles multilingual text better than traditional LSTM baselines.

---

## 📁 Project Structure
```
automatic-ticket-classification-llm/
├─ app.py # Streamlit demo app (XLM-R + Gemini)
├─ src/
│ ├─ final_pipeline_gemini.py # CLI pipeline (predict + reply)
│ ├─ test_gemini.py # Quick Gemini sanity check
│ └─ (other scripts)
├─ models/
│ ├─ xlmr_ticket_classifier/ # Saved Hugging Face model folder
│ │ ├─ config.json
│ │ ├─ model.safetensors
│ │ ├─ tokenizer.json
│ │ ├─ label_encoder.pkl
│ │ └─ meta.pkl
│ └─ (other models if any)
├─ requirements.txt
└─ README.md
```
> ⚠️ Model weights are large. Consider using `.gitignore` to avoid pushing heavy files to GitHub.

---

## ⚙️ Setup

### 
1) Create and activate a virtual environment (macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2) Install dependencies
pip install -r requirements.txt

3) (Optional) Set Gemini API key
export GEMINI_API_KEY="YOUR_API_KEY"
If GEMINI_API_KEY is not set, the pipeline will automatically use a fallback reply.

```bash
## 🖥️ Run the Streamlit App
streamlit run app.py
```


Then open the URL shown in terminal.