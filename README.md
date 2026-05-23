# DentaScribe — AI Dental Voice Assistant

An AI-powered voice assistant that listens to dental clinic conversations
and automatically fills out patient forms using speech recognition and NLP.

## Pipeline
Audio → Whisper ASR (Andy) → NER (Ali) → Form Classifier (Iva) → Filing System (Koroush) → Streamlit Demo (Aparna)

## Team
| Role | Member | Model |
|------|--------|-------|
| 1 — Speech Recognition | Andy | Whisper (fine-tuned) |
| 2 — NLP / NER | Ali | DistilBERT (fine-tuned) |
| 3 — Data Engineering | Varsha | — |
| 4 — Form Intelligence | Iva | DistilBERT classifier |
| 5 — Integration & Backend | Koroush | — |
| 6 — Demo & Presentation | Aparna | Streamlit |

## How to Run
1. Download trained models from Google Colab and place in the project folder:
   - `dental_ner_model/` — Ali (Role 2)
   - `iva_form_classifier/` — Iva (Role 4)
   - `whisper_dental_model/` — Andy (Role 1)
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the demo: `streamlit run app.py`

## Tech Stack
- Python, PyTorch, HuggingFace Transformers
- Streamlit (demo)
- Google Colab Pro+ (H100 GPU)

## Course
Mathematics / Deep Learning — Group Project
