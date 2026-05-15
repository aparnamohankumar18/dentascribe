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

## Tech Stack
- Python, PyTorch, HuggingFace Transformers
- Streamlit (demo)
- Google Colab (T4 GPU)

## Course
Mathematics / Deep Learning — Group Project
