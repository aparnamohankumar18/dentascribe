# DentaScribe — AI Dental Voice Assistant

An AI-powered voice assistant that listens to dental clinic conversations
and automatically fills out patient forms using speech recognition and NLP.

## Pipeline
Audio → Whisper ASR (Andy) → NER (Ali) → Form Classifier (Iva) → Filing System (Koroush) → Streamlit Demo (Aparna)

## Team
| Role | Member | Model |
|------|--------|-------|
| 1 — Speech Recognition | Andy | Whisper-small |
| 2 — NLP / NER | Ali | DistilBERT (fine-tuned on BC5CDR) |
| 3 — Data Engineering | Varsha | — |
| 4 — Form Intelligence | Iva | DistilBERT classifier |
| 5 — Integration & Backend | Koroush | — |
| 6 — Demo & Presentation | Aparna | Streamlit |

## Models on Hugging Face
All trained models are public — no manual download needed; they are auto-fetched on first run.

- NER: [neuroarcane/dental-ner-model](https://huggingface.co/neuroarcane/dental-ner-model)
- Form Classifier: [neuroarcane/dental-form-classifier](https://huggingface.co/neuroarcane/dental-form-classifier)
- ASR: [openai/whisper-small](https://huggingface.co/openai/whisper-small)

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the demo:
   ```bash
   streamlit run app.py
   ```
3. Open `http://localhost:8501` in your browser.

**First run takes ~3 minutes** — Streamlit downloads ~1.2 GB of model weights from Hugging Face and caches them locally at `~/.cache/huggingface/`. Subsequent runs start in seconds.

## Input Modes
Pick one in the sidebar:
- **🎙️ Record Live** — record audio in the browser via your mic
- **📁 Upload File** — upload a WAV/MP3/M4A recording (sample files in `samples/`)
- **📋 Demo Transcript** — skip audio entirely, use a built-in scenario

## Demo / Offline Mode
For live presentations where you want to guarantee zero network dependency:
```bash
HF_HUB_OFFLINE=1 streamlit run app.py
```
This forces Hugging Face to use only the local cache — no network calls. Run the app once normally beforehand so the models are cached.

## Test Samples
The `samples/` folder contains 10 short M4A recordings of dental consultations covering different form types (extraction, periodontal, root canal, etc.). Use them via the Upload File input mode.

## Tech Stack
- Python, PyTorch, HuggingFace Transformers
- Streamlit (demo + live mic recording)
- Google Colab Pro+ (H100 GPU, used for training)

## Course
Mathematics / Deep Learning — Group Project
