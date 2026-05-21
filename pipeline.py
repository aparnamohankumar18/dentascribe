# ============================================================
# DentaScribe — Integration Pipeline
# Role 5 — Integration & Backend (Koroush) | covered by Ali
# Connects all models end-to-end: Whisper → NER → Form → Filing
# ============================================================

import os
import json
from datetime import datetime


# ── Role 1 — Andy: Load Whisper ASR ─────────────────────────
def load_whisper(model_path="openai/whisper-base"):
    """Load Whisper processor and model for speech recognition."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained(model_path)
    model     = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    print(f"[Role 1 — Andy] Whisper loaded from: {model_path}")
    return processor, model


# ── Role 2 — Ali: Load NER Model ────────────────────────────
def load_ner(model_path="dental_ner_model"):
    """Load DistilBERT NER pipeline for dental entity extraction."""
    from transformers import pipeline
    ner_pipe = pipeline(
        "token-classification",
        model=model_path,
        aggregation_strategy="simple",
    )
    print(f"[Role 2 — Ali] NER model loaded from: {model_path}")
    return ner_pipe


# ── Role 4 — Iva: Load Form Classifier ──────────────────────
def load_form_classifier(
    model_path="iva_form_classifier/final",
    tokenizer_name="iva_form_classifier/final",
):
    """Load DistilBERT sequence classifier for dental form type prediction."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print(f"[Role 4 — Iva] Form classifier loaded from: {model_path}")
    return tokenizer, model


# ── Role 1 — Andy: Transcribe Audio ─────────────────────────
def transcribe_audio(audio_path, processor, model):
    """
    Convert WAV/MP3 audio to text using fine-tuned Whisper.
    Steps: Load → Stereo→Mono → Normalize → Resample 16kHz → Whisper inference
    """
    import torch
    import numpy as np

    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception:
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"[Role 1 — Andy] Transcript: {transcript}")
    return transcript


# ── Role 2 — Ali: Extract Entities ──────────────────────────
def extract_entities(transcript, ner_pipeline):
    """
    Run NER on transcript and return list of dental entities.
    Manually merges ## subword tokens into complete words.
    Labels: Disease (diagnoses) and Chemical (medications).
    """
    entities = ner_pipeline(transcript.lower())

    merged = []
    for ent in entities:
        word  = ent.get("word", "")
        label = ent.get("entity_group", ent.get("entity", ""))
        score = ent.get("score", 0)

        if word.startswith("##") and merged:
            # Subword continuation — append to previous entity
            merged[-1]["word"] += word[2:]
            merged[-1]["score"] = round((merged[-1]["score"] + score) / 2, 4)
        else:
            merged.append({"word": word.strip(), "label": label, "score": round(score, 4)})

    # Filter out very short leftovers
    cleaned = [e for e in merged if len(e["word"].strip()) > 2]

    # Complete truncated words by matching against the original transcript
    transcript_words = transcript.lower().split()
    for ent in cleaned:
        fragment = ent["word"].lower()
        for tw in transcript_words:
            tw_clean = tw.strip(".,;:")
            if tw_clean.startswith(fragment) and len(tw_clean) > len(fragment):
                ent["word"] = tw_clean
                break

    print(f"[Role 2 — Ali] Entities found: {cleaned}")
    return cleaned


# ── Role 4 — Iva: Classify Form & Autofill ──────────────────
def classify_and_fill_form(transcript, entities, tokenizer, model):
    """
    Classify which dental form is needed and autofill fields from entities.
    Returns a complete form dict ready for filing.
    """
    import torch

    inputs = tokenizer(
        transcript,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs      = torch.softmax(outputs.logits, dim=1)
    pred_id    = probs.argmax().item()
    confidence = probs[0][pred_id].item()

    id2label = model.config.id2label
    form_type = id2label[pred_id]

    # Extract diagnosis and medication from NER entities
    diagnosis  = [e["word"] for e in entities if "Disease" in e["label"] or "DISEASE" in e["label"]]
    medication = [e["word"] for e in entities if "Chemical" in e["label"] or "CHEM" in e["label"]]

    form = {
        "form_type":     form_type,
        "confidence":    round(confidence, 4),
        "diagnosis":     diagnosis if diagnosis else "not detected",
        "medication":    medication if medication else "not detected",
        "original_text": transcript,
        "status":        "ready_for_review",
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"[Role 4 — Iva] Form classified as: {form_type} ({confidence:.1%})")
    return form


# ── Role 5 — Koroush (covered by Ali): Save to Filing System ─
def save_to_filing_system(patient_id, form, base_dir="filing_system"):
    """
    Save completed form as JSON.
    Structure: filing_system/{patient_id}/{date}/{form_type}_{time}.json
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    folder   = os.path.join(base_dir, patient_id, date_str)
    os.makedirs(folder, exist_ok=True)

    filename = f"{form['form_type']}_{time_str}.json"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w") as f:
        json.dump(form, f, indent=2)

    print(f"[Role 5 — Koroush/Ali] Form saved to: {filepath}")
    return filepath


# ── Full Pipeline ────────────────────────────────────────────
def run_pipeline(audio_path, patient_id,
                 whisper_processor, whisper_model,
                 ner_pipe,
                 form_tokenizer, form_model):
    """
    End-to-end pipeline:
    Audio → Whisper ASR → NER → Form Classifier → Filing System
    """
    print("\n" + "="*50)
    print("DentaScribe Pipeline Starting")
    print("="*50)

    transcript = transcribe_audio(audio_path, whisper_processor, whisper_model)
    entities   = extract_entities(transcript, ner_pipe)
    form       = classify_and_fill_form(transcript, entities, form_tokenizer, form_model)
    saved_path = save_to_filing_system(patient_id, form)

    print("\n✅ Pipeline complete.")
    print(f"   Form type : {form['form_type']}")
    print(f"   Confidence: {form['confidence']:.1%}")
    print(f"   Saved to  : {saved_path}")
    print("="*50 + "\n")

    return {"transcript": transcript, "entities": entities, "form": form, "saved_path": saved_path}
