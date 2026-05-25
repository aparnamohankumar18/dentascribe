# ============================================================
# DentaScribe — AI Dental Voice Assistant
# Role 6 — Demo & Presentation (Aparna)
# Streamlit front-end that runs the full pipeline end-to-end
# ============================================================

import streamlit as st
import tempfile
import os
import json
from datetime import datetime

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="DentaScribe",
    page_icon="🦷",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-box {
        background: #f0f4ff;
        border-left: 4px solid #1a73e8;
        padding: 0.8rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .entity-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    .tag-disease  { background: #ffe0e0; color: #c0392b; }
    .tag-chemical { background: #e0f0ff; color: #1565c0; }
    .form-card {
        background: #ffffff;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #1a73e8, #34a853);
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-header">🦷 DentaScribe</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered voice assistant for dental clinics — speech → NER → auto-filled patient form</div>', unsafe_allow_html=True)

# ── Pipeline loader (cached so models load once) ─────────────
@st.cache_resource(show_spinner="Loading AI models — this takes ~30 s on first run…")
def load_models():
    """
    Load all three models into memory.
    Role 1 — Andy  : Whisper ASR
    Role 2 — Ali   : DistilBERT NER
    Role 4 — Iva   : DistilBERT Form Classifier
    """
    from pipeline import load_whisper, load_ner, load_form_classifier
    whisper_processor, whisper_model = load_whisper()
    ner_pipe                         = load_ner()
    form_tokenizer, form_model       = load_form_classifier()
    return whisper_processor, whisper_model, ner_pipe, form_tokenizer, form_model


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/tooth.png", width=80)
    st.markdown("### DentaScribe")
    st.markdown("**Deep Learning Group Project**")
    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("""
    1. 🎙️ Audio Upload
    2. 📝 Whisper ASR *(Andy)*
    3. 🏷️ NER *(Ali)*
    4. 🗄️ Data Engineering *(Varsha)*
    5. 📋 Form Classifier *(Iva)*
    6. 🔗 Integration *(Koroush)*
    7. 🖥️ This Demo *(Aparna)*
    """)
    st.markdown("---")
    st.markdown("**Patient ID**")
    patient_id = st.text_input("Enter patient ID", value="P001", max_chars=20)
    st.markdown("---")
    st.markdown("**Demo Mode**")
    demo_mode = st.checkbox("Use sample transcript (no audio needed)", value=False)

    DEMO_SCENARIOS = {
        # ── cleaning_form ────────────────────────────────────────
        "🦠 Gingivitis & Cleaning": (
            "The patient presents with moderate gingivitis and early periodontitis. "
            "We prescribed amoxicillin 500 mg three times daily for one week and "
            "recommended ibuprofen for pain management. A thorough cleaning was performed."
        ),
        # ── root_canal_form ──────────────────────────────────────
        "🦷 Root Canal": (
            "Patient has severe pulpitis in the lower left molar with signs of apical abscess. "
            "Root canal therapy was initiated today. We administered lidocaine for local anaesthesia "
            "and prescribed metronidazole 400 mg twice daily for five days along with paracetamol "
            "for post-procedure pain relief."
        ),
        # ── emergency_form ───────────────────────────────────────
        "🚨 Emergency Visit": (
            "Emergency visit for acute pericoronitis around the lower wisdom tooth. "
            "Significant swelling and trismus observed. Patient was given a chlorhexidine "
            "mouthwash prescription and amoxicillin-clavulanate 875 mg twice daily. "
            "Extraction of the third molar is recommended next week."
        ),
        # ── extraction_form ──────────────────────────────────────
        "🔧 Tooth Extraction": (
            "Extraction of the upper left second molar performed today due to severe dental caries "
            "and irreversible pulpitis. Local anaesthesia with articaine was administered. "
            "Patient prescribed amoxicillin 500 mg three times daily and ibuprofen 400 mg "
            "for swelling. Post-extraction instructions provided."
        ),
        # ── filling_form ─────────────────────────────────────────
        "🪥 Cavity Filling": (
            "Composite resin filling placed on the lower right first molar. "
            "Patient presented with moderate dental caries extending to the dentine. "
            "No signs of pulpitis detected. Fluoride varnish applied after the procedure. "
            "Patient advised to avoid hard foods for 24 hours."
        ),
        # ── crown_bridge_form ────────────────────────────────────
        "👑 Crown & Bridge": (
            "The patient requires a porcelain crown on the upper right premolar following "
            "a cracked tooth diagnosis. Impressions were taken today. Temporary crown placed "
            "and patient prescribed ibuprofen 400 mg as needed for sensitivity. "
            "Final crown fitting is scheduled in two weeks."
        ),
        # ── periodontal_form ─────────────────────────────────────
        "🌿 Periodontal Treatment": (
            "Deep periodontal scaling performed on all four quadrants. "
            "Patient has chronic periodontitis with 5 to 7 mm pocket depths. "
            "Doxycycline 100 mg once daily prescribed for two weeks. "
            "Chlorhexidine gel applied subgingivally. Follow-up in six weeks."
        ),
        # ── orthodontic_form ─────────────────────────────────────
        "😁 Orthodontic Consultation": (
            "Initial orthodontic consultation for Class II malocclusion with moderate crowding. "
            "Patient presents with dental fluorosis on the upper incisors. "
            "Treatment plan includes fixed braces over 18 months. "
            "Fluoride gel application recommended before bonding appointment."
        ),
        # ── whitening_form ───────────────────────────────────────
        "✨ Teeth Whitening": (
            "Patient requested professional teeth whitening for mild to moderate tooth discolouration. "
            "In-office bleaching performed using hydrogen peroxide gel. "
            "Mild dentinal hypersensitivity noted post-procedure. "
            "Prescribed fluoride toothpaste and advised to avoid coffee and tea for 48 hours."
        ),
        # ── consultation_form ────────────────────────────────────
        "📋 General Consultation": (
            "New patient consultation. Patient reports intermittent toothache and bleeding gums. "
            "Examination reveals mild gingivitis and early signs of bruxism. "
            "Dental radiographs taken. Recommended night guard for bruxism and "
            "chlorhexidine mouthwash twice daily. Full treatment plan to be discussed at next visit."
        ),
    }

    if "custom_transcript" not in st.session_state:
        st.session_state["custom_transcript"] = list(DEMO_SCENARIOS.values())[0]

    if demo_mode:
        preset = st.selectbox(
            "📋 Load a demo scenario",
            options=["— type your own —"] + list(DEMO_SCENARIOS.keys()),
        )
        if preset != "— type your own —":
            st.session_state["custom_transcript"] = DEMO_SCENARIOS[preset]

        SAMPLE_TRANSCRIPT = st.text_area(
            "Transcript (edit freely)",
            key="custom_transcript",
            height=160,
        )

# ── Main content ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### Step 1 — Upload Audio")
    audio_file = st.file_uploader(
        "Upload a WAV, MP3 or M4A recording of the dental consultation",
        type=["wav", "mp3", "m4a"],
        disabled=demo_mode,
    )
    if demo_mode:
        st.info("📌 Demo mode: using built-in sample transcript — no audio needed.")

    run_btn = st.button("▶ Run DentaScribe Pipeline", type="primary", use_container_width=True)

# ── Pipeline execution ───────────────────────────────────────
if run_btn:
    if not demo_mode and audio_file is None:
        st.error("Please upload an audio file or enable Demo Mode.")
        st.stop()

    # Load models
    try:
        whisper_processor, whisper_model, ner_pipe, form_tokenizer, form_model = load_models()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

    from pipeline import (
        transcribe_audio, extract_entities,
        classify_and_fill_form, save_to_filing_system,
    )

    # ── Step 1: Transcription ────────────────────────────────
    with st.spinner("🎙️ Transcribing audio…"):
        if demo_mode:
            transcript = SAMPLE_TRANSCRIPT
            st.success("✅ Demo transcript loaded.")
        else:
            suffix = "." + audio_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            try:
                transcript = transcribe_audio(tmp_path, whisper_processor, whisper_model)
                st.success("✅ Transcription complete.")
            finally:
                os.unlink(tmp_path)

    # ── Step 2: NER ──────────────────────────────────────────
    with st.spinner("🏷️ Running Named Entity Recognition…"):
        entities = extract_entities(transcript, ner_pipe)
        st.success(f"✅ {len(entities)} entities detected.")

    # ── Step 3: Form Classification + Autofill ───────────────
    with st.spinner("📋 Classifying form & auto-filling fields…"):
        form = classify_and_fill_form(transcript, entities, form_tokenizer, form_model)
        st.success("✅ Form classified and filled.")

    # ── Step 4: Filing ───────────────────────────────────────
    with st.spinner("🗄️ Saving to filing system…"):
        saved_path = save_to_filing_system(patient_id, form)
        st.success(f"✅ Saved → `{saved_path}`")

    # ────────────────────────────────────────────────────────
    # RESULTS
    # ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Results")

    r1, r2, r3 = st.columns(3)
    r1.metric("Form Type",   form.get("form_type", "—").replace("_", " ").title())
    r2.metric("Confidence",  f"{form.get('confidence', 0)*100:.1f}%")
    r3.metric("Entities Found", len(entities))

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📝 Transcript & Entities", "📋 Auto-filled Form", "🗄️ Raw JSON"])

    # Tab 1 — Transcript & NER
    with tab1:
        st.markdown("#### Transcript")
        st.info(transcript)

        st.markdown("#### Detected Entities")
        if entities:
            html_tags = ""
            for ent in entities:
                label = ent.get("entity_group", ent.get("label", ""))
                word  = ent.get("word", "")
                score = ent.get("score", 0)
                if "Disease" in label or "DISEASE" in label:
                    css = "tag-disease"
                    icon = "🦠"
                elif "Chemical" in label or "CHEM" in label:
                    css = "tag-chemical"
                    icon = "💊"
                else:
                    css = "tag-chemical"
                    icon = "🏷️"
                html_tags += (
                    f'<span class="entity-tag {css}">'
                    f'{icon} {word} <small>({label} {score:.0%})</small>'
                    f'</span>'
                )
            st.markdown(html_tags, unsafe_allow_html=True)

            # Entity table
            st.markdown("#### Entity Details")
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Word":       e.get("word", ""),
                    "Label":      e.get("entity_group", e.get("label", "")),
                    "Confidence": f"{e.get('score', 0):.2%}",
                }
                for e in entities
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("No entities detected in transcript.")

    # Tab 2 — Auto-filled form
    with tab2:
        st.markdown("#### Patient Form")
        st.markdown('<div class="form-card">', unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown(f"**Patient ID:** `{patient_id}`")
            st.markdown(f"**Form Type:** {form.get('form_type','—').replace('_',' ').title()}")
            st.markdown(f"**Date / Time:** {form.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M'))}")
            st.markdown(f"**Status:** {form.get('status','—').replace('_',' ').title()}")
        with fc2:
            conf = form.get("confidence", 0)
            st.markdown(f"**Classifier Confidence:** {conf*100:.1f}%")
            st.markdown(
                f'<div class="confidence-bar" style="width:{conf*100:.0f}%;"></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        fc3, fc4 = st.columns(2)
        with fc3:
            st.markdown("**Diagnosis / Condition**")
            diag = form.get("diagnosis", [])
            if diag:
                for d in (diag if isinstance(diag, list) else [diag]):
                    st.markdown(f"- 🦠 {d}")
            else:
                st.markdown("_Not detected_")

        with fc4:
            st.markdown("**Medication / Treatment**")
            meds = form.get("medication", [])
            if meds:
                for m in (meds if isinstance(meds, list) else [meds]):
                    st.markdown(f"- 💊 {m}")
            else:
                st.markdown("_Not detected_")

        st.markdown("---")
        st.markdown("**Full Transcript (recorded)**")
        st.text_area("", value=form.get("original_text", transcript), height=120, disabled=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 3 — Raw JSON
    with tab3:
        st.markdown("#### Raw Form JSON (saved to filing system)")
        st.json(form)
        st.markdown("#### All Detected Entities (JSON)")
        st.json(entities)

else:
    # Landing state — show pipeline diagram
    with col_right:
        st.markdown("### How it works")
        steps = [
            ("🎙️", "Step 1 — Speech Recognition", "Whisper (fine-tuned) converts dental audio to text  *(Andy)*"),
            ("🏷️", "Step 2 — Named Entity Recognition", "DistilBERT NER extracts diseases & medications  *(Ali)*"),
            ("📋", "Step 3 — Form Classification", "DistilBERT classifier picks the right form type  *(Iva)*"),
            ("🗄️", "Step 4 — Filing System", "JSON form saved by patient ID + date  *(Koroush)*"),
        ]
        for icon, title, desc in steps:
            st.markdown(
                f'<div class="step-box">'
                f'<strong>{icon} {title}</strong><br><span style="color:#555">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Quick Start")
    st.markdown("""
    1. Enter a **Patient ID** in the sidebar (e.g. `P001`)
    2. Upload a **WAV/MP3** recording of a dental consultation
       — or tick **Demo Mode** to use the built-in sample
    3. Click **▶ Run DentaScribe Pipeline**
    4. See the transcript, detected entities, and the auto-filled patient form
    """)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#999;font-size:0.8rem;'>"
    "DentaScribe · Deep Learning Group Project · "
    "Andy · Ali · Varsha · Iva · Koroush · Aparna"
    "</div>",
    unsafe_allow_html=True,
)
