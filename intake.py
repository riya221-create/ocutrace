"""
OcuTrace — Patient Intake Form
================================
Symptom collection + PDF report upload UI component.
Integrates with the PubMed RAG module to generate
evidence-backed pre-diagnosis summaries.

This module is imported by app.py and rendered as a
tab in the main Streamlit interface.

Usage (standalone test):
    streamlit run intake.py
"""

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(uploaded_file) -> str:
    """
    Extract text from an uploaded PDF medical report.
    Uses PyPDF2 if available, falls back to raw byte decode.
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        text   = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
        uploaded_file.seek(0)
        return text.strip()
    except ImportError:
        pass

    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        uploaded_file.seek(0)
        return text.strip()
    except ImportError:
        pass

    uploaded_file.seek(0)
    return "[PDF text extraction unavailable — install pypdf: pip install pypdf]"


# ─────────────────────────────────────────────────────────────────────────────
# INTAKE FORM RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_intake_form(api_key: Optional[str] = None) -> Optional[dict]:
    """
    Render the patient intake form and return collected data when submitted.

    Returns dict with keys: symptoms, condition, pdf_text, biomarkers
    or None if form not yet submitted.
    """

    st.markdown("### Patient intake")
    st.markdown(
        '<p style="color:#6B8080;font-size:0.9rem;">'
        "Fill in symptoms and upload any existing reports. "
        "OcuTrace will cross-reference your findings against verified PubMed literature."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown('<hr style="border-top:1px solid #1C3040;margin:0.8rem 0">', unsafe_allow_html=True)

    # ── Symptom entry ─────────────────────────────────────────────────────────
    st.markdown("#### Symptoms")
    col1, col2 = st.columns(2)

    with col1:
        vision_symptoms = st.multiselect(
            "Visual symptoms",
            options=[
                "Sudden vision loss",
                "Blurred vision",
                "Distorted vision (wavy lines)",
                "Dark spot in central vision",
                "Floaters",
                "Flashes of light",
                "Reduced peripheral vision",
                "Double vision",
            ],
            help="Select all that apply"
        )

    with col2:
        other_symptoms = st.multiselect(
            "Other symptoms",
            options=[
                "Headache",
                "Eye pain",
                "Eye redness",
                "Sensitivity to light",
                "Difficulty reading",
                "Colour vision changes",
            ],
        )

    symptom_detail = st.text_area(
        "Describe your symptoms in your own words (optional)",
        placeholder="e.g. I noticed a dark curtain in the lower part of my left eye vision 3 weeks ago...",
        height=100,
    )

    # ── Symptom timeline ──────────────────────────────────────────────────────
    st.markdown("#### Timeline")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        onset = st.selectbox("Onset", ["Sudden (minutes–hours)", "Gradual (days–weeks)", "Chronic (months+)"])
    with t_col2:
        affected_eye = st.selectbox("Affected eye", ["Left", "Right", "Both"])
    with t_col3:
        duration = st.selectbox("Duration", ["< 1 week", "1–4 weeks", "1–3 months", "> 3 months"])

    # ── Medical history ───────────────────────────────────────────────────────
    st.markdown("#### Medical history")
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        conditions = st.multiselect(
            "Existing conditions",
            ["Hypertension", "Diabetes", "Glaucoma", "High cholesterol",
             "Cardiovascular disease", "Thyroid disorder", "Autoimmune condition"],
        )
    with h_col2:
        medications = st.text_input(
            "Current medications (optional)",
            placeholder="e.g. amlodipine, metformin..."
        )

    # ── PDF upload ────────────────────────────────────────────────────────────
    st.markdown("#### Upload medical reports (optional)")
    uploaded_pdfs = st.file_uploader(
        "Upload previous OCT reports, blood tests, or referral letters (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text will be extracted and cross-referenced with medical literature"
    )

    pdf_text = ""
    if uploaded_pdfs:
        for pdf in uploaded_pdfs:
            st.caption(f"✓ {pdf.name} uploaded")
            extracted = extract_pdf_text(pdf)
            if extracted and not extracted.startswith("[PDF"):
                pdf_text += f"\n--- {pdf.name} ---\n{extracted[:2000]}"  # cap at 2k chars per file

    # ── RVO context (pre-filled from main pipeline) ───────────────────────────
    st.markdown("#### Condition context")
    condition = st.selectbox(
        "Primary condition (if known)",
        ["BRVO", "CRVO", "Hemi-RVO", "AMD", "DME", "Unknown / not yet diagnosed"],
        index=0,
    )

    # ── Submit ────────────────────────────────────────────────────────────────
    st.markdown('<hr style="border-top:1px solid #1C3040;margin:1rem 0">', unsafe_allow_html=True)

    submit_col, _ = st.columns([2, 5])
    with submit_col:
        submitted = st.button(
            "🔍  Search medical literature",
            type="primary",
            use_container_width=True,
            disabled=not (vision_symptoms or other_symptoms or symptom_detail),
        )

    if not submitted:
        if not (vision_symptoms or other_symptoms or symptom_detail):
            st.caption("Select at least one symptom to enable search.")
        return None

    # ── Build symptom string ──────────────────────────────────────────────────
    all_symptoms = vision_symptoms + other_symptoms
    symptom_str  = ", ".join(all_symptoms).lower()
    if symptom_detail:
        symptom_str += f". {symptom_detail}"
    if onset:
        symptom_str += f" Onset: {onset}."
    if affected_eye != "Both":
        symptom_str += f" Affected eye: {affected_eye}."
    if conditions:
        symptom_str += f" Medical history: {', '.join(conditions)}."

    return {
        "symptoms":  symptom_str,
        "condition": condition.split(" ")[0],  # "BRVO" not "BRVO (Branch...)"
        "pdf_text":  pdf_text,
        "context": {
            "onset":         onset,
            "affected_eye":  affected_eye,
            "duration":      duration,
            "conditions":    conditions,
            "medications":   medications,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# RAG RESULTS RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_rag_results(result):
    """Render a RAGResult object in the Streamlit UI."""

    st.markdown("### Evidence-backed summary")

    # Summary card
    st.markdown(
        f'<div style="background:#162028;border-left:4px solid #00C9A7;'
        f'border-radius:0 8px 8px 0;padding:1rem 1.2rem;margin-bottom:1rem;">'
        f'<p style="color:#F2F4F3;font-size:1rem;line-height:1.7;margin:0;">'
        f'{result.summary}</p></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**⚠ Risk flags**")
        for flag in result.risk_flags:
            st.markdown(
                f'<div style="background:#1A1010;border-left:3px solid #E05C5C;'
                f'padding:6px 12px;border-radius:0 6px 6px 0;margin:4px 0;'
                f'color:#F2F4F3;font-size:0.9rem;">{flag}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("**Questions for your doctor**")
        for i, q in enumerate(result.doctor_questions, 1):
            st.markdown(
                f'<div style="background:#162028;border:1px solid #1C3040;'
                f'padding:6px 12px;border-radius:6px;margin:4px 0;'
                f'color:#F2F4F3;font-size:0.9rem;">'
                f'<span style="color:#00C9A7;font-weight:600;">{i}.</span> {q}</div>',
                unsafe_allow_html=True,
            )

    # References
    st.markdown("**Evidence sources**")
    st.caption(f"{len(result.references)} PubMed articles retrieved · Every claim above is anchored to these sources")
    for ref in result.references:
        st.markdown(
            f'<div style="background:#0E1A20;border:1px solid #1C3040;'
            f'padding:5px 10px;border-radius:6px;margin:3px 0;font-size:0.82rem;">'
            f'<a href="{ref["url"]}" target="_blank" style="color:#00C9A7;">'
            f'[PMID:{ref["pmid"]}]</a>'
            f' <span style="color:#A8B5B0;">{ref["citation"]}</span></div>',
            unsafe_allow_html=True,
        )

    # Download
    st.download_button(
        "⬇ Download evidence report (JSON)",
        data=json.dumps(result.to_dict(), indent=2),
        file_name="ocutrace_evidence_report.json",
        mime="application/json",
    )


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(page_title="OcuTrace Intake", layout="wide")
    st.markdown("## OcuTrace — Patient Intake (standalone test)")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    intake_data = render_intake_form(api_key)

    if intake_data:
        st.success("Form submitted!")
        st.json(intake_data)

        if api_key:
            from pubmed_rag import MedRAG
            with st.spinner("Searching PubMed..."):
                rag    = MedRAG(api_key=api_key)
                result = rag.query(
                    symptoms  = intake_data["symptoms"],
                    condition = intake_data["condition"],
                )
            render_rag_results(result)
        else:
            st.warning("Set ANTHROPIC_API_KEY to generate evidence report.")
