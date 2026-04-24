"""
OcuTrace — Streamlit UI with RAG Integration
=============================================
Full application: OCT scan analysis + PubMed RAG evidence search.

Run:
    streamlit run app.py

Requires:
    pip install streamlit plotly anthropic pypdf
"""

import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from diff_engine import OcuTraceDiffEngine, generate_synthetic_pair
from narrator import OcuTraceNarrator, rule_based_report
from confidence import score_biomarker_deltas
from intake import render_intake_form, render_rag_results

st.set_page_config(
    page_title="OcuTrace",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #0E1A20; color: #F2F4F3; }
[data-testid="stSidebar"] { background-color: #111C24; border-right: 1px solid #1C3040; }
.oc-card { background: #162028; border: 1px solid #1C3040; border-radius: 10px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
.oc-card-accent-green  { border-left: 4px solid #00C9A7; }
.oc-card-accent-red    { border-left: 4px solid #E05C5C; }
.oc-card-accent-amber  { border-left: 4px solid #D4954A; }
.oc-card-accent-mint   { border-left: 4px solid #00C9A7; }
.oc-metric-val   { font-size: 2.2rem; font-weight: 700; font-family: 'Trebuchet MS', sans-serif; }
.oc-metric-label { font-size: 0.78rem; color: #6B8080; margin-top: -4px; }
.oc-delta-good   { color: #3DBD8A; font-size: 0.9rem; }
.oc-delta-warn   { color: #E05C5C; font-size: 0.9rem; }
.oc-delta-neu    { color: #D4954A; font-size: 0.9rem; }
.risk-low      { background:#1A3A2A; color:#3DBD8A; padding:4px 14px; border-radius:20px; font-weight:600; }
.risk-moderate { background:#3A2A0A; color:#D4954A; padding:4px 14px; border-radius:20px; font-weight:600; }
.risk-high     { background:#3A0A0A; color:#E05C5C; padding:4px 14px; border-radius:20px; font-weight:600; }
.oc-rule { border:none; border-top:1px solid #1C3040; margin:1.2rem 0; }
.conf-high     { color: #3DBD8A; font-weight: 600; }
.conf-moderate { color: #D4954A; font-weight: 600; }
.conf-low      { color: #E05C5C; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

for key in ["diff_result", "report", "confidence_report", "rag_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

@st.cache_resource
def load_engine(weights_path=None):
    return OcuTraceDiffEngine(weights_path=weights_path)

def delta_html(value, pct, higher_is_bad=True):
    sign = "↓" if value < 0 else "↑" if value > 0 else "="
    good = (value < 0 and higher_is_bad) or (value > 0 and not higher_is_bad)
    cls  = "oc-delta-good" if good else "oc-delta-warn" if value != 0 else "oc-delta-neu"
    return f'<span class="{cls}">{sign} {abs(pct):.1f}%</span>'

def risk_badge(level):
    return f'<span class="risk-{level}">{level.upper()}</span>'

def save_temp(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        return Path(f.name)

def make_trajectory_plotly(deltas, dates):
    metrics = [
        ("crt_um","CRT (µm)","#00C9A7"),
        ("irf_mm3","IRF (mm³)","#E05C5C"),
        ("dril_pct","DRIL (%)","#D4954A"),
        ("ez_integrity","EZ integrity","#3DBD8A"),
    ]
    fig = go.Figure()
    for key, label, color in metrics:
        if key not in deltas:
            continue
        d = deltas[key]
        t1_val, t2_val = d["t1"], d["t2"]
        slope = t2_val - t1_val
        fig.add_trace(go.Scatter(
            x=dates[:2], y=[t1_val, t2_val],
            mode="lines+markers", name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
        ))
        fig.add_trace(go.Scatter(
            x=[dates[1], "~Next visit"], y=[t2_val, t2_val + slope],
            mode="lines+markers",
            line=dict(color=color, width=1.5, dash="dot"),
            marker=dict(size=7, color=color, symbol="diamond", opacity=0.6),
            showlegend=False,
        ))
    fig.update_layout(
        paper_bgcolor="#0E1A20", plot_bgcolor="#111C24",
        font=dict(color="#A8B5B0", size=11),
        legend=dict(bgcolor="#162028", bordercolor="#1C3040", borderwidth=1, font=dict(color="#F2F4F3")),
        xaxis=dict(gridcolor="#1C3040", title="Visit"),
        yaxis=dict(gridcolor="#1C3040", title="Value"),
        margin=dict(l=40, r=20, t=20, b=40), height=340,
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## OcuTrace")
    st.markdown('<hr class="oc-rule">', unsafe_allow_html=True)
    st.markdown("#### Patient context")
    rvo_type     = st.selectbox("RVO type", ["BRVO", "CRVO", "Hemi-RVO"], index=0)
    visit_date_1 = st.date_input("Visit 1 date")
    visit_date_2 = st.date_input("Visit 2 date")
    injections   = st.number_input("Injections between visits", min_value=0, max_value=10, value=2)
    st.markdown('<hr class="oc-rule">', unsafe_allow_html=True)
    st.markdown("#### Model settings")
    weights_path  = st.text_input("RETOUCH weights path (optional)", placeholder="/path/to/retouch_unet.pth")
    api_key_input = st.text_input("Anthropic API key (optional)", type="password",
                                   placeholder="sk-ant-...", value=os.environ.get("ANTHROPIC_API_KEY", ""))
    st.markdown('<hr class="oc-rule">', unsafe_allow_html=True)
    st.markdown("#### Demo mode")
    use_synthetic = st.checkbox("Use synthetic patient data", value=False)
    st.markdown('<hr class="oc-rule">', unsafe_allow_html=True)
    st.caption("OcuTrace · April 2025")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_scan, tab_rag = st.tabs(["🔬  Scan Analysis", "📚  Evidence Search (RAG)"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: SCAN ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_scan:
    st.markdown("# OcuTrace")
    st.markdown('<p style="color:#00C9A7;font-size:1rem;margin-top:-8px;">Longitudinal OCT progression analysis for Retinal Vein Occlusion</p>', unsafe_allow_html=True)
    st.markdown('<hr class="oc-rule">', unsafe_allow_html=True)

    if not use_synthetic:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Visit 1 — baseline scan**")
            upload_t1 = st.file_uploader("Upload OCT scan", key="t1", type=["png","jpg","jpeg","dcm","mhd"])
        with col2:
            st.markdown("**Visit 2 — follow-up scan**")
            upload_t2 = st.file_uploader("Upload OCT scan", key="t2", type=["png","jpg","jpeg","dcm","mhd"])
        scans_ready = upload_t1 is not None and upload_t2 is not None
    else:
        upload_t1 = upload_t2 = None
        scans_ready = True
        st.info("Demo mode: synthetic RVO patient data will be generated automatically.")

    run_col, _ = st.columns([2, 6])
    with run_col:
        run_clicked = st.button("▶  Run analysis", type="primary", disabled=not scans_ready, use_container_width=True)

    if run_clicked and scans_ready:
        engine = load_engine(weights_path or None)
        dates  = [str(visit_date_1), str(visit_date_2)]

        with st.spinner("Running pipeline — registration → segmentation → diff…"):
            try:
                if use_synthetic:
                    t1_arr, t2_arr = generate_synthetic_pair(512, 512, seed=42)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1, \
                         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
                        Image.fromarray((t1_arr * 255).astype(np.uint8)).save(f1.name)
                        Image.fromarray((t2_arr * 255).astype(np.uint8)).save(f2.name)
                        t1_path, t2_path = Path(f1.name), Path(f2.name)
                else:
                    t1_path = save_temp(upload_t1)
                    t2_path = save_temp(upload_t2)

                result = engine.run(t1_path, t2_path, visit_dates=dates)
                st.session_state.diff_result = result
                st.success("Pipeline complete.")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        with st.spinner("Scoring confidence…"):
            try:
                st.session_state.confidence_report = score_biomarker_deltas(
                    result.biomarker_deltas, result.diff_map
                )
            except Exception:
                st.session_state.confidence_report = None

        with st.spinner("Generating clinical report…"):
            try:
                key = api_key_input or os.environ.get("ANTHROPIC_API_KEY", "")
                if key:
                    narrator = OcuTraceNarrator(api_key=key)
                    report   = narrator.generate_from_diff_result(result, rvo_type=rvo_type,
                                                                   injections_between_visits=str(injections))
                else:
                    data = json.loads(result.to_json())
                    data["rvo_type"] = rvo_type
                    report = rule_based_report(data)
                    st.caption("ℹ️ No API key — using rule-based report.")
                st.session_state.report = report
            except Exception as e:
                st.warning(f"Report error: {e}")
                st.session_state.report = None

    result = st.session_state.diff_result
    report = st.session_state.report
    conf   = st.session_state.confidence_report

    if result is None:
        st.markdown('<div class="oc-card oc-card-accent-mint" style="text-align:center;padding:2.5rem;"><p style="color:#6B8080;font-size:1.05rem;">Upload two OCT scans and click Run analysis to begin.</p></div>', unsafe_allow_html=True)
    else:
        dates  = result.visit_dates or ["Visit 1", "Visit 2"]
        deltas = result.biomarker_deltas

        st.markdown("### Scan comparison")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**T1 · {dates[0]}**")
            st.image(result.overlay_t1, use_container_width=True, caption=f"CRT {result.biomarkers_t1['crt_um']:.0f}µm")
        with c2:
            st.markdown(f"**T2 · {dates[1]}**")
            st.image(result.overlay_t2, use_container_width=True, caption=f"CRT {result.biomarkers_t2['crt_um']:.0f}µm")
        with c3:
            st.markdown("**Diff overlay**")
            st.image(result.overlay_diff, use_container_width=True, caption="Green = resolved · Red = new / worsening")

        st.markdown('<hr class="oc-rule">', unsafe_allow_html=True)
        st.markdown("### Biomarker changes")
        metric_cols = st.columns(5)
        metrics_display = [("CRT","crt_um","µm",True),("IRF","irf_mm3","mm³",True),
                           ("SRF","srf_mm3","mm³",True),("DRIL","dril_pct","%",True),("EZ","ez_integrity","",False)]
        for col, (label, key, unit, higher_bad) in zip(metric_cols, metrics_display):
            with col:
                d         = deltas.get(key, {})
                t2_val    = d.get("t2", result.biomarkers_t2.get(key, 0))
                delta_abs = d.get("delta_abs", 0)
                delta_pct = d.get("delta_pct", 0)
                dhtml     = delta_html(delta_abs, delta_pct, higher_is_bad=higher_bad)
                conf_str  = ""
                if conf and key in conf.scores:
                    mc = conf.scores[key]
                    conf_str = f'<div class="conf-{mc.level}" style="font-size:0.75rem;margin-top:4px;">⬤ {mc.level} confidence</div>'
                st.markdown(f'<div class="oc-card"><div class="oc-metric-val">{t2_val:.2f}<span style="font-size:0.9rem;color:#6B8080">{unit}</span></div><div class="oc-metric-label">{label}</div>{dhtml}{conf_str}</div>', unsafe_allow_html=True)

        if conf:
            st.markdown(f'<p style="color:#6B8080;font-size:0.85rem;">Overall confidence: <span class="conf-{conf.overall_level}">{conf.overall_confidence*100:.0f}% ({conf.overall_level})</span> · Registration quality: {conf.registration_quality*100:.0f}%</p>', unsafe_allow_html=True)

        st.markdown("### Trajectory")
        st.plotly_chart(make_trajectory_plotly(deltas, dates), use_container_width=True)

        st.markdown('<hr class="oc-rule">', unsafe_allow_html=True)
        st.markdown("### Clinical report")
        if report:
            risk_icons = {"low":"🟢","moderate":"🟡","high":"🔴"}
            r_icon = risk_icons.get(report.risk_level, "⚪")
            accent = {"low":"green","moderate":"amber","high":"red"}.get(report.risk_level, "mint")
            rep1, rep2 = st.columns([3, 2])
            with rep1:
                st.markdown(f'<div class="oc-card oc-card-accent-{accent}"><p style="color:#F2F4F3;font-size:1rem;line-height:1.65;">{report.summary}</p><hr class="oc-rule"><p style="color:#A8B5B0;font-size:0.9rem;">{report.recommendation}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="oc-card"><p style="color:#6B8080;font-size:0.78rem;margin-bottom:4px;">WATCH AT NEXT VISIT</p><p style="color:#F2F4F3;font-size:0.95rem;">{report.watch_next_visit}</p></div>', unsafe_allow_html=True)
            with rep2:
                st.markdown(f'<div class="oc-card" style="text-align:center;padding:2rem 1.2rem;"><p style="color:#6B8080;font-size:0.78rem;">RECURRENCE RISK</p><div style="font-size:2.8rem;">{r_icon}</div>{risk_badge(report.risk_level)}<p style="color:#A8B5B0;font-size:0.85rem;margin-top:1rem;">{report.risk_rationale}</p></div>', unsafe_allow_html=True)

        st.markdown('<hr class="oc-rule">', unsafe_allow_html=True)
        st.markdown("### Export")
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            buf = io.BytesIO()
            Image.fromarray((result.overlay_diff * 255).astype(np.uint8)).save(buf, format="PNG")
            st.download_button("⬇ Diff overlay (PNG)", buf.getvalue(), file_name="ocutrace_diff.png", mime="image/png", use_container_width=True)
        with dl2:
            st.download_button("⬇ Biomarkers (JSON)", result.to_json(), file_name="ocutrace_biomarkers.json", mime="application/json", use_container_width=True)
        with dl3:
            if report:
                st.download_button("⬇ Clinical report (JSON)", json.dumps(report.to_dict(), indent=2), file_name="ocutrace_report.json", mime="application/json", use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: RAG EVIDENCE SEARCH
# ═════════════════════════════════════════════════════════════════════════════
with tab_rag:
    st.markdown("# Evidence Search")
    st.markdown('<p style="color:#00C9A7;font-size:1rem;margin-top:-8px;">Cross-reference symptoms against verified PubMed literature. Every claim is citation-backed — no hallucination.</p>', unsafe_allow_html=True)
    st.markdown('<hr class="oc-rule">', unsafe_allow_html=True)

    api_key     = api_key_input or os.environ.get("ANTHROPIC_API_KEY", "")
    intake_data = render_intake_form(api_key)

    if intake_data:
        biomarkers = None
        if st.session_state.diff_result:
            biomarkers = st.session_state.diff_result.biomarkers_t2
            st.info("Using biomarker values from your scan analysis.")

        with st.spinner("Searching PubMed and generating evidence report…"):
            try:
                if api_key:
                    from pubmed_rag import MedRAG
                    rag    = MedRAG(api_key=api_key)
                    result = rag.query(symptoms=intake_data["symptoms"],
                                       condition=intake_data["condition"],
                                       biomarkers=biomarkers)
                    st.session_state.rag_result = result
                    render_rag_results(result)
                else:
                    from pubmed_rag import build_search_query, search_pubmed, fetch_abstracts
                    query    = build_search_query(intake_data["symptoms"], intake_data["condition"], biomarkers)
                    pmids    = search_pubmed(query, max_results=5)
                    articles = fetch_abstracts(pmids)
                    st.markdown("### Retrieved PubMed articles")
                    st.caption(f"Found {len(articles)} articles · Add an Anthropic API key in the sidebar to generate an AI summary.")
                    for a in articles:
                        st.markdown(
                            f'<div class="oc-card"><p style="color:#00C9A7;font-size:0.8rem;margin:0;">PMID: {a.pmid} · {a.journal} ({a.year})</p>'
                            f'<p style="color:#F2F4F3;font-weight:600;margin:4px 0;">{a.title}</p>'
                            f'<p style="color:#A8B5B0;font-size:0.88rem;">{a.abstract[:300]}…</p>'
                            f'<a href="https://pubmed.ncbi.nlm.nih.gov/{a.pmid}/" target="_blank" style="color:#00C9A7;font-size:0.82rem;">Read on PubMed →</a></div>',
                            unsafe_allow_html=True,
                        )
            except Exception as e:
                st.error(f"RAG error: {e}")
