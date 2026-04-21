"""
OcuTrace — LLM Clinical Narrator
=================================
Takes structured biomarker JSON from the diff engine and produces
a clinician-grade progression report using the Claude API.

Usage:
    from narrator import OcuTraceNarrator

    narrator = OcuTraceNarrator(api_key="sk-ant-...")
    report = narrator.generate(biomarkers_json_or_path)
    print(report.summary)
    print(report.recommendation)

Standalone:
    python narrator.py ocutrace_output/biomarkers.json
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import anthropic


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClinicalReport:
    """Structured output from the LLM narrator."""
    summary:          str   # 2-sentence progression narrative
    risk_level:       str   # "low" | "moderate" | "high"
    risk_rationale:   str   # 1-sentence explanation of risk
    recommendation:   str   # injection interval recommendation
    watch_next_visit: str   # what to look for at next appointment
    raw_response:     str   # full LLM output for debugging

    def to_dict(self) -> dict:
        return {
            "summary":          self.summary,
            "risk_level":       self.risk_level,
            "risk_rationale":   self.risk_rationale,
            "recommendation":   self.recommendation,
            "watch_next_visit": self.watch_next_visit,
        }

    def pretty_print(self):
        risk_icons = {"low": "🟢", "moderate": "🟡", "high": "🔴"}
        icon = risk_icons.get(self.risk_level, "⚪")
        print("\n" + "─" * 64)
        print("OcuTrace — Clinical Report")
        print("─" * 64)
        print(f"\nSummary:\n  {self.summary}")
        print(f"\nRisk level: {icon} {self.risk_level.upper()}")
        print(f"  {self.risk_rationale}")
        print(f"\nRecommendation:\n  {self.recommendation}")
        print(f"\nWatch at next visit:\n  {self.watch_next_visit}")
        print("─" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert retinal specialist AI assistant embedded in \
OcuTrace, a longitudinal OCT analysis system for Retinal Vein Occlusion (RVO).

You receive structured biomarker data comparing a patient's OCT scan at two timepoints \
(T1 = baseline, T2 = follow-up) after anti-VEGF treatment for RVO-associated macular edema.

Your job is to produce a concise, clinician-grade progression report. Write as a \
senior retina specialist would — precise, direct, no unnecessary hedging.

Biomarker reference guide:
- CRT (Central Retinal Thickness, µm): Normal <250µm. Elevated = active edema.
- IRF (Intraretinal Fluid, mm³): Should reduce with treatment. Persistent IRF = poor prognosis.
- SRF (Subretinal Fluid, mm³): Can persist longer than IRF; less damaging acutely.
- DRIL (Disorganization of Retinal Inner Layers, %): Higher = more ischemic damage. Does NOT reverse.
- EZ integrity (0–1): Ellipsoid zone continuity. Lower = photoreceptor damage. Partially reversible.

Risk stratification for edema recurrence:
- Low:      CRT <300µm AND IRF reduced >50% AND DRIL stable or improved
- Moderate: CRT 300–400µm OR IRF reduced <50% OR DRIL worsening 1–5%
- High:     CRT >400µm OR new/persistent IRF OR DRIL worsening >5%

Injection interval guidance (treat-and-extend for RVO):
- Low risk:      Can extend to 8–10 weeks
- Moderate risk: Maintain current interval (6–8 weeks)
- High risk:     Shorten to 4–6 weeks or consider rescue injection

Respond ONLY with a valid JSON object. No preamble, no explanation, no markdown fences.
The JSON must have exactly these keys:
{
  "summary": "<2 sentences: what changed structurally and functionally>",
  "risk_level": "<low|moderate|high>",
  "risk_rationale": "<1 sentence: the specific finding that drives this risk level>",
  "recommendation": "<1–2 sentences: next injection timing and rationale>",
  "watch_next_visit": "<1 sentence: the single most important thing to monitor>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_user_prompt(data: dict) -> str:
    """
    Convert biomarker delta dict into a readable prompt for the model.
    Formats numbers clearly and adds clinical context.
    """
    dates = data.get("visit_dates", ["T1", "T2"])
    b1    = data["biomarkers_t1"]
    b2    = data["biomarkers_t2"]
    deltas = data.get("biomarker_deltas", {})

    def fmt_delta(key: str) -> str:
        if key not in deltas:
            return "N/A"
        d = deltas[key]
        sign = "↓" if d["delta_abs"] < 0 else "↑" if d["delta_abs"] > 0 else "="
        return f"{d['t1']} → {d['t2']} ({sign}{abs(d['delta_pct']):.1f}%)"

    # Determine RVO type from context if available
    rvo_type = data.get("rvo_type", "RVO (type unspecified)")
    injections = data.get("injections_between_visits", "unknown number of")

    lines = [
        f"Patient data — {rvo_type}",
        f"Visit 1 ({dates[0]}) → Visit 2 ({dates[1]})",
        f"Treatment: {injections} anti-VEGF injections between visits",
        "",
        "Biomarker changes (T1 → T2):",
        f"  CRT (µm):       {fmt_delta('crt_um')}",
        f"  IRF (mm³):      {fmt_delta('irf_mm3')}",
        f"  SRF (mm³):      {fmt_delta('srf_mm3')}",
        f"  DRIL (%):       {fmt_delta('dril_pct')}",
        f"  EZ integrity:   {fmt_delta('ez_integrity')}",
        "",
        "Current values at T2:",
        f"  CRT = {b2['crt_um']} µm",
        f"  IRF = {b2['irf_mm3']} mm³",
        f"  SRF = {b2['srf_mm3']} mm³",
        f"  DRIL = {b2['dril_pct']}%",
        f"  EZ integrity = {b2['ez_integrity']}",
        "",
        "Generate the clinical report JSON.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# NARRATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class OcuTraceNarrator:
    """
    Wraps the Claude API to generate clinical progression reports.

    Args:
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        model:   Claude model to use.
    """

    def __init__(
        self,
        api_key:  Optional[str] = None,
        model:    str = "claude-sonnet-4-6",
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "No API key provided. Set ANTHROPIC_API_KEY env var "
                "or pass api_key= to OcuTraceNarrator()."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.model  = model

    def generate(
        self,
        source: str | Path | dict,
        rvo_type: str = "BRVO",
        injections_between_visits: str = "2",
    ) -> ClinicalReport:
        """
        Generate a clinical report from biomarker data.

        Args:
            source:                    Path to biomarkers.json, JSON string, or dict.
            rvo_type:                  "BRVO" or "CRVO"
            injections_between_visits: Number of injections (for context)

        Returns:
            ClinicalReport dataclass
        """
        # Load data
        data = self._load_data(source)
        data["rvo_type"] = rvo_type
        data["injections_between_visits"] = injections_between_visits

        user_prompt = _build_user_prompt(data)

        print(f"[OcuTrace Narrator] Calling {self.model}...")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = response.content[0].text.strip()
        return self._parse_response(raw)

    def generate_from_diff_result(self, diff_result, **kwargs) -> ClinicalReport:
        """
        Convenience method — pass a DiffResult object directly.
        """
        data = json.loads(diff_result.to_json())
        return self.generate(data, **kwargs)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_data(self, source: str | Path | dict) -> dict:
        if isinstance(source, dict):
            return source
        source = Path(source) if not isinstance(source, Path) else source
        if source.exists():
            return json.loads(source.read_text())
        # Maybe it's a JSON string
        try:
            return json.loads(str(source))
        except json.JSONDecodeError:
            raise ValueError(f"Cannot load biomarker data from: {source}")

    def _parse_response(self, raw: str) -> ClinicalReport:
        """Parse JSON response, with fallback extraction if model adds fluff."""
        # Strip markdown fences if present (defensive)
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract JSON object from the text
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = {}

        # Validate and apply fallbacks for any missing keys
        risk = parsed.get("risk_level", "moderate").lower()
        if risk not in ("low", "moderate", "high"):
            risk = "moderate"

        return ClinicalReport(
            summary          = parsed.get("summary",
                               "Biomarker changes detected between visits. Manual review recommended."),
            risk_level       = risk,
            risk_rationale   = parsed.get("risk_rationale",
                               "Risk level based on overall biomarker trajectory."),
            recommendation   = parsed.get("recommendation",
                               "Maintain current treatment interval pending physician review."),
            watch_next_visit = parsed.get("watch_next_visit",
                               "Monitor central retinal thickness and fluid status."),
            raw_response     = raw,
        )


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: RULE-BASED NARRATOR (no API key required)
# ─────────────────────────────────────────────────────────────────────────────

def rule_based_report(data: dict) -> ClinicalReport:
    """
    Deterministic clinical report when no API key is available.
    Uses the same risk stratification rules as the system prompt.
    Good for demos and CI testing.
    """
    b2     = data.get("biomarkers_t2", {})
    deltas = data.get("biomarker_deltas", {})

    crt      = b2.get("crt_um", 300)
    irf      = b2.get("irf_mm3", 0)
    dril     = b2.get("dril_pct", 0)
    ez       = b2.get("ez_integrity", 1.0)

    irf_delta_pct  = deltas.get("irf_mm3", {}).get("delta_pct", 0)
    dril_delta_pct = deltas.get("dril_pct", {}).get("delta_pct", 0)
    crt_delta_um   = deltas.get("crt_um",   {}).get("delta_abs", 0)

    # Risk stratification
    if crt > 400 or (irf > 0 and irf_delta_pct > -10) or dril_delta_pct > 5:
        risk = "high"
        interval = "4–6 weeks — consider rescue injection if CRT continues to rise"
    elif crt > 300 or irf_delta_pct > -50 or (1 < dril_delta_pct <= 5):
        risk = "moderate"
        interval = "maintain current interval (6–8 weeks)"
    else:
        risk = "low"
        interval = "extend to 8–10 weeks given favourable response"

    # Summary sentence
    crt_dir  = "decreased" if crt_delta_um < 0 else "increased"
    crt_mag  = abs(crt_delta_um)
    irf_dir  = "reduced" if irf_delta_pct < 0 else "increased"
    irf_mag  = abs(irf_delta_pct)

    summary = (
        f"Central retinal thickness {crt_dir} by {crt_mag:.0f}µm "
        f"and intraretinal fluid {irf_dir} by {irf_mag:.0f}% between visits, "
        f"indicating {'partial structural improvement' if crt_delta_um < 0 else 'ongoing disease activity'}. "
        f"DRIL extent is {'stable' if abs(dril_delta_pct) < 2 else 'worsening'}, "
        f"and EZ integrity {'is preserved' if ez > 0.8 else 'shows early disruption'}."
    )

    risk_rationale_map = {
        "high":     f"CRT remains elevated at {crt:.0f}µm with insufficient fluid reduction.",
        "moderate": f"Partial treatment response — IRF reduced {irf_mag:.0f}% but edema persists.",
        "low":      "Strong treatment response with significant fluid resolution.",
    }

    watch_map = {
        "high":     "Monitor CRT closely — rising trend warrants early re-assessment.",
        "moderate": "Track DRIL progression and EZ integrity at each visit.",
        "low":      "Confirm stability before extending injection interval further.",
    }

    return ClinicalReport(
        summary          = summary,
        risk_level       = risk,
        risk_rationale   = risk_rationale_map[risk],
        recommendation   = f"Recommended next injection window: {interval}.",
        watch_next_visit = watch_map[risk],
        raw_response     = "[rule-based — no API call made]",
    )


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Accept JSON file path or use built-in demo data
    if len(sys.argv) > 1:
        source = Path(sys.argv[1])
        if not source.exists():
            print(f"Error: file not found: {source}")
            sys.exit(1)
        data = json.loads(source.read_text())
    else:
        print("[OcuTrace Narrator] No JSON file provided — using demo data.\n"
              "  Usage: python narrator.py ocutrace_output/biomarkers.json\n")
        # Realistic demo patient — BRVO, 2 anti-VEGF injections
        data = {
            "visit_dates":   ["2024-01-14", "2024-03-02"],
            "biomarkers_t1": {
                "crt_um":       412.0,
                "irf_mm3":      2.3,
                "srf_mm3":      0.8,
                "ped_mm3":      0.1,
                "irf_pct":      4.5,
                "srf_pct":      1.6,
                "dril_pct":     18.2,
                "ez_integrity": 0.72,
            },
            "biomarkers_t2": {
                "crt_um":       318.0,
                "irf_mm3":      0.9,
                "srf_mm3":      0.2,
                "ped_mm3":      0.05,
                "irf_pct":      1.8,
                "srf_pct":      0.4,
                "dril_pct":     22.1,
                "ez_integrity": 0.68,
            },
            "biomarker_deltas": {
                "crt_um":       {"t1": 412.0, "t2": 318.0, "delta_abs": -94.0,  "delta_pct": -22.8, "direction": "improved"},
                "irf_mm3":      {"t1": 2.3,   "t2": 0.9,   "delta_abs": -1.4,   "delta_pct": -60.9, "direction": "improved"},
                "srf_mm3":      {"t1": 0.8,   "t2": 0.2,   "delta_abs": -0.6,   "delta_pct": -75.0, "direction": "improved"},
                "ped_mm3":      {"t1": 0.1,   "t2": 0.05,  "delta_abs": -0.05,  "delta_pct": -50.0, "direction": "improved"},
                "irf_pct":      {"t1": 4.5,   "t2": 1.8,   "delta_abs": -2.7,   "delta_pct": -60.0, "direction": "improved"},
                "srf_pct":      {"t1": 1.6,   "t2": 0.4,   "delta_abs": -1.2,   "delta_pct": -75.0, "direction": "improved"},
                "dril_pct":     {"t1": 18.2,  "t2": 22.1,  "delta_abs": 3.9,    "delta_pct": 21.4,  "direction": "worsened"},
                "ez_integrity": {"t1": 0.72,  "t2": 0.68,  "delta_abs": -0.04,  "delta_pct": -5.6,  "direction": "worsened"},
            },
        }

    # Try LLM narrator first, fall back to rule-based if no API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if api_key:
        narrator = OcuTraceNarrator(api_key=api_key)
        report   = narrator.generate(
            data,
            rvo_type="BRVO",
            injections_between_visits="2",
        )
        print("[OcuTrace Narrator] Generated via Claude API")
    else:
        print("[OcuTrace Narrator] No ANTHROPIC_API_KEY found — using rule-based fallback.\n"
              "  Set ANTHROPIC_API_KEY=sk-ant-... to use Claude.\n")
        report = rule_based_report(data)

    report.pretty_print()

    # Save report JSON
    out_dir = Path("ocutrace_output")
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "clinical_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"Report saved → {report_path}")
