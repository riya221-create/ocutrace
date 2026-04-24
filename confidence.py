"""
OcuTrace — Confidence Scoring
==============================
Adds statistical confidence scores to every biomarker delta.
Accounts for OCT measurement variability, registration error,
and segmentation uncertainty to give clinicians a reliability signal.

Usage:
    from confidence import score_diff_result, ConfidenceReport

    report = score_diff_result(diff_result)
    print(report.summary_table())
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# OCT MEASUREMENT VARIABILITY (from published literature)
# ─────────────────────────────────────────────────────────────────────────────
# Sources: Tewarie et al. 2012, Menke et al. 2011, CATT study repeatability data

OCT_VARIABILITY = {
    # (coefficient of variation, minimum detectable change)
    "crt_um":       {"cv": 0.03,  "mdc_abs": 11.0,  "unit": "µm"},
    "irf_mm3":      {"cv": 0.08,  "mdc_abs": 0.05,  "unit": "mm³"},
    "srf_mm3":      {"cv": 0.10,  "mdc_abs": 0.04,  "unit": "mm³"},
    "ped_mm3":      {"cv": 0.09,  "mdc_abs": 0.03,  "unit": "mm³"},
    "irf_pct":      {"cv": 0.08,  "mdc_abs": 0.5,   "unit": "%"},
    "srf_pct":      {"cv": 0.10,  "mdc_abs": 0.4,   "unit": "%"},
    "dril_pct":     {"cv": 0.12,  "mdc_abs": 3.0,   "unit": "%"},
    "ez_integrity": {"cv": 0.05,  "mdc_abs": 0.03,  "unit": ""},
}

# Registration error adds ~2-5% uncertainty to spatial measurements
REGISTRATION_UNCERTAINTY = 0.03


@dataclass
class MetricConfidence:
    metric:          str
    delta_abs:       float
    delta_pct:       float
    confidence:      float      # 0.0 – 1.0
    level:           str        # "high" | "moderate" | "low"
    above_mdc:       bool       # is change above minimum detectable change?
    note:            str        # human-readable explanation


@dataclass
class ConfidenceReport:
    scores:              dict[str, MetricConfidence]
    overall_confidence:  float
    overall_level:       str
    registration_quality: float   # 0–1, estimated from diff map entropy

    def summary_table(self) -> str:
        lines = [
            f"{'Metric':<18} {'Delta':>10} {'Confidence':>12}  {'MDC?':>6}  Note",
            "─" * 72,
        ]
        for key, mc in self.scores.items():
            sign  = "↓" if mc.delta_abs < 0 else "↑" if mc.delta_abs > 0 else "="
            delta = f"{sign}{abs(mc.delta_pct):.1f}%"
            conf  = f"{mc.confidence*100:.0f}% ({mc.level})"
            mdc   = "✓" if mc.above_mdc else "✗"
            lines.append(f"{key:<18} {delta:>10} {conf:>12}  {mdc:>6}  {mc.note}")
        lines.append("─" * 72)
        lines.append(
            f"Overall confidence: {self.overall_confidence*100:.0f}% "
            f"({self.overall_level})  |  "
            f"Registration quality: {self.registration_quality*100:.0f}%"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "overall_confidence":   round(self.overall_confidence, 3),
            "overall_level":        self.overall_level,
            "registration_quality": round(self.registration_quality, 3),
            "metrics": {
                k: {
                    "delta_abs":  mc.delta_abs,
                    "delta_pct":  mc.delta_pct,
                    "confidence": round(mc.confidence, 3),
                    "level":      mc.level,
                    "above_mdc":  mc.above_mdc,
                    "note":       mc.note,
                }
                for k, mc in self.scores.items()
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_registration_quality(diff_map: Optional[np.ndarray]) -> float:
    """
    Estimate registration quality from the diff map.
    Well-registered scans have diff concentrated at fluid regions,
    not scattered across the whole image (which indicates misalignment).

    Returns float in [0, 1] — higher is better registration.
    """
    if diff_map is None:
        return 0.75  # assume moderate quality if no diff map

    total_px   = diff_map.size
    changed_px = np.sum(diff_map != 0)
    change_pct = changed_px / total_px

    # If >40% of pixels changed, likely misalignment
    if change_pct > 0.40:
        return 0.45
    elif change_pct > 0.25:
        return 0.65
    elif change_pct > 0.10:
        return 0.80
    else:
        return 0.95


def _score_metric(
    key:          str,
    delta_abs:    float,
    delta_pct:    float,
    t1_val:       float,
    reg_quality:  float,
) -> MetricConfidence:
    """
    Compute confidence score for a single biomarker delta.

    Confidence is reduced by:
    - Small absolute change relative to MDC (below noise floor)
    - High coefficient of variation for this metric
    - Poor registration quality
    - Near-zero baseline (division instability)
    """
    var = OCT_VARIABILITY.get(key, {"cv": 0.10, "mdc_abs": 0.1, "unit": ""})
    mdc      = var["mdc_abs"]
    cv       = var["cv"]
    above_mdc = abs(delta_abs) > mdc

    # Base confidence from signal-to-noise
    if abs(delta_abs) < mdc * 0.5:
        base_conf = 0.35   # well within noise floor
        note = f"Change ({abs(delta_abs):.2f}) below MDC ({mdc}) — likely noise"
    elif abs(delta_abs) < mdc:
        base_conf = 0.55
        note = f"Change near MDC threshold — interpret cautiously"
    elif abs(delta_abs) < mdc * 2:
        base_conf = 0.75
        note = f"Meaningful change, above MDC"
    else:
        base_conf = 0.90
        note = f"Large, reliable change ({abs(delta_pct):.0f}%)"

    # Penalise for high metric variability
    variability_penalty = cv * 0.5
    conf = base_conf - variability_penalty

    # Penalise for poor registration
    reg_penalty = (1.0 - reg_quality) * 0.20
    conf -= reg_penalty

    # Penalise near-zero baseline (percentage unreliable)
    if abs(t1_val) < 1e-3 and abs(delta_abs) > 0:
        conf -= 0.15
        note += " · baseline near zero — % change unreliable"

    conf = float(np.clip(conf, 0.1, 0.97))

    level = "high" if conf >= 0.75 else "moderate" if conf >= 0.50 else "low"

    return MetricConfidence(
        metric     = key,
        delta_abs  = round(delta_abs, 4),
        delta_pct  = round(delta_pct, 2),
        confidence = round(conf, 3),
        level      = level,
        above_mdc  = above_mdc,
        note       = note,
    )


def score_biomarker_deltas(
    deltas:      dict,
    diff_map:    Optional[np.ndarray] = None,
) -> ConfidenceReport:
    """
    Score confidence for all biomarker deltas.

    Args:
        deltas:   biomarker_deltas dict from diff engine (or DiffResult.biomarker_deltas)
        diff_map: signed diff array from diff engine (optional, improves registration estimate)

    Returns:
        ConfidenceReport with per-metric and overall scores
    """
    reg_quality = _estimate_registration_quality(diff_map)

    scores = {}
    for key, d in deltas.items():
        scores[key] = _score_metric(
            key        = key,
            delta_abs  = d.get("delta_abs", 0),
            delta_pct  = d.get("delta_pct", 0),
            t1_val     = d.get("t1", 0),
            reg_quality = reg_quality,
        )

    # Overall = weighted mean, giving more weight to clinically important metrics
    weights = {
        "crt_um":       2.0,
        "irf_mm3":      2.0,
        "dril_pct":     1.5,
        "ez_integrity": 1.5,
        "srf_mm3":      1.0,
        "ped_mm3":      0.8,
        "irf_pct":      1.0,
        "srf_pct":      0.8,
    }
    total_w  = sum(weights.get(k, 1.0) for k in scores)
    overall  = sum(
        scores[k].confidence * weights.get(k, 1.0)
        for k in scores
    ) / total_w if total_w > 0 else 0.5

    overall_level = "high" if overall >= 0.75 else "moderate" if overall >= 0.50 else "low"

    return ConfidenceReport(
        scores               = scores,
        overall_confidence   = round(overall, 3),
        overall_level        = overall_level,
        registration_quality = round(reg_quality, 3),
    )


def score_diff_result(diff_result) -> ConfidenceReport:
    """
    Convenience wrapper — pass a DiffResult object directly.
    """
    return score_biomarker_deltas(
        deltas   = diff_result.biomarker_deltas,
        diff_map = diff_result.diff_map,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate deltas from a real patient
    deltas = {
        "crt_um":       {"t1": 412.0, "t2": 318.0, "delta_abs": -94.0,  "delta_pct": -22.8},
        "irf_mm3":      {"t1": 2.3,   "t2": 0.9,   "delta_abs": -1.4,   "delta_pct": -60.9},
        "srf_mm3":      {"t1": 0.8,   "t2": 0.2,   "delta_abs": -0.6,   "delta_pct": -75.0},
        "ped_mm3":      {"t1": 0.1,   "t2": 0.05,  "delta_abs": -0.05,  "delta_pct": -50.0},
        "irf_pct":      {"t1": 4.5,   "t2": 1.8,   "delta_abs": -2.7,   "delta_pct": -60.0},
        "srf_pct":      {"t1": 1.6,   "t2": 0.4,   "delta_abs": -1.2,   "delta_pct": -75.0},
        "dril_pct":     {"t1": 18.2,  "t2": 22.1,  "delta_abs": 3.9,    "delta_pct": 21.4},
        "ez_integrity": {"t1": 0.72,  "t2": 0.68,  "delta_abs": -0.04,  "delta_pct": -5.6},
    }

    # Simulate a diff map (30% of pixels changed — moderate registration)
    rng      = np.random.default_rng(42)
    diff_map = rng.choice([-1, 0, 0, 0, 1], size=(256, 256)).astype(np.int8)

    report = score_biomarker_deltas(deltas, diff_map)
    print(report.summary_table())
