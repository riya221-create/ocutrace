"""
OcuTrace — Diff Engine
======================
Core pipeline: OCT scan registration → fluid segmentation → signed diff map
→ biomarker extraction → color overlay visualization.

Usage:
    from diff_engine import OcuTraceDiffEngine

    engine = OcuTraceDiffEngine()
    result = engine.run(scan_t1_path, scan_t2_path, visit_dates=["2024-01-14", "2024-03-02"])
    print(result.biomarkers)
    result.save_overlay("output/diff_overlay.png")
    result.save_trajectory("output/trajectory.png")

Requirements:
    pip install SimpleITK torch torchvision numpy scipy matplotlib pillow tqdm
"""

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

warnings.filterwarnings("ignore", category=UserWarning)


def _safe_print(*args, **kwargs):
    """
    Print without crashing when stdout uses a legacy Windows encoding.
    """
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    file = kwargs.get("file", sys.stdout)
    flush = kwargs.get("flush", False)
    message = sep.join(str(arg) for arg in args) + end

    try:
        file.write(message)
    except UnicodeEncodeError:
        encoding = getattr(file, "encoding", None) or "utf-8"
        file.write(message.encode(encoding, errors="replace").decode(encoding, errors="replace"))

    if flush:
        file.flush()

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LIGHTWEIGHT U-NET FOR FLUID SEGMENTATION
#    Designed to run inference on CPU or GPU without pretrained weights.
#    When RETOUCH weights are available, load them with load_retouch_weights().
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Lightweight U-Net for OCT fluid segmentation.
    Outputs 4 channels: [background, IRF, SRF, PED]
    Architecture matches RETOUCH challenge baseline for weight compatibility.
    """
    def __init__(self, in_channels=1, out_channels=4, features=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)

        # Encoder
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)

    def load_retouch_weights(self, path: str):
        """Load pretrained RETOUCH challenge weights if available."""
        state = torch.load(path, map_location=DEVICE)
        # Handle common checkpoint formats
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        self.load_state_dict(state, strict=False)
        _safe_print(f"[OcuTrace] Loaded weights from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. OCT IMAGE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_oct_scan(path: str | Path) -> np.ndarray:
    """
    Load an OCT B-scan from file.
    Supports: PNG, JPG, BMP (single B-scan) or DICOM/MHD/NRRD (volume).
    Returns: float32 array, shape (H, W), values in [0, 1].
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    # DICOM / ITK-compatible formats
    try:
        itk_img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(itk_img).astype(np.float32)
        # Normalize to [0, 1]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        # If 3D volume, return middle slice as representative B-scan
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        return arr
    except Exception as e:
        raise ValueError(
            f"Could not load OCT scan from {path}. "
            f"Supported: PNG/JPG/BMP (2D B-scan) or DICOM/MHD/NRRD (3D volume). "
            f"Error: {e}"
        )


def preprocess_for_model(arr: np.ndarray, target_size=(512, 512)) -> torch.Tensor:
    """Resize + normalize + add batch/channel dims."""
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize(target_size, Image.BILINEAR)
    t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SCAN REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

def register_scans(
    fixed: np.ndarray,
    moving: np.ndarray,
    fovea_hint: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Affine registration of moving → fixed using SimpleITK.

    Uses mutual information as the metric so it works across different
    scan sessions (different brightness, contrast, slight angle changes).

    Args:
        fixed:      Reference scan (T1), shape (H, W), float32 [0,1]
        moving:     Scan to align (T2), same shape
        fovea_hint: Optional (row, col) of foveal center in fixed scan
                    for better initialization. Auto-detected if None.

    Returns:
        registered: Moving scan aligned to fixed, same shape as fixed.
    """
    h, w = fixed.shape

    # Convert to SimpleITK images
    fixed_itk  = sitk.GetImageFromArray((fixed  * 255).astype(np.uint8))
    moving_itk = sitk.GetImageFromArray((moving * 255).astype(np.uint8))

    # Cast to float for registration
    fixed_f  = sitk.Cast(fixed_itk,  sitk.sitkFloat32)
    moving_f = sitk.Cast(moving_itk, sitk.sitkFloat32)

    # Initialize transform — center-of-image alignment first
    tx = sitk.CenteredTransformInitializer(
        fixed_f, moving_f,
        sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Registration method
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)

    # Optimizer — gradient descent with good defaults for OCT
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution pyramid
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    reg.SetInitialTransform(tx, inPlace=False)

    final_tx = reg.Execute(fixed_f, moving_f)

    # Apply transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_itk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_tx)

    registered_itk = resampler.Execute(moving_itk)
    registered = sitk.GetArrayFromImage(registered_itk).astype(np.float32) / 255.0

    return registered


# ─────────────────────────────────────────────────────────────────────────────
# 4. FLUID SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

LABEL_NAMES = {0: "background", 1: "IRF", 2: "SRF", 3: "PED"}
LABEL_COLORS = {
    0: (0.12, 0.12, 0.12),   # background — near-black
    1: (0.94, 0.36, 0.27),   # IRF — coral red
    2: (0.24, 0.73, 0.55),   # SRF — green
    3: (0.96, 0.65, 0.14),   # PED — amber
}


@torch.no_grad()
def segment_fluid(
    model: UNet,
    scan: np.ndarray,
    original_shape: Optional[tuple] = None,
) -> np.ndarray:
    """
    Run U-Net inference on a single B-scan.

    Returns:
        label_map: int32 array same shape as scan,
                   values: 0=bg, 1=IRF, 2=SRF, 3=PED
    """
    model.eval()
    h_orig, w_orig = scan.shape if original_shape is None else original_shape
    tensor = preprocess_for_model(scan)           # (1, 1, 512, 512)
    logits = model(tensor)                        # (1, 4, 512, 512)
    probs  = torch.softmax(logits, dim=1)
    pred   = torch.argmax(probs, dim=1).squeeze() # (512, 512)
    label_map = pred.cpu().numpy().astype(np.int32)

    # Resize back to original scan dimensions
    label_img = Image.fromarray(label_map.astype(np.uint8))
    label_img = label_img.resize((w_orig, h_orig), Image.NEAREST)
    return np.array(label_img, dtype=np.int32)


def extract_biomarkers(label_map: np.ndarray, px_area_mm2: float = 1e-4) -> dict:
    """
    Extract quantitative biomarkers from a segmentation label map.

    Args:
        label_map:    int32 array (H, W), values 0–3
        px_area_mm2:  Area of one pixel in mm².
                      Default 1e-4 ≈ typical OCT resolution (10µm × 10µm).

    Returns dict with:
        crt_um        Central retinal thickness in µm (proxy via fluid extent)
        irf_mm3       Intraretinal fluid volume
        srf_mm3       Subretinal fluid volume
        ped_mm3       Pigment epithelial detachment volume
        irf_pct       IRF as % of scan area
        srf_pct       SRF as % of scan area
        dril_pct      Disorganization of retinal inner layers (heuristic)
        ez_integrity  Ellipsoid zone integrity score [0–1]
    """
    h, w = label_map.shape
    total_px = h * w

    irf_px = np.sum(label_map == 1)
    srf_px = np.sum(label_map == 2)
    ped_px = np.sum(label_map == 3)

    irf_mm3 = float(irf_px * px_area_mm2)
    srf_mm3 = float(srf_px * px_area_mm2)
    ped_mm3 = float(ped_px * px_area_mm2)

    irf_pct = float(irf_px / total_px * 100)
    srf_pct = float(srf_px / total_px * 100)

    # CRT proxy: sum of fluid rows in central 20% of width
    cx = w // 2
    cw = w // 5
    central_col = label_map[:, cx - cw : cx + cw]
    fluid_rows  = np.any(central_col > 0, axis=1)
    crt_um      = float(np.sum(fluid_rows) * 10)  # 10µm per pixel row

    # DRIL heuristic: disruption in inner retinal layers (rows 20–60% of height)
    inner_start = int(h * 0.20)
    inner_end   = int(h * 0.60)
    inner_region = label_map[inner_start:inner_end, :]
    inner_fluid  = np.sum(inner_region > 0)
    inner_total  = inner_region.size
    dril_pct     = float(inner_fluid / inner_total * 100)

    # EZ integrity: rows 65–75% of height should be uninterrupted (bright band)
    # Proxy: absence of fluid disruption in that zone
    ez_start = int(h * 0.65)
    ez_end   = int(h * 0.75)
    ez_region = label_map[ez_start:ez_end, :]
    ez_disrupted = np.sum(ez_region > 0)
    ez_total     = ez_region.size
    ez_integrity = float(1.0 - ez_disrupted / ez_total)

    return {
        "crt_um":       round(crt_um, 1),
        "irf_mm3":      round(irf_mm3, 4),
        "srf_mm3":      round(srf_mm3, 4),
        "ped_mm3":      round(ped_mm3, 4),
        "irf_pct":      round(irf_pct, 2),
        "srf_pct":      round(srf_pct, 2),
        "dril_pct":     round(dril_pct, 2),
        "ez_integrity": round(ez_integrity, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIGNED DIFF ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiffResult:
    """All outputs from one T1→T2 comparison."""
    scan_t1:          np.ndarray           # original T1 scan
    scan_t2:          np.ndarray           # original T2 scan (registered)
    label_t1:         np.ndarray           # segmentation at T1
    label_t2:         np.ndarray           # segmentation at T2
    diff_map:         np.ndarray           # signed diff: +1 new fluid, -1 resolved, 0 stable
    overlay_t1:       np.ndarray           # RGB overlay on T1
    overlay_t2:       np.ndarray           # RGB overlay on T2
    overlay_diff:     np.ndarray           # RGB diff overlay on T2 scan
    biomarkers_t1:    dict
    biomarkers_t2:    dict
    biomarker_deltas: dict                 # t2 - t1 for each metric
    visit_dates:      list[str] = field(default_factory=list)

    def save_overlay(self, path: str | Path, dpi: int = 150):
        """Save the 3-panel comparison figure."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = _make_comparison_figure(self)
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor="#0E1A20", edgecolor="none")
        plt.close(fig)
        _safe_print(f"[OcuTrace] Overlay saved -> {path}")

    def save_trajectory(self, path: str | Path, history: Optional[list] = None, dpi: int = 150):
        """Save biomarker trajectory plot. Pass history list for >2 timepoints."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = _make_trajectory_figure(self, history)
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor="#0E1A20", edgecolor="none")
        plt.close(fig)
        _safe_print(f"[OcuTrace] Trajectory saved -> {path}")

    def to_json(self) -> str:
        """Return structured JSON ready to pass to the LLM narrator."""
        payload = {
            "visit_dates":      self.visit_dates,
            "biomarkers_t1":    self.biomarkers_t1,
            "biomarkers_t2":    self.biomarkers_t2,
            "biomarker_deltas": self.biomarker_deltas,
        }
        return json.dumps(payload, indent=2)


def compute_diff(label_t1: np.ndarray, label_t2: np.ndarray) -> np.ndarray:
    """
    Compute signed pixel-level diff between two segmentation maps.

    Returns int8 map:
       +1  →  new fluid at T2 (worsening / new lesion)
        0  →  no change
       -1  →  fluid resolved at T2 (improvement)
    """
    fluid_t1 = (label_t1 > 0).astype(np.int8)
    fluid_t2 = (label_t2 > 0).astype(np.int8)
    return (fluid_t2 - fluid_t1).astype(np.int8)


def compute_biomarker_deltas(b1: dict, b2: dict) -> dict:
    """Compute absolute and percentage changes for each biomarker."""
    deltas = {}
    for key in b1:
        v1, v2 = b1[key], b2[key]
        delta_abs = round(v2 - v1, 4)
        delta_pct = round((v2 - v1) / (v1 + 1e-8) * 100, 1) if v1 != 0 else 0.0
        direction = "improved" if (key in ("crt_um", "irf_mm3", "srf_mm3", "ped_mm3",
                                            "irf_pct", "srf_pct", "dril_pct")
                                    and delta_abs < 0) else \
                    "worsened" if delta_abs > 0 and key in ("dril_pct",) else \
                    "improved" if delta_abs > 0 and key == "ez_integrity" else \
                    "unchanged" if abs(delta_abs) < 1e-6 else "changed"
        deltas[key] = {
            "t1":          v1,
            "t2":          v2,
            "delta_abs":   delta_abs,
            "delta_pct":   delta_pct,
            "direction":   direction,
        }
    return deltas


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":       "#0E1A20",
    "text":     "#F2F4F3",
    "muted":    "#6B8080",
    "mint":     "#00C9A7",
    "red":      "#E05C5C",
    "amber":    "#D4954A",
    "green":    "#3DBD8A",
    "rule":     "#1C3040",
}


def _seg_to_rgb(scan: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Overlay segmentation labels as transparent color on grayscale scan."""
    h, w = scan.shape
    rgb = np.stack([scan, scan, scan], axis=2)  # (H, W, 3) grayscale

    alpha = 0.55
    for lbl, color in LABEL_COLORS.items():
        if lbl == 0:
            continue
        mask = label == lbl
        for c, val in enumerate(color):
            rgb[:, :, c] = np.where(mask, rgb[:, :, c] * (1 - alpha) + val * alpha, rgb[:, :, c])

    return np.clip(rgb, 0, 1)


def _diff_to_rgb(scan: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """
    Color-coded diff overlay on the T2 scan:
      Green  = resolved fluid (-1)
      Red    = new/worsening fluid (+1)
      Gray   = unchanged (0)
    """
    rgb = np.stack([scan, scan, scan], axis=2)
    alpha = 0.7

    # Resolved (green)
    mask_res = diff == -1
    rgb[mask_res, 0] = rgb[mask_res, 0] * (1 - alpha) + 0.24 * alpha
    rgb[mask_res, 1] = rgb[mask_res, 1] * (1 - alpha) + 0.73 * alpha
    rgb[mask_res, 2] = rgb[mask_res, 2] * (1 - alpha) + 0.55 * alpha

    # New fluid (red)
    mask_new = diff == 1
    rgb[mask_new, 0] = rgb[mask_new, 0] * (1 - alpha) + 0.94 * alpha
    rgb[mask_new, 1] = rgb[mask_new, 1] * (1 - alpha) + 0.36 * alpha
    rgb[mask_new, 2] = rgb[mask_new, 2] * (1 - alpha) + 0.27 * alpha

    return np.clip(rgb, 0, 1)


def _make_comparison_figure(r: DiffResult) -> plt.Figure:
    """Three-panel comparison: T1 seg | T2 seg | Diff overlay."""
    date_t1 = r.visit_dates[0] if r.visit_dates else "Visit 1"
    date_t2 = r.visit_dates[1] if len(r.visit_dates) > 1 else "Visit 2"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor(PALETTE["bg"])

    panels = [
        (r.overlay_t1,   f"T1  ·  {date_t1}", f"CRT {r.biomarkers_t1['crt_um']:.0f}µm"),
        (r.overlay_t2,   f"T2  ·  {date_t2}", f"CRT {r.biomarkers_t2['crt_um']:.0f}µm"),
        (r.overlay_diff, "Diff overlay",        "green = resolved   red = new"),
    ]

    for ax, (img, title, sub) in zip(axes, panels):
        ax.imshow(img, cmap=None, aspect="auto")
        ax.set_title(title, color=PALETTE["text"], fontsize=11,
                     fontweight="bold", pad=6)
        ax.set_xlabel(sub, color=PALETTE["muted"], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["rule"])

    # Legend on diff panel
    legend_items = [
        mpatches.Patch(color="#3DBD8A", label="Resolved fluid"),
        mpatches.Patch(color="#E05C5C", label="New / worsening"),
    ]
    axes[2].legend(handles=legend_items, loc="lower right",
                   facecolor=PALETTE["bg"], edgecolor=PALETTE["rule"],
                   labelcolor=PALETTE["text"], fontsize=8)

    # Delta summary bar below panels
    deltas = r.biomarker_deltas
    summary_parts = []
    for key, label in [("crt_um", "CRT"), ("irf_mm3", "IRF"), ("srf_mm3", "SRF"),
                        ("dril_pct", "DRIL"), ("ez_integrity", "EZ")]:
        d = deltas[key]
        sign = "↓" if d["delta_abs"] < 0 else "↑" if d["delta_abs"] > 0 else "="
        summary_parts.append(f"{label} {sign}{abs(d['delta_pct']):.0f}%")

    fig.text(0.5, 0.01, "   ·   ".join(summary_parts),
             ha="center", color=PALETTE["mint"], fontsize=9.5,
             fontfamily="monospace")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def _make_trajectory_figure(r: DiffResult, history: Optional[list] = None) -> plt.Figure:
    """
    Trajectory chart for CRT, IRF, DRIL, EZ over time.
    If history is provided (list of biomarker dicts), plots all timepoints.
    Otherwise plots just T1 and T2 with a dashed forecast.
    """
    dates = r.visit_dates if r.visit_dates else ["T1", "T2"]

    # Build time series
    if history:
        all_bio  = history
        all_dates = dates[:len(history)]
    else:
        all_bio  = [r.biomarkers_t1, r.biomarkers_t2]
        all_dates = dates[:2]

    metrics = [
        ("crt_um",       "CRT (µm)",       PALETTE["mint"]),
        ("irf_mm3",      "IRF (mm³)",      PALETTE["red"]),
        ("dril_pct",     "DRIL (%)",        PALETTE["amber"]),
        ("ez_integrity", "EZ integrity",    PALETTE["green"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.patch.set_facecolor(PALETTE["bg"])
    axes = axes.flatten()

    for ax, (key, label, color) in zip(axes, metrics):
        ax.set_facecolor("#111C24")
        vals = [b[key] for b in all_bio]
        xs   = list(range(len(all_dates)))

        ax.plot(xs, vals, "o-", color=color, linewidth=2.5,
                markersize=7, markerfacecolor=color, zorder=3)

        # Simple linear forecast for next visit (dashed)
        if len(vals) >= 2:
            slope = vals[-1] - vals[-2]
            forecast_x = len(xs)
            forecast_y = vals[-1] + slope
            ax.plot([xs[-1], forecast_x], [vals[-1], forecast_y],
                    "--", color=color, linewidth=1.5, alpha=0.5)
            ax.scatter([forecast_x], [forecast_y], color=color,
                       alpha=0.5, s=40, zorder=2, marker="D")

        ax.set_title(label, color=PALETTE["text"], fontsize=11, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(all_dates, color=PALETTE["muted"], fontsize=8, rotation=15)
        ax.tick_params(axis="y", colors=PALETTE["muted"])
        ax.spines["bottom"].set_color(PALETTE["rule"])
        ax.spines["left"].set_color(PALETTE["rule"])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=PALETTE["rule"], linewidth=0.5, alpha=0.6)

        # Annotate last value
        ax.annotate(f"{vals[-1]:.2f}",
                    xy=(xs[-1], vals[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    color=color, fontsize=8, va="center")

    fig.text(0.5, 0.97, "OcuTrace — Biomarker Trajectory",
             ha="center", color=PALETTE["text"], fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "Dashed = forecast · Diamond = projected next visit",
             ha="center", color=PALETTE["muted"], fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class OcuTraceDiffEngine:
    """
    End-to-end OcuTrace pipeline.

    Example:
        engine = OcuTraceDiffEngine()
        result = engine.run("scan_jan.png", "scan_mar.png",
                            visit_dates=["2024-01-14", "2024-03-02"])
        result.save_overlay("outputs/diff.png")
        result.save_trajectory("outputs/traj.png")
        print(result.to_json())
    """

    def __init__(self, weights_path: Optional[str] = None):
        self.model = UNet(in_channels=1, out_channels=4).to(DEVICE)
        if weights_path and Path(weights_path).exists():
            self.model.load_retouch_weights(weights_path)
        else:
            _safe_print("[OcuTrace] No pretrained weights - using random init.")
            _safe_print("           Download RETOUCH weights and pass weights_path= for real segmentation.")
        self.model.eval()

    def run(
        self,
        scan_t1_path: str | Path,
        scan_t2_path: str | Path,
        visit_dates: Optional[list[str]] = None,
        px_area_mm2: float = 1e-4,
        skip_registration: bool = False,
    ) -> DiffResult:
        """
        Full pipeline: load → register → segment → diff → biomarkers → visualize.

        Args:
            scan_t1_path:        Path to baseline scan (PNG/JPG/DICOM)
            scan_t2_path:        Path to follow-up scan
            visit_dates:         ["YYYY-MM-DD", "YYYY-MM-DD"] for display
            px_area_mm2:         Pixel area for volume calculations
            skip_registration:   Set True if scans are already aligned

        Returns:
            DiffResult with all outputs populated.
        """
        _safe_print("[OcuTrace] Loading scans...")
        t1 = load_oct_scan(scan_t1_path)
        t2 = load_oct_scan(scan_t2_path)

        # Resize T2 to match T1 shape if needed
        if t1.shape != t2.shape:
            t2_img = Image.fromarray((t2 * 255).astype(np.uint8))
            t2_img = t2_img.resize((t1.shape[1], t1.shape[0]), Image.BILINEAR)
            t2 = np.array(t2_img, dtype=np.float32) / 255.0

        if not skip_registration:
            _safe_print("[OcuTrace] Registering T2 -> T1 (affine, mutual information)...")
            t2_reg = register_scans(t1, t2)
        else:
            t2_reg = t2

        _safe_print("[OcuTrace] Segmenting fluid (T1)...")
        label_t1 = segment_fluid(self.model, t1, original_shape=t1.shape)

        _safe_print("[OcuTrace] Segmenting fluid (T2)...")
        label_t2 = segment_fluid(self.model, t2_reg, original_shape=t1.shape)

        _safe_print("[OcuTrace] Computing diff map...")
        diff_map = compute_diff(label_t1, label_t2)

        _safe_print("[OcuTrace] Extracting biomarkers...")
        bio_t1 = extract_biomarkers(label_t1, px_area_mm2)
        bio_t2 = extract_biomarkers(label_t2, px_area_mm2)
        deltas  = compute_biomarker_deltas(bio_t1, bio_t2)

        _safe_print("[OcuTrace] Generating overlays...")
        ov_t1   = _seg_to_rgb(t1,     label_t1)
        ov_t2   = _seg_to_rgb(t2_reg, label_t2)
        ov_diff = _diff_to_rgb(t2_reg, diff_map)

        return DiffResult(
            scan_t1=t1, scan_t2=t2_reg,
            label_t1=label_t1, label_t2=label_t2,
            diff_map=diff_map,
            overlay_t1=ov_t1, overlay_t2=ov_t2, overlay_diff=ov_diff,
            biomarkers_t1=bio_t1, biomarkers_t2=bio_t2,
            biomarker_deltas=deltas,
            visit_dates=visit_dates or [],
        )

    def run_series(
        self,
        scan_paths: list[str | Path],
        visit_dates: Optional[list[str]] = None,
        px_area_mm2: float = 1e-4,
    ) -> list[DiffResult]:
        """
        Run pairwise comparisons across N timepoints.
        Returns list of N-1 DiffResults: (T1→T2), (T2→T3), ...
        """
        results = []
        for i in range(len(scan_paths) - 1):
            dates = None
            if visit_dates and len(visit_dates) > i + 1:
                dates = [visit_dates[i], visit_dates[i + 1]]
            result = self.run(scan_paths[i], scan_paths[i + 1],
                              visit_dates=dates, px_area_mm2=px_area_mm2)
            results.append(result)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 8. SYNTHETIC TEST DATA GENERATOR
#    Generates realistic OCT-like B-scan pairs for testing without real data.
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_pair(
    height: int = 512,
    width:  int = 512,
    seed:   int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic OCT B-scan pair (T1, T2) for testing.

    T1: baseline with macular edema (bright fluid pockets)
    T2: follow-up with partial resolution (some fluid reduced, new DRIL)

    Returns: (scan_t1, scan_t2) both float32 (H, W) in [0, 1]
    """
    rng = np.random.default_rng(seed)

    def make_base_scan(h, w, rng):
        """Layered OCT structure: background → retinal layers → choroid."""
        scan = np.zeros((h, w), dtype=np.float32)

        # Dark vitreous (top ~15%)
        vit_end = int(h * 0.15)
        scan[:vit_end] = rng.uniform(0.0, 0.04, (vit_end, w))

        # Inner retina layers (~15-55%) — moderately bright
        inner_start, inner_end = vit_end, int(h * 0.55)
        for row in range(inner_start, inner_end):
            base  = 0.3 + 0.2 * np.sin(np.linspace(0, np.pi, w))
            noise = rng.normal(0, 0.04, w)
            scan[row] = np.clip(base + noise, 0, 1)

        # Ellipsoid zone / RPE — bright band (~55-65%)
        ez_start, ez_end = int(h * 0.55), int(h * 0.65)
        for row in range(ez_start, ez_end):
            intensity = 0.85 - 0.03 * abs(row - (ez_start + ez_end) // 2)
            scan[row]  = np.clip(rng.normal(intensity, 0.03, w), 0, 1)

        # Choroid (~65-90%) — medium bright
        cho_start, cho_end = int(h * 0.65), int(h * 0.90)
        for row in range(cho_start, cho_end):
            scan[row] = np.clip(rng.uniform(0.25, 0.50, w), 0, 1)

        # Sclera (>90%) — dark
        scan[cho_end:] = rng.uniform(0.0, 0.08, (h - cho_end, w))
        return scan

    def add_fluid(scan, h, w, rng, severity=1.0):
        """Add fluid-like dark pockets in the macular area."""
        cx = w // 2
        result = scan.copy()
        n_pockets = int(rng.integers(2, 5) * severity)
        for _ in range(n_pockets):
            # Fluid appears as hyporeflective (dark) regions in retinal layers
            row_c = int(rng.uniform(0.2, 0.5) * h)
            col_c = int(rng.uniform(0.35, 0.65) * w)
            rr    = int(rng.uniform(8, 20) * severity)
            rc    = int(rng.uniform(12, 35) * severity)
            for r in range(max(0, row_c - rr), min(h, row_c + rr)):
                for c in range(max(0, col_c - rc), min(w, col_c + rc)):
                    if ((r - row_c) / rr) ** 2 + ((c - col_c) / rc) ** 2 <= 1:
                        result[r, c] = np.clip(result[r, c] * 0.15, 0, 0.1)
        return result

    # T1: moderate edema
    t1_base = make_base_scan(height, width, rng)
    t1 = add_fluid(t1_base, height, width, rng, severity=1.0)

    # T2: partial improvement — some fluid resolved, slight structural change
    t2_base = make_base_scan(height, width, rng)
    t2 = add_fluid(t2_base, height, width, rng, severity=0.45)

    # Add slight misalignment to T2 (simulates real inter-visit shift)
    shift_r = rng.integers(-8, 8)
    shift_c = rng.integers(-5, 5)
    t2 = np.roll(t2, shift_r, axis=0)
    t2 = np.roll(t2, shift_c, axis=1)

    return t1.astype(np.float32), t2.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 9. QUICK DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    out_dir = Path("ocutrace_output")
    out_dir.mkdir(exist_ok=True)

    # ── Option A: use real scans ─────────────────────────────────────────────
    if len(sys.argv) == 3:
        scan_t1_path = sys.argv[1]
        scan_t2_path = sys.argv[2]
        visit_dates  = ["Visit 1", "Visit 2"]
        print(f"\n[OcuTrace] Running on real scans: {scan_t1_path}, {scan_t2_path}")

    # ── Option B: generate synthetic test data ───────────────────────────────
    else:
        print("\n[OcuTrace] No scan paths provided — generating synthetic pair for demo.")
        print("           Usage: python diff_engine.py scan_t1.png scan_t2.png\n")

        t1, t2 = generate_synthetic_pair(512, 512, seed=42)

        # Save synthetic scans so the engine can load them normally
        Image.fromarray((t1 * 255).astype(np.uint8)).save(out_dir / "synthetic_t1.png")
        Image.fromarray((t2 * 255).astype(np.uint8)).save(out_dir / "synthetic_t2.png")

        scan_t1_path = out_dir / "synthetic_t1.png"
        scan_t2_path = out_dir / "synthetic_t2.png"
        visit_dates  = ["2024-01-14", "2024-03-02"]

    # ── Run pipeline ─────────────────────────────────────────────────────────
    engine = OcuTraceDiffEngine(weights_path=None)  # swap in RETOUCH weights here

    result = engine.run(
        scan_t1_path,
        scan_t2_path,
        visit_dates=visit_dates,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    result.save_overlay(out_dir / "diff_overlay.png")
    result.save_trajectory(out_dir / "trajectory.png")

    json_path = out_dir / "biomarkers.json"
    json_path.write_text(result.to_json())

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("OcuTrace — Biomarker Summary")
    print("─" * 60)

    header = f"{'Metric':<18} {'T1':>10} {'T2':>10} {'Delta':>10}  {'Direction'}"
    print(header)
    print("─" * 60)

    for key, d in result.biomarker_deltas.items():
        sign = "↓" if d["delta_abs"] < 0 else "↑" if d["delta_abs"] > 0 else "="
        print(f"{key:<18} {d['t1']:>10.3f} {d['t2']:>10.3f} "
              f"{sign}{abs(d['delta_pct']):>8.1f}%  {d['direction']}")

    print("─" * 60)
    print(f"\nOutputs saved to: {out_dir.resolve()}/")
    print(f"  diff_overlay.png  — 3-panel comparison figure")
    print(f"  trajectory.png    — biomarker trajectory chart")
    print(f"  biomarkers.json   — structured data for LLM narrator")
    print(f"\nPass biomarkers.json to the LLM narrator module next.")
