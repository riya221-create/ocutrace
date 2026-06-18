"""
OcuTrace Training — Data Preparation
=======================================
Two functions:

1. generate_synthetic_dataset() — creates synthetic OCT-like images with
   KNOWN fluid masks, purely so train.py can be verified to actually
   learn something (loss decreases, Dice improves) before you invest
   time downloading and training on real RETOUCH/DUKE data.

2. convert_retouch() / convert_duke() — convert the native RETOUCH/DUKE
   distribution format into the simple images/+masks/ folder structure
   that dataset.py expects.

Usage:
    # Generate synthetic test data (instant, no download needed):
    python prepare_data.py --synthetic --out ./data/synthetic_test --n 200

    # Convert a downloaded RETOUCH directory:
    python prepare_data.py --convert retouch --src /path/to/RETOUCH --out ./data/retouch_converted

    # Convert a downloaded DUKE .mat directory:
    python prepare_data.py --convert duke --src /path/to/duke_mat_files --out ./data/duke_converted
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _make_synthetic_scan_and_mask(h, w, rng):
    """
    Generate one synthetic OCT-like B-scan with a KNOWN ground-truth fluid mask.
    Mask values: 0=background, 1=IRF, 2=SRF, 3=PED
    """
    scan = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.int64)

    vit_end = int(h * 0.15)
    scan[:vit_end] = rng.uniform(0.0, 0.04, (vit_end, w))

    inner_start, inner_end = vit_end, int(h * 0.55)
    for row in range(inner_start, inner_end):
        base  = 0.3 + 0.2 * np.sin(np.linspace(0, np.pi, w))
        scan[row] = np.clip(base + rng.normal(0, 0.04, w), 0, 1)

    ez_start, ez_end = int(h * 0.55), int(h * 0.65)
    for row in range(ez_start, ez_end):
        intensity = 0.85 - 0.03 * abs(row - (ez_start + ez_end) // 2)
        scan[row] = np.clip(rng.normal(intensity, 0.03, w), 0, 1)

    cho_start, cho_end = int(h * 0.65), int(h * 0.90)
    for row in range(cho_start, cho_end):
        scan[row] = np.clip(rng.uniform(0.25, 0.50, w), 0, 1)
    scan[cho_end:] = rng.uniform(0.0, 0.08, (h - cho_end, w))

    n_pockets = rng.integers(2, 6)
    for _ in range(n_pockets):
        fluid_class = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])

        if fluid_class == 1:
            row_c = int(rng.uniform(0.25, 0.50) * h)
            rr, rc = rng.uniform(6, 14), rng.uniform(8, 20)
        elif fluid_class == 2:
            row_c = int(rng.uniform(0.58, 0.68) * h)
            rr, rc = rng.uniform(10, 18), rng.uniform(20, 40)
        else:
            row_c = int(rng.uniform(0.62, 0.70) * h)
            rr, rc = rng.uniform(8, 14), rng.uniform(15, 30)

        col_c = int(rng.uniform(0.30, 0.70) * w)
        rr, rc = int(rr), int(rc)

        for r in range(max(0, row_c - rr), min(h, row_c + rr)):
            for c in range(max(0, col_c - rc), min(w, col_c + rc)):
                if ((r - row_c) / rr) ** 2 + ((c - col_c) / rc) ** 2 <= 1:
                    scan[r, c] = np.clip(scan[r, c] * 0.15, 0, 0.1)
                    mask[r, c] = fluid_class

    return scan, mask


def generate_synthetic_dataset(out_dir, n_examples=200, seed=42):
    """
    Generate a synthetic dataset with known ground truth, for verifying
    that train.py / dataset.py / losses.py work correctly end-to-end.
    NOT a substitute for real data.
    """
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    print(f"[PrepareData] Generating {n_examples} synthetic examples -> {out_dir}")
    for i in range(n_examples):
        scan, mask = _make_synthetic_scan_and_mask(512, 512, rng)
        fname = f"synth_{i:04d}.png"
        Image.fromarray((scan * 255).astype(np.uint8)).save(img_dir / fname)
        Image.fromarray(mask.astype(np.uint8)).save(mask_dir / fname)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n_examples} generated...")

    print(f"[PrepareData] Done. {n_examples} image/mask pairs written.")


def generate_synthetic_test_set(out_dir, n_examples=40, seed=999):
    generate_synthetic_dataset(out_dir, n_examples=n_examples, seed=seed)


def convert_retouch(src_dir, out_dir):
    """
    Convert RETOUCH challenge native format into images/+masks/ structure.
    Expects: src_dir/<Vendor>/<PatientID>/oct.mhd + reference.mhd
    Adjust glob patterns below if your downloaded structure differs.
    """
    import SimpleITK as sitk

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    oct_files = sorted(src_dir.glob("**/oct.mhd")) + sorted(src_dir.glob("**/*.mhd"))
    oct_files = [f for f in oct_files if "reference" not in f.name.lower()]

    if not oct_files:
        print(f"[PrepareData] No .mhd files found under {src_dir}.")
        print(f"  Check your RETOUCH folder structure and adjust the glob pattern.")
        return

    slice_count = 0
    for oct_path in oct_files:
        ref_path = oct_path.parent / "reference.mhd"
        if not ref_path.exists():
            print(f"  Skipping {oct_path} -- no matching reference.mhd found")
            continue

        oct_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(oct_path)))
        ref_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(ref_path)))
        patient_tag = oct_path.parent.name

        for z in range(oct_vol.shape[0]):
            scan_slice = oct_vol[z].astype(np.float32)
            scan_slice = (scan_slice - scan_slice.min()) / (scan_slice.max() - scan_slice.min() + 1e-8)
            mask_slice = ref_vol[z].astype(np.uint8)

            if mask_slice.max() == 0 and z % 2 == 0:
                continue

            fname = f"{patient_tag}_slice{z:03d}.png"
            Image.fromarray((scan_slice * 255).astype(np.uint8)).save(img_dir / fname)
            Image.fromarray(mask_slice).save(mask_dir / fname)
            slice_count += 1

    print(f"[PrepareData] RETOUCH conversion complete. {slice_count} slices written -> {out_dir}")


def convert_duke(src_dir, out_dir):
    """
    Convert Duke DME dataset (.mat files) into images/+masks/ structure.
    Field names can vary by release -- inspect data.keys() if this reports
    missing keys, and adjust possible_image_keys/possible_mask_keys.
    """
    from scipy.io import loadmat

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(src_dir.glob("*.mat"))
    if not mat_files:
        print(f"[PrepareData] No .mat files found in {src_dir}")
        return

    slice_count = 0
    for mat_path in mat_files:
        try:
            data = loadmat(str(mat_path))
        except Exception as e:
            print(f"  Could not load {mat_path}: {e}")
            continue

        possible_image_keys = ["images", "img", "data", "volume"]
        possible_mask_keys  = ["manualFluid1", "manualFluid", "fluid", "labels"]

        img_key = next((k for k in possible_image_keys if k in data), None)
        mask_key = next((k for k in possible_mask_keys if k in data), None)

        if img_key is None or mask_key is None:
            print(f"  Skipping {mat_path.name} -- expected keys not found. "
                  f"Available keys: {[k for k in data.keys() if not k.startswith('__')]}")
            continue

        images = data[img_key]
        masks  = data[mask_key]
        patient_tag = mat_path.stem
        n_slices = images.shape[2] if images.ndim == 3 else 1

        for z in range(n_slices):
            scan_slice = images[:, :, z].astype(np.float32) if images.ndim == 3 else images.astype(np.float32)
            scan_slice = (scan_slice - scan_slice.min()) / (scan_slice.max() - scan_slice.min() + 1e-8)

            mask_slice = masks[:, :, z] if masks.ndim == 3 else masks
            mask_slice = np.nan_to_num(mask_slice, nan=0).astype(np.uint8)
            mask_slice = (mask_slice > 0).astype(np.uint8)

            fname = f"{patient_tag}_slice{z:03d}.png"
            Image.fromarray((scan_slice * 255).astype(np.uint8)).save(img_dir / fname)
            Image.fromarray(mask_slice).save(mask_dir / fname)
            slice_count += 1

    print(f"[PrepareData] DUKE conversion complete. {slice_count} slices written -> {out_dir}")
    print(f"  Note: DUKE annotates fluid as a single class -- mapped to label 1 (IRF).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OcuTrace data preparation")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--convert", choices=["retouch", "duke"])
    parser.add_argument("--src", type=str)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--test_set", action="store_true")

    args = parser.parse_args()

    if args.synthetic:
        if args.test_set:
            generate_synthetic_test_set(args.out, n_examples=args.n)
        else:
            generate_synthetic_dataset(args.out, n_examples=args.n)
    elif args.convert == "retouch":
        if not args.src:
            raise ValueError("--src required for --convert retouch")
        convert_retouch(args.src, args.out)
    elif args.convert == "duke":
        if not args.src:
            raise ValueError("--src required for --convert duke")
        convert_duke(args.src, args.out)
    else:
        print("Specify --synthetic or --convert {retouch,duke}. See --help.")
