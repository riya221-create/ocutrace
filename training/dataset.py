"""
OcuTrace Training — Dataset Loader
=====================================
Expects data organized as:

    data_dir/
        images/
            scan_0001.png
            scan_0002.png
            ...
        masks/
            scan_0001.png   (same filename as corresponding image)
            scan_0002.png
            ...

Masks must be single-channel PNGs with integer pixel values:
    0 = background
    1 = IRF (intraretinal fluid)
    2 = SRF (subretinal fluid)
    3 = PED (pigment epithelial detachment)

See prepare_data.py for converting RETOUCH/DUKE native formats into
this structure.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OCTSegmentationDataset(Dataset):
    """
    Loads paired (image, mask) examples for OCT fluid segmentation training.

    Applies light data augmentation (horizontal flip, small rotation,
    brightness jitter) appropriate for OCT B-scans — vertical flips are
    deliberately excluded since retinal layer order (vitreous → choroid)
    is anatomically fixed and must be preserved.
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_size: tuple[int, int] = (512, 512),
        augment: bool = True,
        file_list: list[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"
        self.image_size = image_size
        self.augment = augment

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Expected {self.image_dir} and {self.mask_dir} to exist. "
                f"See prepare_data.py for how to build this structure "
                f"from RETOUCH or DUKE source data."
            )

        if file_list is not None:
            self.filenames = file_list
        else:
            self.filenames = sorted(
                f.name for f in self.image_dir.glob("*.png")
            )

        if len(self.filenames) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        img = Image.open(self.image_dir / fname).convert("L")
        mask = Image.open(self.mask_dir / fname).convert("L")

        img = img.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)  # NEAREST — never interpolate label values

        img_arr = np.array(img, dtype=np.float32) / 255.0
        mask_arr = np.array(mask, dtype=np.int64)

        # Clamp mask values defensively — guards against stray pixel values
        # from lossy mask encoding (e.g. JPEG artifacts if a user accidentally
        # saved masks as JPEG instead of PNG)
        mask_arr = np.clip(mask_arr, 0, 3)

        if self.augment:
            img_arr, mask_arr = self._augment(img_arr, mask_arr)

        img_tensor = torch.from_numpy(img_arr).unsqueeze(0)   # (1, H, W)
        mask_tensor = torch.from_numpy(mask_arr).long()        # (H, W)

        return img_tensor, mask_tensor

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Horizontal flip (left/right eye symmetry — anatomically valid)
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        # Small rotation (±5 degrees) — simulates minor head tilt during acquisition
        if random.random() < 0.3:
            angle = random.uniform(-5, 5)
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            img_pil = img_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST, fillcolor=0)
            img = np.array(img_pil, dtype=np.float32) / 255.0
            mask = np.array(mask_pil, dtype=np.int64)

        # Brightness jitter — OCT scans vary in brightness across devices/sessions
        if random.random() < 0.4:
            factor = random.uniform(0.85, 1.15)
            img = np.clip(img * factor, 0, 1)

        return img, mask


def make_train_val_split(
    data_dir: str | Path,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split available files into train/val sets.
    Uses a fixed seed for reproducibility — important when reporting
    results, since the train/val split itself affects reported metrics.
    """
    data_dir = Path(data_dir)
    image_dir = data_dir / "images"
    filenames = sorted(f.name for f in image_dir.glob("*.png"))

    rng = random.Random(seed)
    rng.shuffle(filenames)

    n_val = max(1, int(len(filenames) * val_fraction))
    val_files = filenames[:n_val]
    train_files = filenames[n_val:]

    return train_files, val_files
