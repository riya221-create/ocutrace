"""
OcuTrace Training — Evaluation Script
========================================
Runs a trained checkpoint on a held-out test set and reports
Dice/F1 per class — in the exact same format as Table 2 of the paper,
so the output can be dropped directly into the manuscript.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --test_dir ./data/retouch_test
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from diff_engine import UNet, DEVICE

from dataset import OCTSegmentationDataset
from losses import compute_dice_per_class, compute_f1_per_class

LABEL_NAMES = {0: "background", 1: "IRF", 2: "SRF", 3: "PED"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate OcuTrace U-Net on held-out test set")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_dir",   type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--out_json",   type=str, default="evaluation_results.json")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, num_classes=4):
    model.eval()

    dice_sums   = {c: 0.0 for c in range(num_classes)}
    dice_counts = {c: 0 for c in range(num_classes)}
    f1_sums     = {c: 0.0 for c in range(num_classes)}
    f1_counts   = {c: 0 for c in range(num_classes)}

    n_examples = 0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)
        logits = model(images)

        dice_scores = compute_dice_per_class(logits, masks, num_classes=num_classes)
        f1_scores   = compute_f1_per_class(logits, masks, num_classes=num_classes)

        for c in range(num_classes):
            if dice_scores[c] is not None:
                dice_sums[c] += dice_scores[c]
                dice_counts[c] += 1
            if f1_scores[c] is not None:
                f1_sums[c] += f1_scores[c]
                f1_counts[c] += 1

        n_examples += images.shape[0]

    mean_dice = {
        c: round(dice_sums[c] / dice_counts[c], 4) if dice_counts[c] > 0 else None
        for c in range(num_classes)
    }
    mean_f1 = {
        c: round(f1_sums[c] / f1_counts[c], 4) if f1_counts[c] > 0 else None
        for c in range(num_classes)
    }

    fluid_dices = [mean_dice[c] for c in (1, 2, 3) if mean_dice[c] is not None]
    overall_mean_dice = round(sum(fluid_dices) / len(fluid_dices), 4) if fluid_dices else None

    return {
        "n_test_examples": n_examples,
        "dice_per_class": {LABEL_NAMES[c]: mean_dice[c] for c in range(num_classes)},
        "f1_per_class":   {LABEL_NAMES[c]: mean_f1[c] for c in range(num_classes)},
        "overall_mean_dice_fluid_classes": overall_mean_dice,
    }


def main():
    args = parse_args()

    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    model = UNet(in_channels=1, out_channels=4).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Eval] Checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"training-time best_dice={ckpt.get('best_dice', '?')}")

    print(f"[Eval] Loading test set: {args.test_dir}")
    test_ds = OCTSegmentationDataset(
        args.test_dir, image_size=(args.image_size, args.image_size), augment=False,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[Eval] Test examples: {len(test_ds)}")

    print(f"\n[Eval] Running evaluation...\n")
    results = evaluate(model, test_loader, DEVICE)

    print("─" * 56)
    print("OcuTrace — Held-Out Test Set Results")
    print("─" * 56)
    print(f"Test examples: {results['n_test_examples']}\n")
    print(f"{'Class':<14} {'Dice':>10} {'F1':>10}")
    for label in ("IRF", "SRF", "PED"):
        d = results["dice_per_class"][label]
        f = results["f1_per_class"][label]
        d_str = f"{d:.4f}" if d is not None else "N/A"
        f_str = f"{f:.4f}" if f is not None else "N/A"
        print(f"{label:<14} {d_str:>10} {f_str:>10}")
    print("─" * 56)
    print(f"Overall mean Dice (fluid classes): {results['overall_mean_dice_fluid_classes']}")
    print("─" * 56)

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"\n[Eval] Results saved → {args.out_json}")
    print(f"[Eval] Copy these numbers directly into Table 2's OcuTrace row.")


if __name__ == "__main__":
    main()
