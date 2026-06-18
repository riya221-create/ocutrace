"""
OcuTrace Training — Train Script
===================================
Trains the U-Net fluid segmentation network on annotated OCT data.

Usage:
    python train.py --data_dir ./data/retouch_converted --epochs 50 --batch_size 8

    # Quick smoke test on synthetic data (no real dataset needed):
    python prepare_data.py --synthetic --out ./data/synthetic_test
    python train.py --data_dir ./data/synthetic_test --epochs 3 --batch_size 4

Produces:
    checkpoints/best_model.pth   — best validation Dice checkpoint
    checkpoints/last_model.pth   — most recent epoch checkpoint
    checkpoints/training_log.csv — per-epoch metrics for plotting
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))  # allow importing diff_engine
from diff_engine import UNet, DEVICE

from dataset import OCTSegmentationDataset, make_train_val_split
from losses import CombinedLoss, compute_dice_per_class


def parse_args():
    p = argparse.ArgumentParser(description="Train OcuTrace U-Net fluid segmentation model")
    p.add_argument("--data_dir",    type=str, required=True, help="Path to converted dataset (images/ + masks/)")
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--image_size",  type=int, default=512)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--resume",      type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_dice_loss = 0.0
    total_ce_loss = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        losses = criterion(logits, masks)
        losses["total"].backward()
        optimizer.step()

        total_loss      += losses["total"].item()
        total_dice_loss  += losses["dice_loss"].item()
        total_ce_loss    += losses["ce_loss"].item()
        n_batches += 1

    return {
        "loss":      total_loss / n_batches,
        "dice_loss": total_dice_loss / n_batches,
        "ce_loss":   total_ce_loss / n_batches,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=4):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    dice_sums   = {c: 0.0 for c in range(num_classes)}
    dice_counts = {c: 0 for c in range(num_classes)}

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        losses = criterion(logits, masks)
        total_loss += losses["total"].item()
        n_batches += 1

        dice_scores = compute_dice_per_class(logits, masks, num_classes=num_classes)
        for c, score in dice_scores.items():
            if score is not None:
                dice_sums[c] += score
                dice_counts[c] += 1

    mean_dice_per_class = {
        c: (dice_sums[c] / dice_counts[c] if dice_counts[c] > 0 else None)
        for c in range(num_classes)
    }
    valid_dices = [v for v in mean_dice_per_class.values() if v is not None]
    overall_mean_dice = sum(valid_dices) / len(valid_dices) if valid_dices else 0.0

    return {
        "loss": total_loss / n_batches,
        "dice_per_class": mean_dice_per_class,
        "mean_dice": overall_mean_dice,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Train] Device: {DEVICE}")
    print(f"[Train] Data dir: {args.data_dir}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_files, val_files = make_train_val_split(
        args.data_dir, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"[Train] Train examples: {len(train_files)}  |  Val examples: {len(val_files)}")

    train_ds = OCTSegmentationDataset(
        args.data_dir, image_size=(args.image_size, args.image_size),
        augment=True, file_list=train_files,
    )
    val_ds = OCTSegmentationDataset(
        args.data_dir, image_size=(args.image_size, args.image_size),
        augment=False, file_list=val_files,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(DEVICE.type == "cuda"))

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet(in_channels=1, out_channels=4).to(DEVICE)

    start_epoch = 0
    best_dice = 0.0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"[Train] Resumed from {args.resume} at epoch {start_epoch}, best_dice={best_dice:.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = CombinedLoss(num_classes=4, dice_weight=0.6, ce_weight=0.4)

    # ── Training loop ─────────────────────────────────────────────────────────
    log_path = checkpoint_dir / "training_log.csv"
    log_exists = log_path.exists()
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow([
            "epoch", "train_loss", "train_dice_loss", "train_ce_loss",
            "val_loss", "val_mean_dice", "val_dice_bg", "val_dice_irf",
            "val_dice_srf", "val_dice_ped", "lr", "epoch_time_sec",
        ])

    print(f"\n[Train] Starting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics   = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_metrics["mean_dice"])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        dpc = val_metrics["dice_per_class"]
        print(
            f"Epoch {epoch+1:>3} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_mean_dice={val_metrics['mean_dice']:.4f} | "
            f"IRF={dpc[1] if dpc[1] else 0:.3f} SRF={dpc[2] if dpc[2] else 0:.3f} "
            f"PED={dpc[3] if dpc[3] else 0:.3f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        log_writer.writerow([
            epoch + 1, train_metrics["loss"], train_metrics["dice_loss"], train_metrics["ce_loss"],
            val_metrics["loss"], val_metrics["mean_dice"],
            dpc[0], dpc[1], dpc[2], dpc[3],
            current_lr, round(epoch_time, 1),
        ])
        log_file.flush()

        # Save last checkpoint every epoch
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "val_metrics": val_metrics,
        }, checkpoint_dir / "last_model.pth")

        # Save best checkpoint
        if val_metrics["mean_dice"] > best_dice:
            best_dice = val_metrics["mean_dice"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_dice": best_dice,
                "val_metrics": val_metrics,
            }, checkpoint_dir / "best_model.pth")
            print(f"           ↳ New best model saved (val_mean_dice={best_dice:.4f})")

    log_file.close()
    print(f"\n[Train] Training complete. Best val_mean_dice: {best_dice:.4f}")
    print(f"[Train] Checkpoints saved to: {checkpoint_dir.resolve()}")
    print(f"[Train] Run evaluate.py on the held-out test set next to produce")
    print(f"        a real entry for Table 2 of the paper.")


if __name__ == "__main__":
    main()
