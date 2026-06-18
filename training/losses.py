"""
OcuTrace Training — Loss Functions
====================================
Combined Dice + Cross-Entropy loss for multi-class OCT fluid segmentation.
This is the standard loss formulation used by the RETOUCH challenge
top-performing methods and subsequent attention U-Net variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multi-class soft Dice loss.

    Computed per-class then averaged, so rare classes (e.g. PED, which
    typically occupies far fewer pixels than background) contribute
    equally to the loss rather than being drowned out by class imbalance.
    """

    def __init__(self, num_classes: int = 4, smooth: float = 1e-6, ignore_background: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) raw model output
            targets: (B, H, W) integer class labels
        """
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        start_class = 1 if self.ignore_background else 0
        dice_per_class = []

        for c in range(start_class, self.num_classes):
            p = probs[:, c, :, :]
            t = targets_onehot[:, c, :, :]
            intersection = (p * t).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice.mean())

        mean_dice = torch.stack(dice_per_class).mean()
        return 1.0 - mean_dice


class CombinedLoss(nn.Module):
    """
    Weighted sum of Dice loss and Cross-Entropy loss.

    Cross-entropy provides stable, well-behaved gradients early in training
    (especially with random initialization), while Dice loss directly
    optimizes the metric we actually care about and care most strongly
    about for the minority fluid classes.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 0.6,
        ce_weight: float = 0.4,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        dice_loss = self.dice(logits, targets)
        ce_loss = self.ce(logits, targets)
        total = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        return {
            "total": total,
            "dice_loss": dice_loss.detach(),
            "ce_loss": ce_loss.detach(),
        }


@torch.no_grad()
def compute_dice_per_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    smooth: float = 1e-6,
) -> dict:
    """
    Compute Dice score per class for evaluation/reporting.
    Matches the metric format reported in Table 2 of the paper
    (RETOUCH challenge and FAM-U-Net benchmarks).

    Returns dict: {0: dice_bg, 1: dice_irf, 2: dice_srf, 3: dice_ped}
    """
    preds = torch.argmax(logits, dim=1)
    scores = {}

    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        # If class absent in both prediction and ground truth, Dice is
        # undefined; we report it as None rather than a misleading 1.0 or 0.0
        if union.item() == 0:
            scores[c] = None
        else:
            scores[c] = ((2 * intersection + smooth) / (union + smooth)).item()

    return scores


@torch.no_grad()
def compute_f1_per_class(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
) -> dict:
    """
    Compute pixel-level F1 score per class.
    Matches the F1 reporting format used by the multiscale attention
    U-Net benchmark in Table 2.
    """
    preds = torch.argmax(logits, dim=1)
    scores = {}

    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        tp = (pred_c & target_c).sum().item()
        fp = (pred_c & ~target_c).sum().item()
        fn = (~pred_c & target_c).sum().item()

        if tp + fp + fn == 0:
            scores[c] = None
            continue

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        scores[c] = f1

    return scores
