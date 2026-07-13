import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Multi-label Dice loss for lesion segmentation.

    logits shape:
        [B, C, H, W]

    targets shape:
        [B, C, H, W]
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        dims = (0, 2, 3)

        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs + targets, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice.mean()

        return loss


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogits + Dice loss.
    Good first baseline for small lesion segmentation.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        dice_loss = self.dice(logits, targets.float())

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
