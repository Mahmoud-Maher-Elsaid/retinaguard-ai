import torch


@torch.no_grad()
def segmentation_metrics_from_logits(logits, targets, threshold: float = 0.5, smooth: float = 1.0):
    """
    Compute Dice and IoU for multi-label segmentation.

    logits:
        [B, C, H, W]

    targets:
        [B, C, H, W]
    """

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    dims = (0, 2, 3)

    intersection = torch.sum(preds * targets, dim=dims)

    pred_sum = torch.sum(preds, dim=dims)
    target_sum = torch.sum(targets, dim=dims)

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)

    return {
        "mean_dice": float(dice.mean().detach().cpu()),
        "mean_iou": float(iou.mean().detach().cpu()),
        "dice_microaneurysms": float(dice[0].detach().cpu()),
        "dice_haemorrhages": float(dice[1].detach().cpu()),
        "dice_hard_exudates": float(dice[2].detach().cpu()),
        "dice_soft_exudates": float(dice[3].detach().cpu()),
        "iou_microaneurysms": float(iou[0].detach().cpu()),
        "iou_haemorrhages": float(iou[1].detach().cpu()),
        "iou_hard_exudates": float(iou[2].detach().cpu()),
        "iou_soft_exudates": float(iou[3].detach().cpu()),
    }
