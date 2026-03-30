
import torch
import torch.nn as nn
import torch.nn.functional as F


def SoftIoULoss(pred, target, smooth=1e-5, reduction="mean"):
    """Soft-IoU loss - a differentiable IoU-based loss.
    
    Args:
        pred: model prediction (B, C, H, W), after sigmoid
        target: binary mask (B, C, H, W)
        smooth: smoothing term to avoid division by zero
        reduction: 'mean' or 'sum'
    
    Note:
        IoU = intersection / union
        Difference from Dice: Dice denominator is sum(pred)+sum(target),
        while IoU denominator is sum(pred)+sum(target)-intersection.
    """
    assert pred.shape == target.shape
    
    # Compute IoU per sample
    batch_size = pred.shape[0]
    pred_flat = pred.view(batch_size, -1)
    target_flat = target.view(batch_size, -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    iou_loss = 1.0 - iou
    
    if reduction == "mean":
        return iou_loss.mean()
    elif reduction == "sum":
        return iou_loss.sum()
    else:
        return iou_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


