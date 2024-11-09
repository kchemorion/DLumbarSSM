# src/training/metrics.py

import torch
import numpy as np
from typing import Tuple

def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor,
                        smooth: float = 1e-6) -> torch.Tensor:
    """Calculate Dice score"""
    intersection = (pred & target).float().sum((1, 2))
    union = pred.float().sum((1, 2)) + target.float().sum((1, 2))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()

def calculate_iou(pred: torch.Tensor, target: torch.Tensor,
                 smooth: float = 1e-6) -> torch.Tensor:
    """Calculate IoU score"""
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def calculate_precision_recall(pred: torch.Tensor, target: torch.Tensor,
                             smooth: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate precision and recall"""
    true_positives = (pred & target).float().sum((1, 2))
    pred_positives = pred.float().sum((1, 2))
    actual_positives = target.float().sum((1, 2))
    
    precision = (true_positives + smooth) / (pred_positives + smooth)
    recall = (true_positives + smooth) / (actual_positives + smooth)
    
    return precision.mean(), recall.mean()
