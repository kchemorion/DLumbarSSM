# src/training/metrics.py
import torch
import numpy as np
from typing import Tuple

def calculate_dice_score(pred: torch.Tensor, target: torch.Tensor,
                        smooth: float = 1e-6) -> torch.Tensor:
    """Calculate Dice score"""
    # Convert bool/float predictions to binary float
    pred = pred.float()
    target = target.float()
    
    # Calculate intersection and union using multiplication instead of bitwise operations
    intersection = (pred * target).sum((1, 2))
    union = pred.sum((1, 2)) + target.sum((1, 2))
    
    # Calculate dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()

def calculate_iou(pred: torch.Tensor, target: torch.Tensor,
                 smooth: float = 1e-6) -> torch.Tensor:
    """Calculate IoU score"""
    # Convert to float
    pred = pred.float()
    target = target.float()
    
    # Calculate intersection and union using multiplication
    intersection = (pred * target).sum((1, 2))
    total = pred.sum((1, 2)) + target.sum((1, 2))
    union = total - intersection  # This is more efficient than using OR
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def calculate_precision_recall(pred: torch.Tensor, target: torch.Tensor,
                             smooth: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate precision and recall"""
    # Convert to float
    pred = pred.float()
    target = target.float()
    
    # Calculate true positives and positives
    true_positives = (pred * target).sum((1, 2))
    pred_positives = pred.sum((1, 2))
    actual_positives = target.sum((1, 2))
    
    # Calculate precision and recall
    precision = (true_positives + smooth) / (pred_positives + smooth)
    recall = (true_positives + smooth) / (actual_positives + smooth)
    
    return precision.mean(), recall.mean()