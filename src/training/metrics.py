"""
Training components for the Multi-view Spine Analysis model.
Replaces /src/training/metrics.py and updates /src/training/trainer.py
"""

# /src/training/metrics.py
import torch
import torch.nn.functional as F
from typing import Dict, Tuple

class MultiTaskMetrics:
    """Handles metric computation for all tasks"""
    
    @staticmethod
    def compute_segmentation_metrics(pred: torch.Tensor, 
                                   target: torch.Tensor, 
                                   smooth: float = 1e-6) -> Dict[str, torch.Tensor]:
        """Compute segmentation metrics"""
        pred = pred.float()
        target = target.float()
        
        # Dice score
        intersection = (pred * target).sum((1, 2))
        union = pred.sum((1, 2)) + target.sum((1, 2))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        # IoU score
        iou = intersection / (union - intersection + smooth)
        
        # Precision and recall
        true_positives = (pred * target).sum((1, 2))
        pred_positives = pred.sum((1, 2))
        actual_positives = target.sum((1, 2))
        
        precision = (true_positives + smooth) / (pred_positives + smooth)
        recall = (true_positives + smooth) / (actual_positives + smooth)
        
        return {
            'dice': dice.mean(),
            'iou': iou.mean(),
            'precision': precision.mean(),
            'recall': recall.mean()
        }
    
    @staticmethod
    def compute_landmark_metrics(pred: torch.Tensor, 
                               target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute landmark detection metrics"""
        # Mean distance error
        distance_error = torch.norm(pred - target, dim=-1)
        
        # PCK (Percentage of Correct Keypoints)
        threshold = 0.05  # 5% of image size
        correct_keypoints = (distance_error < threshold).float()
        
        return {
            'mean_distance': distance_error.mean(),
            'pck': correct_keypoints.mean()
        }
    
    @staticmethod
    def compute_classification_metrics(pred: torch.Tensor, 
                                     target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute classification metrics"""
        pred_labels = pred.argmax(dim=1)
        accuracy = (pred_labels == target).float().mean()
        
        return {
            'accuracy': accuracy,
            'cross_entropy': F.cross_entropy(pred, target)
        }

class MultiTaskLoss(torch.nn.Module):
    """Multi-task loss with learnable weights"""
    
    def __init__(self, num_tasks: int = 4):
        super().__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted multi-task loss
        
        Args:
            outputs: Dictionary of model outputs
            targets: Dictionary of target values
            
        Returns:
            Total loss and individual task losses
        """
        # Individual task losses
        segmentation_loss = F.binary_cross_entropy_with_logits(
            outputs['segmentation'], 
            targets['segmentation']
        )
        
        landmark_loss = F.mse_loss(
            outputs['landmarks'],
            targets['landmarks']
        )
        
        level_loss = F.cross_entropy(
            outputs['vertebral_levels'],
            targets['vertebral_levels']
        )
        
        condition_loss = F.cross_entropy(
            outputs['conditions'],
            targets['conditions']
        )
        
        # Get task weights
        weights = torch.exp(-self.log_vars)
        
        # Compute weighted losses
        losses = {
            'segmentation': weights[0] * segmentation_loss + self.log_vars[0],
            'landmarks': weights[1] * landmark_loss + self.log_vars[1],
            'levels': weights[2] * level_loss + self.log_vars[2],
            'conditions': weights[3] * condition_loss + self.log_vars[3]
        }
        
        total_loss = sum(losses.values())
        
        return total_loss, losses

