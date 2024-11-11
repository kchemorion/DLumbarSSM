"""
Training components for the Multi-view Spine Analysis model.
Replaces /src/training/metrics.py and updates /src/training/trainer.py
"""

# /src/training/metrics.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)  


class MultiTaskMetrics:
    """Handles metric computation for all tasks"""

    @staticmethod
    def compute_segmentation_metrics(preds: Dict[str, torch.Tensor],
                                       targets: Dict[str, torch.Tensor],
                                       smooth: float = 1e-6) -> Dict[str, torch.Tensor]:
        """Compute segmentation metrics for multiple views."""
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []

        for view in preds.keys():
            pred = preds[view].float()
            target = targets[view].float()

            intersection = (pred * target).sum((1, 2))
            union = pred.sum((1, 2)) + target.sum((1, 2))

            dice = (2.0 * intersection + smooth) / (union + smooth)
            iou = intersection / (union - intersection + smooth)

            true_positives = (pred * target).sum((1, 2))
            pred_positives = pred.sum((1, 2))
            actual_positives = target.sum((1, 2))

            precision = (true_positives + smooth) / (pred_positives + smooth)
            recall = (true_positives + smooth) / (actual_positives + smooth)
            
            dice_scores.append(dice.mean())
            iou_scores.append(iou.mean())
            precision_scores.append(precision.mean())
            recall_scores.append(recall.mean())

        return {
            'dice': torch.mean(torch.stack(dice_scores)),
            'iou': torch.mean(torch.stack(iou_scores)),
            'precision': torch.mean(torch.stack(precision_scores)),
            'recall': torch.mean(torch.stack(recall_scores))
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

class MultiTaskLoss(nn.Module):
    """Multi-task loss with learnable weights and proper dimension handling"""
    
    def __init__(self, num_tasks: int = 4, num_levels: int = 5):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.num_levels = num_levels
    
    def _verify_dimensions(self, outputs: Dict[str, torch.Tensor],
                          targets: Dict[str, torch.Tensor]) -> None:
        """Verify input dimensions match expected shapes"""
        # Check segmentation dimensions
        for view in outputs['segmentation']:
            out_shape = outputs['segmentation'][view].shape
            target_shape = targets['segmentation'][view].shape
            if out_shape[2:] != target_shape[2:]:  # Only check spatial dimensions
                raise ValueError(
                    f"Segmentation shape mismatch for view {view}: "
                    f"output {out_shape} vs target {target_shape}"
                )
        
        # Check landmark dimensions
        out_landmarks = outputs['landmarks']
        target_landmarks = targets['landmarks']
        if out_landmarks.shape[-1] != target_landmarks.shape[-1]:
            raise ValueError(
                f"Landmark dimension mismatch: "
                f"output {out_landmarks.shape} vs target {target_landmarks.shape}"
            )
    
    def _compute_segmentation_loss(self, outputs: Dict[str, torch.Tensor],
                                 targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute segmentation loss with class weighting"""
        seg_loss = 0.0
        num_views = len(outputs)
        
        for view in outputs:
            # Compute class weights to handle imbalance
            pos_weight = (targets[view] == 0).float().sum() / (targets[view] == 1).float().sum().clamp(min=1)
            
            loss = F.binary_cross_entropy_with_logits(
                outputs[view],
                targets[view],
                pos_weight=pos_weight
            )
            seg_loss += loss
            
        return seg_loss / num_views
    
    def _compute_landmark_loss(self, output: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
        """Compute landmark detection loss with proper reshaping"""
        # Ensure proper shapes for MSE loss
        if len(output.shape) == 4:  # [B, C, H, W]
            B, C = output.shape[:2]
            output = output.view(B, -1)  # Flatten spatial dimensions
            
        # Normalize coordinates if needed
        if output.abs().max() > 1:
            output = torch.sigmoid(output)
            
        return F.mse_loss(output, target)
    
    def _compute_classification_loss(self, output: torch.Tensor,
                                   target: torch.Tensor,
                                   num_classes: int) -> torch.Tensor:
        """Compute classification loss with label smoothing"""
        if len(output.shape) == 4:  # [B, C, H, W]
            output = output.mean(dim=[2, 3])  # Global average pooling
            
        return F.cross_entropy(
            output,
            target,
            label_smoothing=0.1
        )
    
    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        try:
            # 1. Segmentation Loss
            segmentation_loss = 0.0
            for name, pred in outputs['segmentation'].items():
                segmentation_loss += F.binary_cross_entropy_with_logits(
                    pred,
                    targets['segmentation'][name]
                )
            segmentation_loss = segmentation_loss / len(outputs['segmentation'])
            
            # 2. Landmark Loss
            landmark_loss = F.mse_loss(
                outputs['landmarks'],
                targets['landmarks']
            )
            
            # 3. Vertebral Level Loss - Fixed to handle one-hot encoded targets
            level_loss = F.cross_entropy(
                outputs['vertebral_levels'].reshape(-1, 3),
                targets['vertebral_levels'].reshape(-1, 3).argmax(dim=1)
            )
            
            # 4. Condition Loss
            condition_loss = F.cross_entropy(
                outputs['conditions'],
                targets['conditions'].argmax(dim=1) if len(targets['conditions'].shape) > 1 else targets['conditions']
            )
            
            # Apply learned weights
            weights = torch.exp(-self.log_vars)
            losses = {
                'segmentation': weights[0] * segmentation_loss + self.log_vars[0],
                'landmarks': weights[1] * landmark_loss + self.log_vars[1],
                'levels': weights[2] * level_loss + self.log_vars[2],
                'conditions': weights[3] * condition_loss + self.log_vars[3]
            }
            
            total_loss = sum(losses.values())
            return total_loss, losses
            
        except Exception as e:
            logger.error(f"Error computing multi-task loss: {str(e)}")
            logger.error("Vertebral Levels:")
            if 'outputs' in locals() and 'vertebral_levels' in outputs:
                logger.error(f"- Original pred shape: {outputs['vertebral_levels'].shape}")
            if 'targets' in locals() and 'vertebral_levels' in targets:
                logger.error(f"- Original target shape: {targets['vertebral_levels'].shape}")
                logger.error(f"- Target unique values: {targets['vertebral_levels'].unique()}")
                logger.error(f"- Target tensor device: {targets['vertebral_levels'].device}")
            if 'outputs' in locals() and 'vertebral_levels' in outputs:
                logger.error(f"- Pred tensor device: {outputs['vertebral_levels'].device}")
            
            logger.error("\nFull shapes:")
            for key, value in outputs.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        logger.error(f"- {key}.{k}: {v.shape}")
                else:
                    logger.error(f"- {key}: {value.shape}")
            
            logger.error("\nTarget shapes:")
            for key, value in targets.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        logger.error(f"- {key}.{k}: {v.shape}")
                else:
                    logger.error(f"- {key}: {value.shape}")
            raise

    def _verify_shapes(self, pred: torch.Tensor, target: torch.Tensor, name: str):
        """Helper to verify tensor shapes match expectations"""
        logger.info(f"Shape check for {name}:")
        logger.info(f"- Pred shape: {pred.shape}")
        logger.info(f"- Target shape: {target.shape}")
        logger.info(f"- Pred device: {pred.device}")
        logger.info(f"- Target device: {target.device}")
        if pred.shape[0] != target.shape[0]:
            logger.error(f"Batch size mismatch for {name}!")