import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb

from src.segmentation.models import SpineSegmentationModel
from src.training.datasets import SpineSegmentationDataset
from src.training.metrics import (
    calculate_dice_score,
    calculate_iou,
    calculate_precision_recall
)

logger = logging.getLogger(__name__)

class SegmentationTrainer:
    """Complete training pipeline for spine segmentation"""
    
    def __init__(self,
                 data_root: Path,
                 model_save_dir: Path,
                 config: Optional[Dict] = None):
        """
        Initialize trainer
        
        Args:
            data_root: Root directory containing data
            model_save_dir: Directory to save models
            config: Optional configuration dictionary
        """
        self.data_root = Path(data_root)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            'batch_size': 8,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_workers': 4,
            'patience': 10,  # Early stopping patience
            'T_0': 10,  # Cosine annealing period
            'eta_min': 1e-6,  # Minimum learning rate
            'grad_clip': 1.0,
            'mixed_precision': True
        }
        if config:
            self.config.update(config)
        
        # Initialize model
        self.model = SpineSegmentationModel().to(self.config['device'])
        
        # Create datasets and dataloaders
        self.train_dataset = SpineSegmentationDataset(
            self.data_root,
            mode='train'
        )
        self.val_dataset = SpineSegmentationDataset(
            self.data_root,
            mode='val'
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        # Setup loss, optimizer and scheduler
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['T_0'],
            eta_min=self.config['eta_min']
        )
        
        # Setup mixed precision training
        self.scaler = GradScaler() if self.config['mixed_precision'] else None
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
        
        # Initialize wandb
        wandb.init(project='spine-segmentation', config=self.config)
    
    def train(self):
        """Run complete training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate_epoch()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint and check early stopping
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                self._save_checkpoint('best.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['patience']:
                    logger.info("Early stopping triggered")
                    break
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
    
    def _train_epoch(self) -> Dict:
        """Run one epoch of training"""
        self.model.train()
        epoch_metrics = {
            'train_loss': 0,
            'train_dice': 0,
            'train_iou': 0
        }
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}") as pbar:
            for batch in pbar:
                images = batch['image'].to(self.config['device'])
                masks = batch['mask'].to(self.config['device'])
                
                # Forward pass with mixed precision
                with autocast(enabled=self.config['mixed_precision']):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                
                # Backward pass
                self.optimizer.zero_grad()
                if self.config['mixed_precision']:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )
                    self.optimizer.step()
                
                # Calculate metrics
                with torch.no_grad():
                    predictions = torch.sigmoid(outputs) > 0.5
                    dice = calculate_dice_score(predictions, masks)
                    iou = calculate_iou(predictions, masks)
                
                # Update metrics
                epoch_metrics['train_loss'] += loss.item()
                epoch_metrics['train_dice'] += dice.item()
                epoch_metrics['train_iou'] += iou.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'dice': dice.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
        
        # Average metrics
        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _validate_epoch(self) -> Dict:
        """Run validation"""
        self.model.eval()
        val_metrics = {
            'val_loss': 0,
            'val_dice': 0,
            'val_iou': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.config['device'])
                masks = batch['mask'].to(self.config['device'])
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                predictions = torch.sigmoid(outputs) > 0.5
                dice = calculate_dice_score(predictions, masks)
                iou = calculate_iou(predictions, masks)
                
                val_metrics['val_loss'] += loss.item()
                val_metrics['val_dice'] += dice.item()
                val_metrics['val_iou'] += iou.item()
        
        # Average metrics
        num_batches = len(self.val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_metrics
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to wandb and console"""
        metrics = {**train_metrics, **val_metrics}
        wandb.log(metrics, step=self.current_epoch)
        
        logger.info(
            f"Epoch {self.current_epoch+1} - "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Val Loss: {val_metrics['val_loss']:.4f}, "
            f"Train Dice: {train_metrics['train_dice']:.4f}, "
            f"Val Dice: {val_metrics['val_iou']:.4f}"
        )
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        save_path = self.model_save_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'scaler': self.scaler.state_dict() if self.scaler else None
        }, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint {checkpoint_path} does not exist")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.config['device'])
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and checkpoint['scaler']:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def predict(self, image: torch.Tensor) -> np.ndarray:
        """
        Make prediction with post-processing
        
        Args:
            image: Input image tensor [1, C, H, W]
            
        Returns:
            Processed segmentation mask
        """
        self.model.eval()
        with torch.no_grad():
            with autocast(enabled=self.config['mixed_precision']):
                output = self.model(image.to(self.config['device']))
                pred = torch.sigmoid(output) > 0.5
                
        # Convert to numpy
        mask = pred.cpu().numpy()[0]
        
        # Post-process
        processed_mask = self._post_process_prediction(mask)
        
        return processed_mask
    
    def _post_process_prediction(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to prediction mask
        
        Args:
            mask: Raw prediction mask [C, H, W]
            
        Returns:
            Processed mask
        """
        processed = mask.copy()
        
        # 1. Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        for c in range(processed.shape[0]):
            processed[c] = cv2.morphologyEx(
                processed[c].astype(np.uint8),
                cv2.MORPH_OPEN,
                kernel
            )
            processed[c] = cv2.morphologyEx(
                processed[c],
                cv2.MORPH_CLOSE,
                kernel
            )
        
        # 2. Remove small components
        for c in range(processed.shape[0]):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                processed[c].astype(np.uint8),
                connectivity=8
            )
            
            # Remove components smaller than threshold
            min_size = 50  # pixels
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < min_size:
                    labels[labels == i] = 0
            
            processed[c] = labels > 0
        
        # 3. Apply smoothing
        for c in range(processed.shape[0]):
            processed[c] = gaussian_filter(processed[c].astype(float), sigma=0.5)
            processed[c] = processed[c] > 0.5
        
        return processed