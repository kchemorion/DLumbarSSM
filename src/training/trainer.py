#trainer.py
"""
Multi-view Multi-task Spine Analysis Trainer
Author: Francis Kiptengwer Chemorion
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from src.segmentation.models import MultiViewSpineNet, initialize_weights
from src.training.metrics import MultiTaskMetrics, MultiTaskLoss
from src.training.datasets import SpineSegmentationDataset
from contextlib import nullcontext
import torch.cuda.amp as amp
from torch.cuda.amp import autocast, GradScaler
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class SegmentationTrainer:
    def __init__(self,
                data_root: Path,
                model_save_dir: Path,
                config: Optional[Dict] = None):
        """Initialize the SegmentationTrainer with all required attributes"""
        self.data_root = Path(data_root)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking attributes
        self.distributed = False
        self.global_step = 0
        self.scaler = None
        
        # Initialize metric tracking
        self.metrics = MultiTaskMetrics()
        self.train_metrics = {}
        self.val_metrics = {}
        self.best_metrics = {
        'val_loss': float('inf'),
        'val_seg_dice': 0,
        'val_landmark_error': float('inf'),
        'val_level_accuracy': 0,
        'val_condition_accuracy': 0
    }
        
        # Setup configuration and components
        self._setup_config(config)
        self._setup_model()
        self._setup_datasets()
        self._setup_training_components()
        
        # Setup mixed precision if enabled
        self.mixed_precision = False
        if config.get('mixed_precision', False) and torch.cuda.is_available():
            self.mixed_precision = True
            self.scaler = GradScaler()  # Remove device_type argument
        else:
            self.scaler = None
        
        # Initialize wandb
        if not self.distributed or (self.distributed and dist.get_rank() == 0):
            self._setup_wandb()

    def _setup_config(self, config: Optional[Dict]):
        """Setup configuration with memory-optimized defaults"""
        self.config = {
            'batch_size': 1,  # Keep at 1
            'num_epochs': 50,
            'learning_rate': 1e-5,
            'weight_decay': 1e-4,
            'num_workers': 0,  # Reduce workers
            'patience': 10,
            'T_0': 10,
            'eta_min': 1e-6,
            'grad_clip': 0.5,
            'mixed_precision': True,
            'val_split': 0.2,
            'seed': 42,
            'pin_memory': False,
            'accumulation_steps': 4,
            'max_samples': 1000,  # Reduce samples
            'target_size': (128, 128),  # Reduce image size
            'series_types': ["Sagittal T2/STIR"],  # Use only one modality initially
            'tasks': {
                'segmentation': True,
                'landmarks': True,
                'levels': True,
                'conditions': True
            }
        }

        if config:
            self.config.update(config)

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config['device'] = self.device
        
        # Set memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True'


    # In trainer.py, after creating the model
    def _setup_model(self):
        """Initialize model"""
        try:
            self.model = MultiViewSpineNet(
                in_channels=1,
                num_classes=4,
                num_levels=5
            ).to(self.device)
            
            # Initialize weights
            initialize_weights(self.model)
            
            logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
            
        except Exception as e:
            logger.error(f"Error setting up model: {str(e)}")
            raise

    def _setup_datasets(self):
        """Setup datasets with proper error handling"""
        try:
            logger.info("Setting up datasets...")
            
            # Verify data directory exists
            if not self.data_root.exists():
                raise ValueError(f"Data root directory not found: {self.data_root}")
            
            # Create full dataset
            full_dataset = SpineSegmentationDataset(
                data_root=self.data_root,
                mode='train',
                series_types=self.config['series_types'],
                target_size=self.config['target_size'],
                max_samples=self.config['max_samples']
            )
            
            if len(full_dataset) == 0:
                raise ValueError("Dataset is empty")
            
            # Calculate split sizes
            val_size = int(len(full_dataset) * self.config['val_split'])
            train_size = len(full_dataset) - val_size
            
            if train_size == 0 or val_size == 0:
                raise ValueError(f"Invalid split sizes: train={train_size}, val={val_size}")
            
            # Split dataset
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.config['seed'])
            )
            
            logger.info(f"Dataset split: train={len(self.train_dataset)}, val={len(self.val_dataset)}")
            
            # Create data loaders
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory']
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=self.config['pin_memory']
            )
            
            logger.info("Dataset setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up datasets: {str(e)}")
            raise

    def _setup_training_components(self):
        """Setup loss, optimizer, and other training components"""
        # Initialize loss and metrics
        self.criterion = MultiTaskLoss(num_tasks=4)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['T_0'],
            eta_min=self.config['eta_min']
        )
        
        # Initialize AMP scaler
        if self.config['mixed_precision'] and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0

    def _setup_wandb(self):
        """Initialize wandb logging"""
        wandb.init(
            project='spine-analysis',
            name=f"multi-view-spine-model_{self.config['target_size'][0]}",
            config=self.config,
            resume=True if self.current_epoch > 0 else False
        )

    def train(self):
        """Run complete training loop with enhanced error handling"""
        try:
            logger.info("Starting training...")
            
            for epoch in range(self.current_epoch, self.config['num_epochs']):
                self.current_epoch = epoch
                
                if self.distributed:
                    self.train_loader.sampler.set_epoch(epoch)
                
                # Training phase
                train_metrics = self._train_epoch()
                
                # Validation phase
                if not self.distributed or (self.distributed and dist.get_rank() == 0):
                    val_metrics = self._validate_epoch()
                    self._log_metrics(train_metrics, val_metrics)
                    
                    # Check improvements
                    improved = self._check_improvements(val_metrics)
                    if improved:
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
                
                # Update learning rate
                self.scheduler.step()
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_checkpoint('interrupted.pth')
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            self._save_checkpoint('error.pth')
            raise
        finally:
            self.cleanup()

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = {
            'train_loss': 0,
            'train_seg_dice': 0,
            'train_landmark_error': 0,
            'train_level_accuracy': 0,
            'train_condition_accuracy': 0
        }
        
        # Clear cache before training loop
        torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move data to device
                    images = {k: v.to(self.device) for k, v in batch['images'].items()}
                    masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
                    landmarks = batch['landmarks'].to(self.device)
                    vertebral_levels = batch['vertebral_levels'].to(self.device)
                    conditions = batch['conditions'].to(self.device)
                    
                    targets = {
                        'segmentation': masks,
                        'landmarks': landmarks,
                        'vertebral_levels': vertebral_levels,
                        'conditions': conditions
                    }
                    
                    # Forward pass with gradient checkpointing
                    self.model.use_checkpointing = True
                    with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                        outputs = self.model(images)
                        loss, task_losses = self.criterion(outputs, targets)
                        loss = loss / self.config['accumulation_steps']
                    
                    # Clear unused tensors
                    del images
                    del outputs
                    del masks
                    del landmarks
                    del vertebral_levels
                    del conditions
                    torch.cuda.empty_cache()
                    
                    # Backward pass
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                    else:
                        loss.backward()
                        if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    
                    # Update metrics and clear memory
                    loss_value = loss.item()
                    epoch_metrics['train_loss'] += loss_value
                    del loss
                    torch.cuda.empty_cache()
                    
                    pbar.set_postfix({
                        'loss': loss_value,
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
                    
                    # Clear cache more frequently
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    torch.cuda.empty_cache()
                    continue
            
            num_batches = len(self.train_loader)
            epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
            
            return epoch_metrics

    def _validate_epoch(self) -> Dict[str, float]:
        """Validation with fixed autocast"""
        self.model.eval()
        val_metrics = {
            'val_loss': 0,
            'val_seg_dice': 0,
            'val_landmark_error': 0,
            'val_level_accuracy': 0,
            'val_condition_accuracy': 0
        }
        
        # Fixed autocast usage
        context_manager = nullcontext() if not self.mixed_precision else torch.cuda.amp.autocast()
        
        with torch.no_grad(), context_manager:
            for batch in tqdm(self.val_loader, desc="Validation"):
                try:
                    # Move data to device
                    images = {k: v.to(self.device) for k, v in batch['images'].items()}
                    masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
                    
                    targets = {
                        'segmentation': masks,
                        'landmarks': batch['landmarks'].to(self.device),
                        'vertebral_levels': batch['vertebral_levels'].to(self.device),
                        'conditions': batch['conditions'].to(self.device)
                    }
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss, task_losses = self.criterion(outputs, targets)
                    
                    # Update metrics
                    val_metrics['val_loss'] += loss.item()
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {str(e)}")
                    continue
        
        # Average metrics
        num_batches = len(self.val_loader)
        val_metrics = {k: v/num_batches for k, v in val_metrics.items()}
        
        return val_metrics

    def _compute_batch_metrics(self, outputs, targets):
        metrics = {}
        
        # Compute Dice with small epsilon
        dice_scores = []
        for name in outputs['segmentation']:
            dice = self._compute_dice(outputs['segmentation'][name], 
                                    targets['segmentation'][name],
                                    epsilon=1e-7)
            dice_scores.append(dice)
        metrics['dice'] = torch.mean(torch.stack(dice_scores))
        
        # Normalize landmark error by image size
        metrics['landmark_error'] = F.l1_loss(
            outputs['landmarks'], 
            targets['landmarks']
        ) / torch.tensor(self.config['target_size']).mean()
        
        # Use argmax for level accuracy
        level_preds = outputs['vertebral_levels'].argmax(dim=-1)
        level_targets = targets['vertebral_levels'].argmax(dim=-1)
        metrics['level_accuracy'] = (level_preds == level_targets).float().mean()
        
        return metrics

    def _check_improvements(self, val_metrics: Dict[str, float]) -> bool:
        """Check if any metrics improved"""
        improved = False
        
        # Check each metric
        if val_metrics['val_loss'] < self.best_metrics['val_loss']:
            self.best_metrics['val_loss'] = val_metrics['val_loss']
            improved = True
            
        if val_metrics['val_seg_dice'] > self.best_metrics['val_seg_dice']:
            self.best_metrics['val_seg_dice'] = val_metrics['val_seg_dice']
            improved = True
            
        return improved

    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to wandb and console"""
        try:
            # Combine metrics
            metrics = {
                **train_metrics,
                **val_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch': self.current_epoch
            }
            
            # Log to wandb
            # Log to wandb
            if wandb.run is not None:
                wandb.log(metrics, step=self.global_step)
            
            # Log to console
            metric_str = (
                f"Epoch {self.current_epoch+1} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Train Dice: {train_metrics['train_seg_dice']:.4f}, "
                f"Val Dice: {val_metrics['val_seg_dice']:.4f}, "
                f"Landmark Error: {val_metrics['val_landmark_error']:.4f}, "
                f"Level Acc: {val_metrics['val_level_accuracy']:.4f}"
            )
            logger.info(metric_str)
            
            # Store metrics
            for key, value in metrics.items():
                if key.startswith('train_'):
                    if key not in self.train_metrics:
                        self.train_metrics[key] = []
                    self.train_metrics[key].append(value)
                elif key.startswith('val_'):
                    if key not in self.val_metrics:
                        self.val_metrics[key] = []
                    self.val_metrics[key].append(value)
            
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        try:
            save_path = self.model_save_dir / filename
            
            # Prepare checkpoint data
            checkpoint = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.module.state_dict() if self.distributed 
                                  else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict() if self.scaler else None,
                'best_metrics': self.best_metrics,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics,
                'config': self.config
            }
            
            # Save checkpoint
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")
            
            # Save backup for important checkpoints
            if 'best' in filename or 'final' in filename:
                backup_path = save_path.parent / f"backup_{filename}"
                torch.save(checkpoint, backup_path)
                logger.info(f"Saved backup checkpoint to {backup_path}")
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            # Try emergency save
            try:
                emergency_path = Path('emergency_checkpoint.pth')
                torch.save(checkpoint, emergency_path)
                logger.info(f"Saved emergency checkpoint to {emergency_path}")
            except:
                logger.error("Failed to save emergency checkpoint")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint"""
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            # Load to CPU first
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Validate checkpoint content
            required_keys = {
                'epoch', 'model_state_dict', 'optimizer_state_dict',
                'scheduler_state_dict', 'config', 'best_metrics'
            }
            if not all(k in checkpoint for k in required_keys):
                raise ValueError(f"Checkpoint missing required keys: {required_keys}")
            
            # Update config
            self.config.update(checkpoint['config'])
            
            # Load model state
            if self.distributed:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load scaler if exists
            if self.scaler and checkpoint.get('scaler'):
                self.scaler.load_state_dict(checkpoint['scaler'])
            
            # Restore training state
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint.get('global_step', 0)
            self.best_metrics = checkpoint['best_metrics']
            
            # Restore metrics history
            if 'train_metrics' in checkpoint:
                self.train_metrics = checkpoint['train_metrics']
            if 'val_metrics' in checkpoint:
                self.val_metrics = checkpoint['val_metrics']
            
            logger.info(f"Successfully loaded checkpoint from epoch {self.current_epoch}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
    
    @torch.no_grad()
    def predict(self, 
               images: Dict[str, torch.Tensor], 
               return_all: bool = False) -> Dict[str, np.ndarray]:
        """
        Make predictions with the model
        
        Args:
            images: Dictionary of input images for each view
            return_all: Whether to return predictions for all tasks
            
        Returns:
            Dictionary of predictions
        """
        try:
            self.model.eval()
            
            # Move images to device
            images = {k: v.to(self.device) for k, v in images.items()}
            
            # Make prediction
            with autocast(enabled=self.config['mixed_precision']):
                outputs = self.model(images)
                
                # Process outputs
                predictions = {}
                
                # Segmentation predictions
                seg_preds = {k: torch.sigmoid(v) > 0.5 for k, v in outputs['segmentation'].items()}
                predictions['segmentation'] = {
                    k: self._post_process_prediction(v.cpu().numpy())
                    for k, v in seg_preds.items()
                }
                
                if return_all:
                    # Landmark predictions
                    predictions['landmarks'] = outputs['landmarks'].cpu().numpy()
                    
                    # Level predictions
                    predictions['vertebral_levels'] = torch.softmax(
                        outputs['vertebral_levels'], dim=1
                    ).cpu().numpy()
                    
                    # Condition predictions
                    predictions['conditions'] = torch.softmax(
                        outputs['conditions'], dim=1
                    ).cpu().numpy()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def _post_process_prediction(self, mask: np.ndarray) -> np.ndarray:
        """Post-process segmentation predictions"""
        try:
            processed = mask.copy()
            
            # 1. Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            for c in range(processed.shape[0]):
                # Remove small objects
                processed[c] = cv2.morphologyEx(
                    processed[c].astype(np.uint8),
                    cv2.MORPH_OPEN,
                    kernel,
                    iterations=1
                )
                
                # Fill small holes
                processed[c] = cv2.morphologyEx(
                    processed[c],
                    cv2.MORPH_CLOSE,
                    kernel,
                    iterations=1
                )
            
            # 2. Remove small components
            for c in range(processed.shape[0]):
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    processed[c].astype(np.uint8),
                    connectivity=8
                )
                
                min_size = max(50, processed[c].size * 0.001)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < min_size:
                        labels[labels == i] = 0
                
                processed[c] = labels > 0
            
            # 3. Smooth boundaries
            for c in range(processed.shape[0]):
                smooth = gaussian_filter(
                    processed[c].astype(float),
                    sigma=0.5,
                    mode='reflect'
                )
                processed[c] = smooth > 0.5
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            return mask

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.distributed:
                dist.destroy_process_group()
            
            if wandb.run is not None:
                wandb.finish()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")