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
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
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
    """Production-ready training pipeline for spine segmentation"""
    def __init__(self,
                 data_root: Path,
                 model_save_dir: Path,
                 config: Optional[Dict] = None):
        self.data_root = Path(data_root)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup configuration
        self._setup_config(config)
        
        # Initialize model and components after config setup
        self._setup_model()
        self._setup_datasets()
        self._setup_training_components()

    def _setup_config(self, config: Optional[Dict]):
        """Setup configuration with device setup"""
        # Default configuration
        self.config = {
            'batch_size': 4,
            'num_epochs': 10,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_workers': 2,
            'patience': 5,
            'T_0': 5,
            'eta_min': 1e-6,
            'grad_clip': 1.0,
            'mixed_precision': False,  # Disabled for CPU
            'val_split': 0.2,
            'seed': 42,
            'pin_memory': False,
            'accumulation_steps': 4,
            'max_samples': 5000,  # Limit dataset size
            'target_size': (256, 256)  # Fixed image size
        }

        # Update with provided config
        if config:
            self.config.update(config)

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config['device'] = self.device
        
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['seed'])
    
    def _validate_config(self):
        """Validate configuration values"""
        validations = {
            'batch_size': (int, (1, 512)),
            'num_epochs': (int, (1, 1000)),
            'learning_rate': (float, (1e-6, 1.0)),
            'weight_decay': (float, (0.0, 0.1)),
            'num_workers': (int, (0, os.cpu_count() or 1)),
            'patience': (int, (1, 100)),
            'T_0': (int, (1, 100)),
            'eta_min': (float, (1e-12, 1e-3)),
            'grad_clip': (float, (0.0, 100.0)),
            'val_split': (float, (0.0, 0.5))
        }
        
        for key, (dtype, (min_val, max_val)) in validations.items():
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
            
            val = self.config[key]
            if not isinstance(val, dtype):
                raise TypeError(f"Config {key} must be {dtype}, got {type(val)}")
            
            if not min_val <= val <= max_val:
                raise ValueError(
                    f"Config {key} must be between {min_val} and {max_val}, "
                    f"got {val}"
                )
    
    def _setup_distributed(self):
        """Setup distributed training"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for distributed training")
        
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        torch.cuda.set_device(dist.get_rank())
        self.config['device'] = torch.device(f"cuda:{dist.get_rank()}")
        
        logger.info(
            f"Initialized distributed process group: "
            f"rank {dist.get_rank()}/{dist.get_world_size()}"
        )
    
    def _setup_model(self):
        """Initialize model"""
        self.model = SpineSegmentationModel().to(self.device)
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def _setup_datasets(self):
        """Setup datasets with proper splitting"""
        # Create full dataset
        full_dataset = SpineSegmentationDataset(
            data_root=self.data_root,
            mode='train',
            series_type="Sagittal T2/STIR",
            target_size=self.config['target_size'],
            max_samples=self.config['max_samples']
        )
        
        # Calculate split sizes
        val_size = int(len(full_dataset) * self.config['val_split'])
        train_size = len(full_dataset) - val_size
        
        # Split dataset
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config['seed'])
        )
        
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
    
    def _setup_training_components(self):
        """Setup loss, optimizer, and other training components"""
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
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
    
    def _setup_wandb(self):
        """Initialize wandb logging"""
        wandb.init(
            project='spine-segmentation',
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
                
                # Validation phase (only on main process if distributed)
                if not self.distributed or (self.distributed and dist.get_rank() == 0):
                    val_metrics = self._validate_epoch()
                    
                    # Log metrics
                    self._log_metrics(train_metrics, val_metrics)
                    
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
            if not self.distributed or (self.distributed and dist.get_rank() == 0):
                wandb.finish()
    
    def _train_epoch(self) -> Dict:
        """Run one epoch of training with gradient accumulation"""
        self.model.train()
        epoch_metrics = {
            'train_loss': 0,
            'train_dice': 0,
            'train_iou': 0
        }
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                images = batch['image'].to(self.config['device'])
                masks = batch['mask'].to(self.config['device'])
                
                # Forward pass with mixed precision
                with autocast(enabled=self.config['mixed_precision']):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    # Scale loss for gradient accumulation
                    loss = loss / self.config['accumulation_steps']
                
                # Backward pass with gradient accumulation
                if self.config['mixed_precision']:
                    self.scaler.scale(loss).backward()
                    if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['grad_clip']
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    loss.backward()
                    if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['grad_clip']
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                # Calculate metrics
                with torch.no_grad():
                    predictions = torch.sigmoid(outputs) > 0.5
                    dice = calculate_dice_score(predictions, masks)
                    iou = calculate_iou(predictions, masks)
                
                # Update metrics
                epoch_metrics['train_loss'] += loss.item() * self.config['accumulation_steps']
                epoch_metrics['train_dice'] += dice.item()
                epoch_metrics['train_iou'] += iou.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item() * self.config['accumulation_steps'],
                    'dice': dice.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                
                self.global_step += 1
        
        # Average metrics
        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    @torch.no_grad()
    def _validate_epoch(self) -> Dict:
        """Run validation with enhanced metrics"""
        self.model.eval()
        val_metrics = {
            'val_loss': 0,
            'val_dice': 0,
            'val_iou': 0,
            'val_precision': 0,
            'val_recall': 0
        }
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.config['device'])
            masks = batch['mask'].to(self.config['device'])
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Calculate metrics
            predictions = torch.sigmoid(outputs) > 0.5
            dice = calculate_dice_score(predictions, masks)
            iou = calculate_iou(predictions, masks)
            precision, recall = calculate_precision_recall(predictions, masks)
            
            # Update metrics
            val_metrics['val_loss'] += loss.item()
            val_metrics['val_dice'] += dice.item()
            val_metrics['val_iou'] += iou.item()
            val_metrics['val_precision'] += precision.item()
            val_metrics['val_recall'] += recall.item()
        
        # Average metrics
        num_batches = len(self.val_loader)
        for key in val_metrics:
            val_metrics[key] /= num_batches        
        
        return val_metrics
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Enhanced metric logging with error handling"""
        try:
            # Combine metrics
            metrics = {
                **train_metrics,
                **val_metrics,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch': self.current_epoch
            }
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log(metrics, step=self.global_step)
            
            # Log to console
            metric_str = (
                f"Epoch {self.current_epoch+1} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Train Dice: {train_metrics['train_dice']:.4f}, "
                f"Val Dice: {val_metrics['val_dice']:.4f}, "
                f"Val IoU: {val_metrics['val_iou']:.4f}"
            )
            logger.info(metric_str)
            
            # Store metrics for later analysis
            self.train_metrics['loss'].append(train_metrics['train_loss'])
            self.train_metrics['dice'].append(train_metrics['train_dice'])
            self.train_metrics['iou'].append(train_metrics['train_iou'])
            
            self.val_metrics['loss'].append(val_metrics['val_loss'])
            self.val_metrics['dice'].append(val_metrics['val_dice'])
            self.val_metrics['iou'].append(val_metrics['val_iou'])
            
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
    
    def _save_checkpoint(self, filename: str):
        """Enhanced checkpoint saving with error handling"""
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
                'best_val_loss': self.best_val_loss,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics,
                'config': self.config
            }
            
            # Save checkpoint with error handling
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")
            
            # Also save a backup
            if 'best' in filename or 'final' in filename:
                backup_path = save_path.parent / f"backup_{filename}"
                torch.save(checkpoint, backup_path)
                logger.info(f"Saved backup checkpoint to {backup_path}")
                
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            # Try to save in a different location as backup
            try:
                emergency_path = Path('emergency_checkpoint.pth')
                torch.save(checkpoint, emergency_path)
                logger.info(f"Saved emergency checkpoint to {emergency_path}")
            except:
                logger.error("Failed to save emergency checkpoint")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Enhanced checkpoint loading with validation"""
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            # Load checkpoint to CPU first to save memory
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Validate checkpoint content
            required_keys = {
                'epoch', 'model_state_dict', 'optimizer_state_dict',
                'scheduler_state_dict', 'config'
            }
            if not all(k in checkpoint for k in required_keys):
                raise ValueError(f"Checkpoint missing required keys: {required_keys}")
            
            # Update config if needed
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
            
            # Load scaler if it exists
            if self.scaler and checkpoint.get('scaler'):
                self.scaler.load_state_dict(checkpoint['scaler'])
            
            # Restore training state
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            # Restore metrics if they exist
            if 'train_metrics' in checkpoint:
                self.train_metrics = checkpoint['train_metrics']
            if 'val_metrics' in checkpoint:
                self.val_metrics = checkpoint['val_metrics']
            
            logger.info(f"Successfully loaded checkpoint from epoch {self.current_epoch}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
    
    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> np.ndarray:
        """
        Make prediction with enhanced error handling and post-processing
        
        Args:
            image: Input image tensor [B, C, H, W]
            
        Returns:
            Processed segmentation mask
        """
        try:
            self.model.eval()
            
            # Ensure input is on correct device
            image = image.to(self.config['device'])
            
            # Make prediction
            with autocast(enabled=self.config['mixed_precision']):
                output = self.model(image)
                pred = torch.sigmoid(output) > 0.5
            
            # Move to CPU and convert to numpy
            mask = pred.cpu().numpy()
            
            # Post-process each image in batch
            processed_masks = []
            for i in range(mask.shape[0]):
                processed = self._post_process_prediction(mask[i])
                processed_masks.append(processed)
            
            return np.stack(processed_masks)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def _post_process_prediction(self, mask: np.ndarray) -> np.ndarray:
        """
        Enhanced post-processing with better error handling
        
        Args:
            mask: Raw prediction mask [C, H, W]
            
        Returns:
            Processed mask
        """
        try:
            processed = mask.copy()
            
            # 1. Apply morphological operations
            kernel = np.ones((3, 3), np.uint8)
            for c in range(processed.shape[0]):
                # Open operation (remove small objects)
                processed[c] = cv2.morphologyEx(
                    processed[c].astype(np.uint8),
                    cv2.MORPH_OPEN,
                    kernel,
                    iterations=1
                )
                
                # Close operation (fill small holes)
                processed[c] = cv2.morphologyEx(
                    processed[c],
                    cv2.MORPH_CLOSE,
                    kernel,
                    iterations=1
                )
            
            # 2. Remove small components with area filtering
            for c in range(processed.shape[0]):
                # Find connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    processed[c].astype(np.uint8),
                    connectivity=8
                )
                
                # Remove small components
                min_size = max(50, processed[c].size * 0.001)  # Dynamic threshold
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < min_size:
                        labels[labels == i] = 0
                
                processed[c] = labels > 0
            
            # 3. Apply boundary smoothing
            for c in range(processed.shape[0]):
                # Convert to float for gaussian filter
                smooth = gaussian_filter(
                    processed[c].astype(float),
                    sigma=0.5,
                    mode='reflect'
                )
                processed[c] = smooth > 0.5
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            # Return original mask if post-processing fails
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