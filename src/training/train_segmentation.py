"""
Memory-Optimized Multi-view Spine Segmentation Training Script
Author: Francis Kiptengwer Chemorion
"""

import os
import wandb
import torch
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional
import argparse
import torch.cuda.amp as amp
import logging
from src.config.logging_config import setup_logging
from src.training.trainer import SegmentationTrainer
from src.segmentation.models import enable_memory_efficient_mode

logger = logging.getLogger(__name__)


# Configure CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'max_split_size_mb:128,'
    'garbage_collection_threshold:0.8,'
    'expandable_segments:True'
)
torch.backends.cudnn.benchmark = True

def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load and validate configuration with memory optimizations"""
    default_config = {
        'training': {
            'batch_size': 1,
            'num_epochs': 50,
            'max_samples': 2000,
            'learning_rate': 1e-4,
            'min_lr': 1e-6,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0,
            'patience': 10,
            'val_split': 0.2,
            'gradient_accumulation_steps': 4
        },
        'data': {
            'target_size': (128, 128),  # Reduced size for memory efficiency
            'series_types': ["Sagittal T2/STIR"],  # Start with one modality
            'num_workers': 2,  # Reduced workers
            'pin_memory': True,
            'prefetch_factor': 2
        },
        'model': {
            'in_channels': 1,
            'num_classes': 4,
            'num_levels': 5,
            'attention': {
                'chunk_size': 1024,
                'num_heads': 4,
                'dropout': 0.1
            },
            'feature_dims': [32, 64, 128],  # Reduced dimensions
            'enable_checkpointing': True,
            'channels_last': True,  # Memory format optimization
        },
        'optimization': {
            'mixed_precision': True,
            'gradient_checkpointing': True,
            'empty_cache_freq': 100,  # Clear CUDA cache frequency
            'compile_model': True,  # Enable torch.compile
            'optimizer': {
                'type': 'AdamW',
                'lr': 1e-4,
                'weight_decay': 1e-5,
                'gradient_clip': 1.0
            },
            'scheduler': {
                'type': 'cosine',
                'T_0': 10,
                'eta_min': 1e-6
            }
        },
        'memory': {
            'clear_cache_freq': 50,
            'max_batch_size': 1,
            'enable_chunking': True,
            'chunk_size': 1024
        }
    }

    if config_path and config_path.exists():
        with open(config_path) as f:
            loaded_config = yaml.safe_load(f)
            # Deep merge loaded config with defaults
            for key, value in loaded_config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value

    return default_config

def setup_environment(config: Dict) -> None:
    """Setup optimized training environment"""
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Create directories
    for path in ['models', 'logs', 'results']:
        Path(path).mkdir(parents=True, exist_ok=True)
    
    # Optimize CUDA settings
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocator settings
        torch.cuda.set_per_process_memory_fraction(0.95)  # Reserve some memory
        
        # Enable anomaly detection in debug mode
        if config.get('debug', False):
            torch.autograd.set_detect_anomaly(True)

def setup_model_optimizations(trainer: SegmentationTrainer, config: Dict) -> None:
    """Apply memory and performance optimizations to model"""
    # Enable memory efficient mode
    enable_memory_efficient_mode(trainer.model)
    
    # Use channels_last memory format
    if config['model']['channels_last']:
        trainer.model = trainer.model.to(memory_format=torch.channels_last)
    
    # Enable torch.compile if available and requested
    if config['optimization']['compile_model'] and hasattr(torch, 'compile'):
        trainer.model = torch.compile(trainer.model)
    
    # Move model to device
    trainer.model = trainer.model.to(trainer.device)

def main():
    parser = argparse.ArgumentParser(description='Train Multi-view Spine Segmentation Model')
    parser.add_argument('--config', type=Path, help='Path to config file')
    parser.add_argument('--resume', type=Path, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Setup logging
    log_path = Path('logs/training.log')
    setup_logging(log_file=log_path)
    logger = logging.getLogger(__name__)

    try:
        # Load and setup
        config = load_config(args.config)
        config['debug'] = args.debug
        setup_environment(config)
        
        logger.info("Initializing training...")
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device_name}")
        
        if torch.cuda.is_available():
            memory_info = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Available GPU memory: {memory_info:.2f} GB")
        
        # Initialize trainer
        trainer = SegmentationTrainer(
            data_root=Path('data'),
            model_save_dir=Path('models'),
            config=config
        )
        
        # Apply optimizations
        setup_model_optimizations(trainer, config)

        # Resume from checkpoint if specified
        if args.resume and args.resume.exists():
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Start training with memory monitoring
        trainer.train()

    except torch.cuda.OutOfMemoryError as e:
        logger.error("CUDA out of memory error. Try reducing batch size or model size.")
        raise
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'trainer' in locals():
            trainer.cleanup()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()