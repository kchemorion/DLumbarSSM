# src/training/train_segmentation.py

import wandb
from pathlib import Path
import logging
import yaml
from src.config.logging_config import setup_logging
from src.training.trainer import SegmentationTrainer

def main():
    # Setup logging
    setup_logging(log_file='results/training.log')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize trainer with memory-efficient settings
        config = {
            'batch_size': 4,
            'num_epochs': 10,
            'max_samples': 5000,  # Limit samples for initial testing
            'target_size': (256, 256),  # Smaller image size
            'mixed_precision': False,  # Disabled for CPU
            'num_workers': 2
        }
        
        trainer = SegmentationTrainer(
            data_root=Path('data'),
            model_save_dir=Path('models'),
            config=config
        )
        
        # Run training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()