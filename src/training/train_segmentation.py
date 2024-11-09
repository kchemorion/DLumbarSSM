# train_segmentation.py

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
    
    # Load configuration
    config_path = Path('config/training_config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = SegmentationTrainer(
        data_root=Path('data'),
        model_save_dir=Path('models'),
        config=config
    )
    
    # Resume from checkpoint if exists
    checkpoint_path = Path('models/latest.pth')
    if checkpoint_path.exists():
        trainer.load_checkpoint(checkpoint_path)
    
    try:
        # Run training
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
    finally:
        # Save final checkpoint
        trainer._save_checkpoint('final.pth')
        wandb.finish()

if __name__ == "__main__":
    main()