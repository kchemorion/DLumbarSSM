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
        # Initialize trainer with updated configuration
        config = {
            'batch_size': 1,
            'num_epochs': 50,
            'max_samples': 2000,
            'target_size': (256, 256),
            'mixed_precision': True,  # Enable mixed precision
            'num_workers': 2,
            'learning_rate': 1e-4,
            'min_lr': 1e-6,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0,
            'series_types': ["Sagittal T2/STIR", "Sagittal T1", "Axial T2"],
            'tasks': {
                'segmentation': True,
                'landmarks': True,
                'levels': True,
                'conditions': True
            },
            'scheduler': {
                'type': 'cosine',
                'T_0': 10,
                'eta_min': 1e-6
            },
            'augmentation': {
                'rotate_prob': 0.5,
                'flip_prob': 0.5,
                'contrast_prob': 0.3
            }
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


