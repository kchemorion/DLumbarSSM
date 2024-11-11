import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .data_loader import SpineDataLoader

class SpineDataAnalyzer:
    def __init__(self, loader: SpineDataLoader):
        self.loader = loader
        self.condition_stats = None
        self.level_stats = None
        
    def generate_dataset_statistics(self):
        """Generate comprehensive statistics about the dataset"""
        # Load all annotations
        self.loader.load_annotations()
        
        # Get condition distribution
        self.condition_stats = self.loader.get_condition_distribution()
        
        # Get level-specific statistics
        self.level_stats = self.loader.get_level_statistics()
        
        # Analyze image characteristics
        self.image_stats = self._analyze_image_characteristics()
        
        return {
            'condition_stats': self.condition_stats,
            'level_stats': self.level_stats,
            'image_stats': self.image_stats
        }
    
    def _analyze_image_characteristics(self):
        """Analyze characteristics of the imaging data"""
        image_stats = {
            'series_types': {},
            'intensity_stats': {},
            'dimension_stats': {}
        }
        
        # Analyze first 10 studies for quick statistics
        study_ids = self.loader.train_csv['study_id'].iloc[:10]
        
        for study_id in study_ids:
            patient_data = self.loader.get_patient_data(str(study_id))
            
            for series_id, images in patient_data['images'].items():
                if not images:
                    continue
                    
                # Get series type
                series_type = patient_data['series'][
                    patient_data['series']['series_id'] == series_id
                ]['series_description'].iloc[0]
                
                if series_type not in image_stats['series_types']:
                    image_stats['series_types'][series_type] = 0
                image_stats['series_types'][series_type] += 1
                
                # Analyze first image in series
                img = images[0]['image']
                
                # Intensity statistics
                image_stats['intensity_stats'][series_type] = {
                    'min': float(np.min(img)),
                    'max': float(np.max(img)),
                    'mean': float(np.mean(img)),
                    'std': float(np.std(img))
                }
                
                # Dimension statistics
                if series_type not in image_stats['dimension_stats']:
                    image_stats['dimension_stats'][series_type] = []
                image_stats['dimension_stats'][series_type].append(img.shape)
        
        return image_stats
    
    def visualize_statistics(self):
        """Create visualizations of the dataset statistics"""
        # Create output directory if it doesn't exist
        output_dir = Path('results/statistics')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Condition Distribution Plot
        plt.figure(figsize=(15, 8))
        for i, condition in enumerate(self.condition_stats.columns):
            plt.subplot(2, 3, i+1)
            self.condition_stats[condition].plot(kind='bar')
            plt.title(condition.replace('_', ' ').title())
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'condition_distribution.png')
        
        # 2. Level-specific statistics
        fig, axes = plt.subplots(5, 1, figsize=(15, 25))
        for i, (level, stats) in enumerate(self.level_stats.items()):
            axes[i].set_title(f'Level {level} Statistics')
            pd.concat([stats['stenosis_count'], 
                      stats['foraminal_count'], 
                      stats['subarticular_count']], 
                     axis=1).plot(kind='bar', ax=axes[i])
            axes[i].legend(['Stenosis', 'Foraminal', 'Subarticular'])
        plt.tight_layout()
        plt.savefig(output_dir / 'level_statistics.png')
        
        return output_dir