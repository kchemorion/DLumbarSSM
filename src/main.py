import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing.data_loader import SpineDataLoader
from preprocessing.image_preprocessor import SpineImagePreprocessor, SpineSegmentationPreprocessor
from preprocessing.feature_extractor import SpineFeatureExtractor
from segmentation.segmentation_pipeline import SpineSegmentation
from shape_modeling.statistical_shape_model import HierarchicalSpineModel, SpineInstance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpineModelingPipeline:
    """
    Main pipeline for building and analyzing hierarchical statistical shape models
    of the lumbar spine.
    """
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        
        # Initialize paths
        self.data_root = Path(self.config['paths']['data_root'])
        self.results_root = Path(self.config['paths']['results_root'])
        self.models_root = Path(self.config['paths']['models_root'])
        
        # Ensure directories exist
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = SpineDataLoader(self.data_root)
        self.preprocessor = SpineImagePreprocessor()
        self.seg_preprocessor = SpineSegmentationPreprocessor(self.preprocessor)
        self.segmentation = SpineSegmentation(
            model_path=self.models_root / 'segmentation_model.pth'
        )
        self.feature_extractor = SpineFeatureExtractor(self.preprocessor)
        self.shape_model = HierarchicalSpineModel(
            n_components=self.config['model']['n_components']
        )
        
        # Initialize storage for results
        self.spine_instances = []
        self.analysis_results = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'paths': {
                'data_root': 'data',
                'results_root': 'results',
                'models_root': 'models'
            },
            'model': {
                'n_components': 10,
                'min_explained_variance': 0.95
            },
            'processing': {
                'batch_size': 16,
                'num_workers': 4
            },
            'visualization': {
                'plot_types': ['variation_modes', 'condition_correlations', 'shape_space']
            }
        }
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        logger.info("Starting spine modeling pipeline...")
        
        # 1. Load and preprocess data
        self._process_dataset()
        
        # 2. Build statistical shape model
        self._build_shape_model()
        
        # 3. Analyze variations
        self._analyze_variations()
        
        # 4. Generate visualizations
        self._create_visualizations()
        
        # 5. Save results
        self._save_results()
        
        logger.info("Pipeline completed successfully.")
        
    def _process_dataset(self):
        """Process all cases in the dataset"""
        logger.info("Loading and processing dataset...")
        
        # Load annotations
        self.loader.load_annotations()
        
        # Process each case
        for study_id in tqdm(self.loader.train_csv['study_id'].values):
            try:
                # Load study data
                study_data = self.loader.get_patient_data(str(study_id))
                
                # Process each series
                segmentations = {}
                for series_id, images in study_data['images'].items():
                    if not images:
                        continue
                    
                    # Preprocess for segmentation
                    volume, metadata = self.seg_preprocessor.prepare_for_segmentation(images)
                    
                    # Perform segmentation
                    segmentation = self.segmentation.segment_volume(volume)
                    segmentations[series_id] = segmentation
                
                # Extract features and create spine instance
                spine_instance = self.feature_extractor.extract_spine_features(
                    study_data, segmentations
                )
                
                self.spine_instances.append(spine_instance)
                
            except Exception as e:
                logger.error(f"Error processing study {study_id}: {str(e)}")
                continue
    
    def _build_shape_model(self):
        """Build the hierarchical statistical shape model"""
        logger.info("Building statistical shape model...")
        
        # Fit the model to all spine instances
        self.shape_model.fit(self.spine_instances)
        
        # Save the model
        self._save_model()
    
    def _analyze_variations(self):
        """Analyze shape variations and correlations"""
        logger.info("Analyzing shape variations...")
        
        # 1. Analyze global shape variations
        self.analysis_results['global_variations'] = {
            'explained_variance': self.shape_model.global_pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.shape_model.global_pca.explained_variance_ratio_)
        }
        
        # 2. Analyze level-specific variations
        level_variations = {}
        for level, model in self.shape_model.level_models.items():
            level_variations[level] = {
                'explained_variance': model['pca'].explained_variance_ratio_,
                'cumulative_variance': np.cumsum(model['pca'].explained_variance_ratio_)
            }
        self.analysis_results['level_variations'] = level_variations
        
        # 3. Analyze condition-specific patterns
        self.analysis_results['condition_patterns'] = self._analyze_condition_patterns()
    
    def _analyze_condition_patterns(self) -> Dict:
        """Analyze patterns specific to each condition"""
        patterns = {}
        
        for condition in ['stenosis', 'foraminal_narrowing', 'subarticular_stenosis']:
            # Group instances by condition severity
            severity_groups = {
                'Normal/Mild': [],
                'Moderate': [],
                'Severe': []
            }
            
            for instance in self.spine_instances:
                if condition in instance.condition_states:
                    severity = instance.condition_states[condition]
                    severity_groups[severity].append(instance)
            
            # Compute mean shapes for each severity
            mean_shapes = {}
            for severity, instances in severity_groups.items():
                if instances:
                    shapes = np.stack([inst.global_shape for inst in instances])
                    mean_shapes[severity] = np.mean(shapes, axis=0)
            
            patterns[condition] = mean_shapes
        
        return patterns
    
    def _create_visualizations(self):
        """Create visualizations of the results"""
        logger.info("Creating visualizations...")
        
        # 1. Plot variation modes
        self._plot_variation_modes()
        
        # 2. Plot condition correlations
        self._plot_condition_correlations()
        
        # 3. Plot shape space
        self._plot_shape_space()
    
    def _plot_variation_modes(self):
        """Plot principal modes of variation"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Principal Modes of Variation')
        
        # Global variations
        axes[0, 0].plot(self.analysis_results['global_variations']['cumulative_variance'])
        axes[0, 0].set_title('Global Shape Variations')
        axes[0, 0].set_xlabel('Components')
        axes[0, 0].set_ylabel('Cumulative Explained Variance')
        
        # Level variations
        for i, (level, vars) in enumerate(self.analysis_results['level_variations'].items()):
            ax = axes.flat[i+1]
            ax.plot(vars['cumulative_variance'])
            ax.set_title(f'Level {level} Variations')
            ax.set_xlabel('Components')
            ax.set_ylabel('Cumulative Explained Variance')
        
        plt.tight_layout()
        plt.savefig(self.results_root / 'variation_modes.png')
        plt.close()
    
    def _plot_condition_correlations(self):
        """Plot correlations between conditions and shape variations"""
        # Extract condition severities and shape parameters
        conditions = []
        shape_params = []
        
        for instance in self.spine_instances:
            for condition, severity in instance.condition_states.items():
                conditions.append((condition, severity))
                shape_params.append(instance.global_shape[:5])  # Use first 5 components
        
        # Create correlation matrix
        corr_matrix = np.corrcoef(np.array(shape_params).T)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Shape Parameter Correlations')
        plt.savefig(self.results_root / 'condition_correlations.png')
        plt.close()
    
    def _plot_shape_space(self):
        """Plot the shape space with condition annotations"""
        # Project instances into 2D shape space
        shape_vectors = np.stack([inst.global_shape for inst in self.spine_instances])
        projected = self.shape_model.global_pca.transform(shape_vectors)[:, :2]
        
        # Plot with condition colors
        plt.figure(figsize=(12, 8))
        conditions = [list(inst.condition_states.values())[0] for inst in self.spine_instances]
        
        for condition in set(conditions):
            mask = np.array(conditions) == condition
            plt.scatter(projected[mask, 0], projected[mask, 1], label=condition, alpha=0.6)
        
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Shape Space Distribution')
        plt.legend()
        plt.savefig(self.results_root / 'shape_space.png')
        plt.close()
    
    def _save_model(self):
        """Save the trained shape model"""
        model_path = self.models_root / 'shape_model.pth'
        torch.save({
            'model_state': self.shape_model.__dict__,
            'config': self.config
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def _save_results(self):
        """Save analysis results"""
        # Save numerical results
        results_file = self.results_root / 'analysis_results.npz'
        np.savez(results_file, **self.analysis_results)
        
        # Save summary report
        self._generate_summary_report()
        
    def _generate_summary_report(self):
        """Generate a summary report of the analysis"""
        report_path = self.results_root / 'summary_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Spine Shape Analysis Report\n\n")
            
            # Dataset statistics
            f.write("## Dataset Statistics\n")
            f.write(f"Total cases analyzed: {len(self.spine_instances)}\n\n")
            
            # Variation analysis
            f.write("## Shape Variations\n")
            f.write("### Global Shape Variations\n")
            var_explained = self.analysis_results['global_variations']['explained_variance']
            f.write(f"Top 3 components explain {var_explained[:3].sum()*100:.1f}% of variation\n\n")
            
            # Condition patterns
            f.write("## Condition Patterns\n")
            for condition, patterns in self.analysis_results['condition_patterns'].items():
                f.write(f"### {condition}\n")
                for severity, shape in patterns.items():
                    f.write(f"- {severity}: {len(shape)} instances\n")
                f.write("\n")

def main():
    """Main entry point"""
    try:
        # Initialize and run pipeline
        pipeline = SpineModelingPipeline()
        pipeline.run_pipeline()
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise
    
if __name__ == "__main__":
    main()