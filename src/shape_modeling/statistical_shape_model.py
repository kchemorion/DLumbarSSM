"""
Hierarchical Statistical Shape Model for Spine Analysis

This module implements a comprehensive statistical shape model for analyzing
lumbar spine variations and degenerative conditions. It provides functionality
for building hierarchical models that capture both global spine shape variations
and level-specific changes associated with different pathological conditions.

Author: Your Name
Created: 2024-02-08
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelValidationMetrics:
    """Metrics for validating shape model performance"""
    reconstruction_error: float
    explained_variance_ratio: float
    compactness: float
    specificity: float
    generalization: float
    validation_timestamps: List[float] = field(default_factory=list)

@dataclass
class VertebralLevel:
    """Represents a single vertebral level with its associated measurements"""
    level_id: str  # e.g., 'L1_L2'
    canal_diameter: float
    left_foraminal_width: float
    right_foraminal_width: float
    left_subarticular_space: float
    right_subarticular_space: float
    coordinates: np.ndarray  # Key anatomical landmarks
    severity_grade: str  # 'Normal/Mild', 'Moderate', 'Severe'
    
    def __post_init__(self):
        """Validate measurements after initialization"""
        self._validate_measurements()
    
    def _validate_measurements(self) -> None:
        """Validate measurement values are within anatomically possible ranges"""
        if not 0 < self.canal_diameter < 30:  # mm
            raise ValueError(f"Invalid canal diameter: {self.canal_diameter}")
        if not 0 < self.left_foraminal_width < 20:  # mm
            raise ValueError(f"Invalid left foraminal width: {self.left_foraminal_width}")
        if not 0 < self.right_foraminal_width < 20:  # mm
            raise ValueError(f"Invalid right foraminal width: {self.right_foraminal_width}")
        if not 0 < self.left_subarticular_space < 10:  # mm
            raise ValueError(f"Invalid left subarticular space: {self.left_subarticular_space}")
        if not 0 < self.right_subarticular_space < 10:  # mm
            raise ValueError(f"Invalid right subarticular space: {self.right_subarticular_space}")
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ValueError(f"Invalid coordinates shape: {self.coordinates.shape}")
        if self.severity_grade not in ['Normal/Mild', 'Moderate', 'Severe']:
            raise ValueError(f"Invalid severity grade: {self.severity_grade}")

@dataclass
class SpineInstance:
    """Represents a complete spine instance with all levels and conditions"""
    study_id: str
    levels: Dict[str, VertebralLevel]
    global_shape: np.ndarray  # Global spine shape parameters
    condition_states: Dict[str, str]  # Overall condition classifications
    
    def __post_init__(self):
        """Validate instance after initialization"""
        self._validate_instance()
    
    def _validate_instance(self) -> None:
        """Validate instance data"""
        required_levels = {'L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1'}
        if not all(level in self.levels for level in required_levels):
            raise ValueError(f"Missing required levels. Found: {set(self.levels.keys())}")
        if not isinstance(self.global_shape, np.ndarray):
            raise ValueError("global_shape must be a numpy array")
        required_conditions = {'stenosis', 'foraminal_narrowing', 'subarticular_stenosis'}
        if not all(cond in self.condition_states for cond in required_conditions):
            raise ValueError(f"Missing required conditions. Found: {set(self.condition_states.keys())}")

class HierarchicalSpineModel:
    """
    Hierarchical Statistical Shape Model for lumbar spine analysis.
    Handles both global spine shape and level-specific variations.
    """
    
    def __init__(self, 
                 n_components: int = 10,
                 validation_split: float = 0.2,
                 random_state: int = 42) -> None:
        """
        Initialize the hierarchical shape model.
        
        Args:
            n_components: Number of principal components to retain
            validation_split: Fraction of data to use for validation
            random_state: Random seed for reproducibility
        """
        self._validate_init_params(n_components, validation_split)
        
        self.n_components = n_components
        self.validation_split = validation_split
        self.random_state = random_state
        
        # Initialize model components
        self.global_pca = PCA(n_components=n_components, random_state=random_state)
        self.level_models: Dict[str, Dict] = {}
        self.condition_models: Dict[str, Dict] = {}
        
        # Initialize statistical parameters
        self.mean_shape: Optional[np.ndarray] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        
        # Initialize validation metrics
        self.validation_metrics = ModelValidationMetrics(
            reconstruction_error=float('inf'),
            explained_variance_ratio=0.0,
            compactness=0.0,
            specificity=0.0,
            generalization=0.0
        )
        
        logger.info(f"Initialized HierarchicalSpineModel with {n_components} components")
    
    @staticmethod
    def _validate_init_params(n_components: int, validation_split: float) -> None:
        """Validate initialization parameters"""
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError(f"Invalid n_components: {n_components}")
        if not 0 < validation_split < 1:
            raise ValueError(f"Invalid validation_split: {validation_split}")
    
    def fit(self, spine_instances: List[SpineInstance]) -> None:
        """
        Fit the hierarchical model to a set of spine instances.
        
        Args:
            spine_instances: List of SpineInstance objects to fit the model to
        """
        try:
            logger.info(f"Starting model fitting with {len(spine_instances)} instances")
            
            # Validate input data
            self._validate_spine_instances(spine_instances)
            
            # Split data for validation
            train_instances, val_instances = self._split_validation_data(spine_instances)
            
            # 1. Extract and align global shapes
            global_shapes = self._extract_global_shapes(train_instances)
            aligned_shapes = self._align_shapes(global_shapes)
            
            # 2. Compute global statistics
            self.mean_shape = np.mean(aligned_shapes, axis=0)
            self.global_pca.fit(aligned_shapes)
            
            # 3. Build level-specific models
            self._build_level_models(train_instances)
            
            # 4. Build condition-specific models
            self._build_condition_models(train_instances)
            
            # 5. Validate model
            self._validate_model(val_instances)
            
            logger.info("Model fitting completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise
    
    def _validate_spine_instances(self, instances: List[SpineInstance]) -> None:
        """Validate spine instances before fitting"""
        if not instances:
            raise ValueError("Empty instance list")
        if not all(isinstance(inst, SpineInstance) for inst in instances):
            raise ValueError("All instances must be SpineInstance objects")
        
    def _split_validation_data(self, 
                             instances: List[SpineInstance]
                             ) -> Tuple[List[SpineInstance], List[SpineInstance]]:
        """Split data into training and validation sets"""
        np.random.seed(self.random_state)
        n_val = int(len(instances) * self.validation_split)
        indices = np.random.permutation(len(instances))
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        return ([instances[i] for i in train_indices], 
                [instances[i] for i in val_indices])
    
    def _extract_global_shapes(self, instances: List[SpineInstance]) -> np.ndarray:
        """Extract global shape vectors from instances"""
        try:
            shapes = np.stack([inst.global_shape for inst in instances])
            if shapes.ndim != 2:
                raise ValueError(f"Invalid shape array dimensions: {shapes.ndim}")
            return shapes
        except Exception as e:
            logger.error(f"Error extracting global shapes: {str(e)}")
            raise
    
    def _align_shapes(self, shapes: np.ndarray) -> np.ndarray:
        """
        Align shapes using Generalized Procrustes Analysis.
        
        Args:
            shapes: Array of shape vectors to align
        
        Returns:
            Array of aligned shape vectors
        """
        try:
            n_shapes = shapes.shape[0]
            aligned = shapes.copy()
            reference = aligned[0]
            max_iterations = 100
            tolerance = 1e-6
            
            for iteration in tqdm(range(max_iterations), desc="Aligning shapes"):
                previous_reference = reference.copy()
                
                # Align each shape to reference
                for i in range(n_shapes):
                    R, _ = orthogonal_procrustes(aligned[i], reference)
                    aligned[i] = aligned[i] @ R
                
                # Update reference
                reference = np.mean(aligned, axis=0)
                reference /= np.linalg.norm(reference)
                
                # Check convergence
                if np.linalg.norm(reference - previous_reference) < tolerance:
                    logger.info(f"Shape alignment converged after {iteration+1} iterations")
                    break
            
            return aligned
            
        except Exception as e:
            logger.error(f"Error during shape alignment: {str(e)}")
            raise
    
    def _build_level_models(self, instances: List[SpineInstance]) -> None:
        """Build statistical models for each vertebral level"""
        try:
            levels = ['L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1']
            
            for level in tqdm(levels, desc="Building level models"):
                level_data = []
                for inst in instances:
                    if level in inst.levels:
                        level_data.append(self._extract_level_features(inst.levels[level]))
                
                if level_data:
                    level_data = np.stack(level_data)
                    self.level_models[level] = {
                        'pca': PCA(n_components=self.n_components,
                                 random_state=self.random_state).fit(level_data),
                        'mean': np.mean(level_data, axis=0),
                        'std': np.std(level_data, axis=0)
                    }
                    
        except Exception as e:
            logger.error(f"Error building level models: {str(e)}")
            raise
    
    def _validate_model(self, val_instances: List[SpineInstance]) -> None:
        """
        Validate the fitted model using various metrics.
        
        Args:
            val_instances: List of validation instances
        """
        try:
            # 1. Compute reconstruction error
            reconstruction_errors = []
            for instance in val_instances:
                reconstructed = self.generate_new_instance(
                    weights=self.global_pca.transform(instance.global_shape.reshape(1, -1))[0]
                )
                error = np.mean((instance.global_shape - reconstructed.global_shape) ** 2)
                reconstruction_errors.append(error)
            
            self.validation_metrics.reconstruction_error = np.mean(reconstruction_errors)
            
            # 2. Compute explained variance ratio
            self.validation_metrics.explained_variance_ratio = \
                np.sum(self.global_pca.explained_variance_ratio_)
            
            # 3. Compute model compactness
            self.validation_metrics.compactness = \
                self._compute_compactness(val_instances)
            
            # 4. Compute model specificity
            self.validation_metrics.specificity = \
                self._compute_specificity(val_instances)
            
            # 5. Compute model generalization
            self.validation_metrics.generalization = \
                self._compute_generalization(val_instances)
            
            logger.info("Model validation completed successfully")
            logger.info(f"Validation metrics: {self.validation_metrics}")
            
        except Exception as e:
            logger.error(f"Error during model validation: {str(e)}")
            raise
    
    def _compute_compactness(self, instances: List[SpineInstance]) -> float:
        """Compute model compactness metric"""
        try:
            variances = self.global_pca.explained_variance_ratio_
            return np.sum(variances[:self.n_components]) / np.sum(variances)
        except Exception as e:
            logger.error(f"Error computing compactness: {str(e)}")
            raise
    
    def _compute_specificity(self, instances: List[SpineInstance]) -> float:
        """Compute model specificity metric"""
        try:
            # Generate random instances and compute distance to training set
            n_samples = 100
            distances = []
            
            for _ in range(n_samples):
                random_instance = self.generate_new_instance()
                min_dist = float('inf')
                
                for instance in instances:
                    dist = np.mean((random_instance.global_shape - instance.global_shape) ** 2)
                    min_dist = min(min_dist, dist)
                
                distances.append(min_dist)
            
            return np.mean(distances)
            
        except Exception as e:
            logger.error(f"Error computing specificity: {str(e)}")
            raise
    
    def _compute_generalization(self, instances: List[SpineInstance]) -> float:
        """Compute model generalization metric"""
        try:
            # Leave-one-out cross validation
            errors = []
            
            for i, test_instance in enumerate(instances):
                train_instances = instances[:i] + instances[i+1:]
                temp_model = HierarchicalSpineModel(
                    n_components=self.n_components,
                    random_state=self.random_state
                )
                temp_model.fit(train_instances)
                
                reconstructed = temp_model.generate_new_instance(
                    weights=temp_model.global_pca.transform(
                        test_instance.global_shape.reshape(1, -1)
                    )[0]
                )
                
                error = np.mean((test_instance.global_shape - reconstructed.global_shape) ** 2)
                errors.append(error)
            
            return np.mean(errors)
            
        except Exception as e:
                        logger.error(f"Error computing generalization: {str(e)}")
                        raise
    
    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            save_path: Path where model should be saved
        """
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_state = {
                'n_components': self.n_components,
                'validation_split': self.validation_split,
                'random_state': self.random_state,
                'global_pca': self.global_pca,
                'level_models': self.level_models,
                'condition_models': self.condition_models,
                'mean_shape': self.mean_shape,
                'eigenvectors': self.eigenvectors,
                'eigenvalues': self.eigenvalues,
                'validation_metrics': self.validation_metrics
            }
            
            torch.save(model_state, save_path)
            logger.info(f"Model saved successfully to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, model_path: Union[str, Path]) -> 'HierarchicalSpineModel':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Loaded HierarchicalSpineModel instance
        """
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model_state = torch.load(model_path)
            
            # Create new model instance
            model = cls(
                n_components=model_state['n_components'],
                validation_split=model_state['validation_split'],
                random_state=model_state['random_state']
            )
            
            # Restore model state
            model.global_pca = model_state['global_pca']
            model.level_models = model_state['level_models']
            model.condition_models = model_state['condition_models']
            model.mean_shape = model_state['mean_shape']
            model.eigenvectors = model_state['eigenvectors']
            model.eigenvalues = model_state['eigenvalues']
            model.validation_metrics = model_state['validation_metrics']
            
            logger.info(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_new_instance(self, 
                            weights: Optional[np.ndarray] = None,
                            condition_constraints: Optional[Dict[str, str]] = None
                            ) -> SpineInstance:
        """
        Generate a new spine instance using the statistical model.
        
        Args:
            weights: Optional weights for shape parameters. If None, random weights are generated.
            condition_constraints: Optional constraints on conditions (e.g., {'stenosis': 'Severe'})
            
        Returns:
            Generated SpineInstance
        """
        try:
            if weights is None:
                weights = np.random.normal(0, 1, self.n_components)
            
            # Validate weights
            if len(weights) != self.n_components:
                raise ValueError(f"Expected {self.n_components} weights, got {len(weights)}")
            
            # Generate global shape
            global_shape = self.mean_shape + self.global_pca.components_.T @ weights
            
            # Generate level-specific variations
            levels = {}
            for level_id, level_model in self.level_models.items():
                level_features = (level_model['mean'] + 
                                level_model['pca'].components_.T @ weights)
                
                # Apply condition constraints if specified
                if condition_constraints:
                    level_features = self._apply_condition_constraints(
                        level_features, condition_constraints
                    )
                
                levels[level_id] = self._reconstruct_level(level_features)
            
            # Generate condition states
            condition_states = condition_constraints or {
                'stenosis': 'Normal/Mild',
                'foraminal_narrowing': 'Normal/Mild',
                'subarticular_stenosis': 'Normal/Mild'
            }
            
            return SpineInstance(
                study_id='generated',
                levels=levels,
                global_shape=global_shape,
                condition_states=condition_states
            )
            
        except Exception as e:
            logger.error(f"Error generating new instance: {str(e)}")
            raise
    
    def _apply_condition_constraints(self,
                                  features: np.ndarray,
                                  constraints: Dict[str, str]) -> np.ndarray:
        """
        Apply condition constraints to generated features.
        
        Args:
            features: Generated level features
            constraints: Condition constraints to apply
            
        Returns:
            Modified features conforming to constraints
        """
        try:
            modified_features = features.copy()
            
            # Apply constraints based on condition type
            for condition, severity in constraints.items():
                if condition in self.condition_models:
                    condition_model = self.condition_models[condition]
                    severity_index = ['Normal/Mild', 'Moderate', 'Severe'].index(severity)
                    
                    # Modify relevant features based on severity
                    if condition == 'stenosis':
                        modified_features[0] *= (1.0 - 0.2 * severity_index)  # Reduce canal diameter
                    elif condition == 'foraminal_narrowing':
                        modified_features[1:3] *= (1.0 - 0.2 * severity_index)  # Reduce foraminal widths
                    elif condition == 'subarticular_stenosis':
                        modified_features[3:5] *= (1.0 - 0.2 * severity_index)  # Reduce subarticular spaces
            
            return modified_features
            
        except Exception as e:
            logger.error(f"Error applying condition constraints: {str(e)}")
            raise
    
    def _reconstruct_level(self, features: np.ndarray) -> VertebralLevel:
        """
        Reconstruct a vertebral level from features.
        
        Args:
            features: Level features to reconstruct from
            
        Returns:
            Reconstructed VertebralLevel
        """
        try:
            if len(features) < 5:
                raise ValueError(f"Invalid feature vector length: {len(features)}")
            
            return VertebralLevel(
                level_id='reconstructed',
                canal_diameter=float(features[0]),
                left_foraminal_width=float(features[1]),
                right_foraminal_width=float(features[2]),
                left_subarticular_space=float(features[3]),
                right_subarticular_space=float(features[4]),
                coordinates=features[5:].reshape(-1, 3),
                severity_grade='Unknown'
            )
            
        except Exception as e:
            logger.error(f"Error reconstructing level: {str(e)}")
            raise
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model's current state and performance.
        
        Returns:
            Dictionary containing model summary information
        """
        try:
            return {
                'model_parameters': {
                    'n_components': self.n_components,
                    'validation_split': self.validation_split,
                    'random_state': self.random_state
                },
                'training_status': {
                    'is_fitted': self.mean_shape is not None,
                    'n_level_models': len(self.level_models),
                    'n_condition_models': len(self.condition_models)
                },
                'validation_metrics': {
                    'reconstruction_error': self.validation_metrics.reconstruction_error,
                    'explained_variance_ratio': self.validation_metrics.explained_variance_ratio,
                    'compactness': self.validation_metrics.compactness,
                    'specificity': self.validation_metrics.specificity,
                    'generalization': self.validation_metrics.generalization
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating model summary: {str(e)}")
            raise