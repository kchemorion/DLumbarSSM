import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn.functional as F

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

@dataclass
class SpineInstance:
    """Represents a complete spine instance with all levels and conditions"""
    study_id: str
    levels: Dict[str, VertebralLevel]
    global_shape: np.ndarray  # Global spine shape parameters
    condition_states: Dict[str, str]  # Overall condition classifications

class HierarchicalSpineModel:
    """
    Hierarchical Statistical Shape Model for lumbar spine analysis.
    Handles both global spine shape and level-specific variations.
    """
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
        self.global_pca = PCA(n_components=n_components)
        self.level_models = {}
        self.condition_models = {}
        self.mean_shape = None
        self.eigenvectors = None
        self.eigenvalues = None
        
    def fit(self, spine_instances: List[SpineInstance]):
        """Fit the hierarchical model to a set of spine instances"""
        # 1. Extract global shapes
        global_shapes = np.stack([inst.global_shape for inst in spine_instances])
        
        # 2. Align shapes using Procrustes analysis
        aligned_shapes = self._align_shapes(global_shapes)
        
        # 3. Compute global shape statistics
        self.mean_shape = np.mean(aligned_shapes, axis=0)
        self.global_pca.fit(aligned_shapes)
        
        # 4. Build level-specific models
        self._build_level_models(spine_instances)
        
        # 5. Build condition-specific variation models
        self._build_condition_models(spine_instances)
        
    def _align_shapes(self, shapes: np.ndarray) -> np.ndarray:
        """Align shapes using Generalized Procrustes Analysis"""
        n_shapes = shapes.shape[0]
        aligned = shapes.copy()
        
        # Initialize reference as first shape
        reference = aligned[0]
        max_iterations = 100
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
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
                break
                
        return aligned
    
    def _build_level_models(self, instances: List[SpineInstance]):
        """Build statistical models for each vertebral level"""
        levels = ['L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1']
        
        for level in levels:
            level_data = []
            for inst in instances:
                if level in inst.levels:
                    level_data.append(self._extract_level_features(inst.levels[level]))
            
            if level_data:
                level_data = np.stack(level_data)
                self.level_models[level] = {
                    'pca': PCA(n_components=self.n_components).fit(level_data),
                    'mean': np.mean(level_data, axis=0),
                    'std': np.std(level_data, axis=0)
                }
    
    def _build_condition_models(self, instances: List[SpineInstance]):
        """Build condition-specific variation models"""
        conditions = ['stenosis', 'foraminal_narrowing', 'subarticular_stenosis']
        
        for condition in conditions:
            condition_data = []
            for inst in instances:
                if condition in inst.condition_states:
                    features = self._extract_condition_features(inst, condition)
                    condition_data.append(features)
            
            if condition_data:
                condition_data = np.stack(condition_data)
                self.condition_models[condition] = {
                    'pca': PCA(n_components=self.n_components).fit(condition_data),
                    'mean': np.mean(condition_data, axis=0),
                    'std': np.std(condition_data, axis=0)
                }
    
    def _extract_level_features(self, level: VertebralLevel) -> np.ndarray:
        """Extract features for a vertebral level"""
        features = np.concatenate([
            [level.canal_diameter],
            [level.left_foraminal_width],
            [level.right_foraminal_width],
            [level.left_subarticular_space],
            [level.right_subarticular_space],
            level.coordinates.flatten()
        ])
        return features
    
    def _extract_condition_features(self, 
                                  instance: SpineInstance, 
                                  condition: str) -> np.ndarray:
        """Extract condition-specific features"""
        features = []
        for level in instance.levels.values():
            if condition == 'stenosis':
                features.append(level.canal_diameter)
            elif condition == 'foraminal_narrowing':
                features.extend([level.left_foraminal_width, 
                               level.right_foraminal_width])
            elif condition == 'subarticular_stenosis':
                features.extend([level.left_subarticular_space, 
                               level.right_subarticular_space])
        return np.array(features)
    
    def generate_new_instance(self, 
                            weights: Optional[np.ndarray] = None) -> SpineInstance:
        """Generate a new spine instance using the statistical model"""
        if weights is None:
            weights = np.random.normal(0, 1, self.n_components)
            
        # Generate global shape
        global_shape = self.mean_shape + self.global_pca.components_.T @ weights
        
        # Generate level-specific variations
        levels = {}
        for level_id, level_model in self.level_models.items():
            level_features = (level_model['mean'] + 
                            level_model['pca'].components_.T @ weights)
            levels[level_id] = self._reconstruct_level(level_features)
            
        return SpineInstance(
            study_id='generated',
            levels=levels,
            global_shape=global_shape,
            condition_states={}
        )
    
    def _reconstruct_level(self, features: np.ndarray) -> VertebralLevel:
        """Reconstruct a vertebral level from features"""
        return VertebralLevel(
            level_id='reconstructed',
            canal_diameter=features[0],
            left_foraminal_width=features[1],
            right_foraminal_width=features[2],
            left_subarticular_space=features[3],
            right_subarticular_space=features[4],
            coordinates=features[5:].reshape(-1, 3),  # Assuming 3D coordinates
            severity_grade='Unknown'
        )