import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import torch
import torch.nn.functional as F
from ..shape_modeling.statistical_shape_model import SpineInstance, VertebralLevel

class SpineFeatureExtractor:
    """Extracts features from spine MRI data for statistical shape modeling"""
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        
    def extract_spine_features(self, 
                             study_data: Dict,
                             segmentations: Dict) -> SpineInstance:
        """Extract comprehensive features for a spine instance"""
        # Extract global shape features
        global_shape = self._extract_global_shape(segmentations)
        
        # Extract level-specific features
        levels = self._extract_level_features(study_data, segmentations)
        
        # Extract condition states
        condition_states = self._extract_condition_states(study_data)
        
        return SpineInstance(
            study_id=study_data['labels']['study_id'],
            levels=levels,
            global_shape=global_shape,
            condition_states=condition_states
        )
    
    def _extract_global_shape(self, segmentations: Dict) -> np.ndarray:
        """Extract global spine shape features"""
        # Combine segmentations from different views
        combined_features = []
        
        for series_id, seg in segmentations.items():
            # Extract centerline
            centerline = self._extract_centerline(seg)
            # Extract curvature
            curvature = self._compute_curvature(centerline)
            combined_features.extend([centerline, curvature])
            
        return np.concatenate(combined_features)
    
    def _extract_level_features(self, 
                              study_data: Dict,
                              segmentations: Dict) -> Dict[str, VertebralLevel]:
        """Extract features for each vertebral level"""
        levels = {}
        level_ids = ['L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1']
        
        for level_id in level_ids:
            # Get coordinates for this level
            coords = self._get_level_coordinates(study_data, level_id)
            
            # Measure canal diameter
            canal_diameter = self._measure_canal_diameter(segmentations, coords)
            
            # Measure foraminal widths
            left_width = self._measure_foraminal_width(segmentations, coords, 'left')
            right_width = self._measure_foraminal_width(segmentations, coords, 'right')
            
            # Measure subarticular spaces
            left_space = self._measure_subarticular_space(segmentations, coords, 'left')
            right_space = self._measure_subarticular_space(segmentations, coords, 'right')
            
            # Get severity grade
            severity = study_data['labels'][f'spinal_canal_stenosis_{level_id.lower()}']
            
            levels[level_id] = VertebralLevel(
                level_id=level_id,
                canal_diameter=canal_diameter,
                left_foraminal_width=left_width,
                right_foraminal_width=right_width,
                left_subarticular_space=left_space,
                right_subarticular_space=right_space,
                coordinates=coords,
                severity_grade=severity
            )
            
        return levels
    
    def _extract_condition_states(self, study_data: Dict) -> Dict[str, str]:
        """Extract overall condition states"""
        conditions = {}
        
        # Extract stenosis states
        stenosis_cols = [col for col in study_data['labels'] 
                        if 'stenosis' in col]
        conditions['stenosis'] = max(study_data['labels'][col] 
                                   for col in stenosis_cols)
        
        # Extract foraminal narrowing states
        foraminal_cols = [col for col in study_data['labels'] 
                         if 'foraminal' in col]
        conditions['foraminal_narrowing'] = max(study_data['labels'][col] 
                                              for col in foraminal_cols)
        
        # Extract subarticular stenosis states
        subarticular_cols = [col for col in study_data['labels'] 
                            if 'subarticular' in col]
        conditions['subarticular_stenosis'] = max(study_data['labels'][col] 
                                                for col in subarticular_cols)
        
        return conditions
    
    def _extract_centerline(self, segmentation: np.ndarray) -> np.ndarray:
        """Extract spine centerline from segmentation"""
        centerline = []
        for slice_idx in range(segmentation.shape[0]):
            mask = segmentation[slice_idx] > 0
            if np.any(mask):
                # Find centroid
                coords = np.where(mask)
                centroid = np.mean(coords, axis=1)
                centerline.append([slice_idx, centroid[0], centroid[1]])
        return np.array(centerline)
    
    def _compute_curvature(self, centerline: np.ndarray) -> np.ndarray:
        """Compute spine curvature from centerline"""
        # Compute finite differences
        dx = np.gradient(centerline[:, 1])
        dy = np.gradient(centerline[:, 2])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        curvature = np.abs(d2x * dy - dx * d2y) / (dx * dx + dy * dy)**1.5
        return curvature
    
    def _get_level_coordinates(self, study_data: Dict, level_id: str) -> np.ndarray:
        """Get coordinates for a specific vertebral level"""
        level_coords = study_data['coordinates'][
            study_data['coordinates']['level'] == level_id]
        return np.array([level_coords['x'], level_coords['y']]).T
    
    def _measure_canal_diameter(self, 
                              segmentations: Dict,
                              coords: np.ndarray) -> float:
        """Measure spinal canal diameter"""
        # Implementation depends on segmentation format
        # This is a placeholder that should be implemented based on actual data
        return np.mean([10.0])  # Example return
    
    def _measure_foraminal_width(self,
                               segmentations: Dict,
                               coords: np.ndarray,
                               side: str) -> float:
        """Measure foraminal width"""
        # Implementation depends on segmentation format
        # This is a placeholder that should be implemented based on actual data
        return np.mean([5.0])  # Example return
    
    def _measure_subarticular_space(self,
                                  segmentations: Dict,
                                  coords: np.ndarray,
                                  side: str) -> float:
        """Measure subarticular space"""
        # Implementation depends on segmentation format
        # This is a placeholder that should be implemented based on actual data
        return np.mean([3.0])  # Example return