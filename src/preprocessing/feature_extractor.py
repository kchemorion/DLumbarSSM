"""
Spine Feature Extraction Pipeline

This module handles the extraction of meaningful features from spine MRI data
for use in statistical shape modeling. It processes both image data and clinical
annotations to create comprehensive feature vectors.

Author: Francis Kiptengwer Chemorion
Created: 2024-8-11
"""

import logging
import time
import cv2
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.preprocessing.image_preprocessor import SpineImagePreprocessor
from src.shape_modeling.statistical_shape_model import SpineInstance, VertebralLevel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureType(Enum):
    """Types of features that can be extracted"""
    MORPHOLOGICAL = auto()
    INTENSITY = auto()
    TEXTURE = auto()
    GEOMETRIC = auto()
    CLINICAL = auto()

@dataclass
class ExtractionConfig:
    """Configuration for feature extraction"""
    feature_types: List[FeatureType]
    window_size: int = 32
    gaussian_sigma: float = 1.0
    distance_threshold: float = 5.0
    min_confidence: float = 0.8
    use_gpu: bool = torch.cuda.is_available()

@dataclass
class FeatureQualityMetrics:
    """Metrics for assessing feature quality"""
    noise_level: float
    consistency_score: float
    coverage_ratio: float
    landmark_confidence: Dict[str, float]
    extraction_time: float

class SpineFeatureExtractor:
    """
    Extracts comprehensive features from spine MRI data for statistical shape modeling.
    
    This class handles the extraction of various types of features including
    morphological characteristics, intensity patterns, and geometric measurements.
    """
    
    def __init__(self, 
                 preprocessor: SpineImagePreprocessor,
                 config: Optional[ExtractionConfig] = None) -> None:
        """
        Initialize the feature extractor.
        
        Args:
            preprocessor: Image preprocessor instance
            config: Feature extraction configuration
        """
        self.preprocessor = preprocessor
        self.config = config or ExtractionConfig(
            feature_types=[FeatureType.MORPHOLOGICAL, 
                         FeatureType.GEOMETRIC]
        )
        
        self.device = torch.device('cuda' if self.config.use_gpu else 'cpu')
        self.quality_metrics: Dict[str, FeatureQualityMetrics] = {}
        
        logger.info(f"Initialized SpineFeatureExtractor with config: {self.config}")
    
    def extract_spine_features(self,
                             study_data: Dict[str, Any],
                             segmentations: Dict[str, np.ndarray],
                             quality_threshold: float = 0.7) -> SpineInstance:
        """
        Extract comprehensive features for a spine instance.
        
        Args:
            study_data: Dictionary containing study metadata and annotations
            segmentations: Dictionary of segmentation masks for different series
            quality_threshold: Minimum quality score for feature acceptance
            
        Returns:
            SpineInstance object containing extracted features
            
        Raises:
            ValueError: If input data is invalid or feature quality is insufficient
        """
        try:
            logger.info(f"Starting feature extraction for study {study_data['labels']['study_id']}")
            
            # Validate inputs
            self._validate_inputs(study_data, segmentations)
            
            # Extract features
            start_time = time.time()
            
            # 1. Extract global shape features
            global_shape = self._extract_global_shape(segmentations)
            
            # 2. Extract level-specific features
            levels = self._extract_level_features(study_data, segmentations)
            
            # 3. Extract condition states
            condition_states = self._extract_condition_states(study_data)
            
            # 4. Assess feature quality
            quality_metrics = self._assess_feature_quality(
                global_shape, levels, condition_states
            )
            
            # Store quality metrics
            self.quality_metrics[study_data['labels']['study_id']] = quality_metrics
            
            # Validate feature quality
            if quality_metrics.consistency_score < quality_threshold:
                logger.warning(f"Feature quality below threshold: {quality_metrics.consistency_score}")
                self._attempt_feature_refinement(global_shape, levels)
            
            extraction_time = time.time() - start_time
            logger.info(f"Feature extraction completed in {extraction_time:.2f} seconds")
            
            return SpineInstance(
                study_id=study_data['labels']['study_id'],
                levels=levels,
                global_shape=global_shape,
                condition_states=condition_states
            )
            
        except Exception as e:
            logger.error(f"Error during feature extraction: {str(e)}")
            raise
    
    def _validate_inputs(self,
                        study_data: Dict[str, Any],
                        segmentations: Dict[str, np.ndarray]) -> None:
        """Validate input data format and content"""
        required_keys = ['labels', 'coordinates', 'series']
        if not all(key in study_data for key in required_keys):
            raise ValueError(f"Missing required keys in study_data: {required_keys}")
            
        if not segmentations:
            raise ValueError("Empty segmentations dictionary")
            
        # Validate segmentation masks
        for series_id, mask in segmentations.items():
            if not isinstance(mask, np.ndarray):
                raise ValueError(f"Invalid segmentation type for series {series_id}")
            if mask.ndim != 3:
                raise ValueError(f"Invalid segmentation dimensions for series {series_id}")
    
    def _extract_global_shape(self, 
                            segmentations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract global spine shape features from segmentations.
        
        Args:
            segmentations: Dictionary of segmentation masks
            
        Returns:
            Array of global shape features
        """
        try:
            combined_features = []
            
            for series_id, seg in segmentations.items():
                # 1. Extract centerline
                centerline = self._extract_centerline(seg)
                
                # 2. Compute geometric features
                if FeatureType.GEOMETRIC in self.config.feature_types:
                    curvature = self._compute_curvature(centerline)
                    torsion = self._compute_torsion(centerline)
                    combined_features.extend([centerline, curvature, torsion])
                
                # 3. Extract morphological features
                if FeatureType.MORPHOLOGICAL in self.config.feature_types:
                    morph_features = self._extract_morphological_features(seg)
                    combined_features.append(morph_features)
                
                # 4. Extract intensity features if available
                if FeatureType.INTENSITY in self.config.feature_types:
                    intensity_features = self._extract_intensity_features(seg)
                    combined_features.append(intensity_features)
            
            # Combine and normalize features
            global_features = np.concatenate(combined_features)
            global_features = self._normalize_features(global_features)
            
            return global_features
            
        except Exception as e:
            logger.error(f"Error extracting global shape features: {str(e)}")
            raise
    
    def _extract_centerline(self, 
                          segmentation: np.ndarray,
                          smooth: bool = True) -> np.ndarray:
        """
        Extract spine centerline from segmentation mask.
        
        Args:
            segmentation: 3D segmentation mask
            smooth: Whether to apply smoothing to the centerline
            
        Returns:
            Array of centerline points
        """
        try:
            centerline = []
            
            for slice_idx in range(segmentation.shape[0]):
                mask = segmentation[slice_idx] > 0
                if np.any(mask):
                    # Find centroid using distance transform
                    coords = self._compute_weighted_centroid(mask)
                    centerline.append([slice_idx, coords[0], coords[1]])
            
            centerline = np.array(centerline)
            
            if smooth and len(centerline) > 3:
                # Apply smoothing using Gaussian filter
                centerline[:, 1:] = gaussian_filter(
                    centerline[:, 1:],
                    sigma=self.config.gaussian_sigma,
                    axis=0
                )
            
            return centerline
            
        except Exception as e:
            logger.error(f"Error extracting centerline: {str(e)}")
            raise
    
    def _compute_weighted_centroid(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute weighted centroid of a binary mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Coordinates of weighted centroid
        """
        try:
            # Compute distance transform
            dist_transform = cv2.distanceTransform(
                mask.astype(np.uint8),
                cv2.DIST_L2,
                5
            )
            
            # Weight coordinates by distance
            coords = np.where(mask)
            weights = dist_transform[coords]
            
            # Compute weighted centroid
            centroid = np.average(
                np.column_stack(coords),
                weights=weights,
                axis=0
            )
            
            return centroid
            
        except Exception as e:
            logger.error(f"Error computing weighted centroid: {str(e)}")
            raise
    
    def _compute_curvature(self, centerline: np.ndarray) -> np.ndarray:
        """
        Compute spine curvature from centerline points.
        
        Args:
            centerline: Array of centerline points
            
        Returns:
            Array of curvature values
        """
        try:
            # Compute derivatives
            dx = np.gradient(centerline[:, 1])
            dy = np.gradient(centerline[:, 2])
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            
            # Compute curvature using differential geometry formula
            curvature = np.abs(d2x * dy - dx * d2y) / (dx * dx + dy * dy)**1.5
            
            # Remove infinity values and smooth
            curvature = np.nan_to_num(curvature)
            curvature = gaussian_filter(curvature, sigma=self.config.gaussian_sigma)
            
            return curvature
            
        except Exception as e:
            logger.error(f"Error computing curvature: {str(e)}")
            raise

    def _compute_torsion(self, centerline: np.ndarray) -> np.ndarray:
        """
        Compute spine torsion from centerline points.
        
        Args:
            centerline: Array of centerline points
            
        Returns:
            Array of torsion values
        """
        try:
            # Compute derivatives up to third order
            dx = np.gradient(centerline[:, 1])
            dy = np.gradient(centerline[:, 2])
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            d3x = np.gradient(d2x)
            d3y = np.gradient(d2y)
            
            # Compute torsion
            numerator = (d3y * dx - d3x * dy)
            denominator = ((dx * dx + dy * dy) * 
                         np.sqrt(d2x * d2x + d2y * d2y))
            
            torsion = numerator / denominator
            
            # Clean up and smooth
            torsion = np.nan_to_num(torsion)
            torsion = gaussian_filter(torsion, sigma=self.config.gaussian_sigma)
            
            return torsion
            
        except Exception as e:
            logger.error(f"Error computing torsion: {str(e)}")
            raise
    
    def _extract_morphological_features(self, 
                                     segmentation: np.ndarray) -> np.ndarray:
        """
        Extract morphological features from segmentation mask.
        
        Args:
            segmentation: 3D segmentation mask
            
        Returns:
            Array of morphological features
        """
        try:
            features = []
            
            for slice_idx in range(segmentation.shape[0]):
                mask = segmentation[slice_idx] > 0
                if np.any(mask):
                    # Compute area and perimeter
                    area = np.sum(mask)
                    perimeter = self._compute_perimeter(mask)
                    
                    # Compute shape descriptors
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Compute oriented bounding box
                    bbox_features = self._compute_bbox_features(mask)
                    
                    features.append([
                        area, perimeter, circularity, *bbox_features
                    ])
            
            return np.mean(features, axis=0)
            
        except Exception as e:
            logger.error(f"Error extracting morphological features: {str(e)}")
            raise
    
    def _extract_level_features(self,
                              study_data: Dict[str, Any],
                              segmentations: Dict[str, np.ndarray]
                              ) -> Dict[str, VertebralLevel]:
        """
        Extract features for each vertebral level.
        
        Args:
            study_data: Study metadata and annotations
            segmentations: Segmentation masks
            
        Returns:
            Dictionary of VertebralLevel objects
        """
        try:
            levels = {}
            level_ids = ['L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1']
            
            for level_id in level_ids:
                # Get coordinates for this level
                coords = self._get_level_coordinates(study_data, level_id)
                
                # Extract measurements
                canal_diameter = self._measure_canal_diameter(
                    segmentations, coords
                )
                left_width = self._measure_foraminal_width(
                    segmentations, coords, 'left'
                )
                right_width = self._measure_foraminal_width(
                    segmentations, coords, 'right'
                )
                left_space = self._measure_subarticular_space(
                    segmentations, coords, 'left'
                )
                right_space = self._measure_subarticular_space(
                    segmentations, coords, 'right'
                )
                
                # Get severity grade
                severity = study_data['labels'][
                    f'spinal_canal_stenosis_{level_id.lower()}'
                ]
                
                # Create VertebralLevel instance
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
            
        except Exception as e:
            logger.error(f"Error extracting level features: {str(e)}")
            raise

    def _assess_feature_quality(self,
                              global_shape: np.ndarray,
                              levels: Dict[str, VertebralLevel],
        condition_states: Dict[str, str]) -> FeatureQualityMetrics:
                """
                Assess the quality of extracted features.
                
                Args:
                    global_shape: Global shape features
                    levels: Dictionary of vertebral level features
                    condition_states: Dictionary of condition states
                    
                Returns:
                    FeatureQualityMetrics object containing quality assessments
                """
                try:
                    # 1. Assess noise level
                    noise_level = self._compute_noise_level(global_shape)
                    
                    # 2. Assess feature consistency
                    consistency_score = self._compute_consistency_score(levels)
                    
                    # 3. Assess anatomical coverage
                    coverage_ratio = self._compute_coverage_ratio(levels)
                    
                    # 4. Assess landmark confidence
                    landmark_confidence = self._compute_landmark_confidence(levels)
                    
                    # Create quality metrics object
                    metrics = FeatureQualityMetrics(
                        noise_level=noise_level,
                        consistency_score=consistency_score,
                        coverage_ratio=coverage_ratio,
                        landmark_confidence=landmark_confidence,
                        extraction_time=time.time()
                    )
                    
                    logger.info(f"Feature quality metrics computed: {metrics}")
                    return metrics
                    
                except Exception as e:
                    logger.error(f"Error assessing feature quality: {str(e)}")
                    raise
    
    def _compute_noise_level(self, features: np.ndarray) -> float:
        """
        Compute noise level in extracted features.
        
        Args:
            features: Extracted feature array
            
        Returns:
            Estimated noise level (0-1 scale)
        """
        try:
            # Compute local variations
            diff = np.diff(features)
            local_var = np.var(diff)
            
            # Normalize to 0-1 scale
            noise_level = np.clip(local_var / (np.var(features) + 1e-6), 0, 1)
            
            return float(noise_level)
            
        except Exception as e:
            logger.error(f"Error computing noise level: {str(e)}")
            raise
    
    def _compute_consistency_score(self, 
                                 levels: Dict[str, VertebralLevel]) -> float:
        """
        Compute consistency score for level features.
        
        Args:
            levels: Dictionary of vertebral level features
            
        Returns:
            Consistency score (0-1 scale)
        """
        try:
            inconsistencies = []
            
            # Check anatomical constraints
            for level_id, level in levels.items():
                # 1. Check bilateral symmetry
                symmetry_score = self._check_bilateral_symmetry(level)
                
                # 2. Check anatomical proportions
                proportion_score = self._check_anatomical_proportions(level)
                
                # 3. Check measurement ranges
                range_score = self._check_measurement_ranges(level)
                
                inconsistencies.extend([
                    1 - symmetry_score,
                    1 - proportion_score,
                    1 - range_score
                ])
            
            # Compute overall consistency
            consistency_score = 1 - np.mean(inconsistencies)
            
            return float(consistency_score)
            
        except Exception as e:
            logger.error(f"Error computing consistency score: {str(e)}")
            raise
    
    def _check_bilateral_symmetry(self, level: VertebralLevel) -> float:
        """Check bilateral symmetry of measurements"""
        try:
            # Compare left and right measurements
            foraminal_diff = abs(level.left_foraminal_width - 
                               level.right_foraminal_width)
            subarticular_diff = abs(level.left_subarticular_space - 
                                  level.right_subarticular_space)
            
            # Normalize differences
            max_foraminal = max(level.left_foraminal_width, 
                              level.right_foraminal_width)
            max_subarticular = max(level.left_subarticular_space, 
                                 level.right_subarticular_space)
            
            foraminal_score = 1 - (foraminal_diff / (max_foraminal + 1e-6))
            subarticular_score = 1 - (subarticular_diff / (max_subarticular + 1e-6))
            
            return np.mean([foraminal_score, subarticular_score])
            
        except Exception as e:
            logger.error(f"Error checking bilateral symmetry: {str(e)}")
            raise
    
    def _check_anatomical_proportions(self, level: VertebralLevel) -> float:
        """Check if measurements follow expected anatomical proportions"""
        try:
            # Define expected proportions (based on literature)
            expected_ratios = {
                'canal_to_foraminal': (1.5, 2.5),  # Expected range
                'foraminal_to_subarticular': (1.2, 2.0)
            }
            
            # Compute actual ratios
            actual_ratios = {
                'canal_to_foraminal': level.canal_diameter / np.mean(
                    [level.left_foraminal_width, level.right_foraminal_width]
                ),
                'foraminal_to_subarticular': np.mean(
                    [level.left_foraminal_width, level.right_foraminal_width]
                ) / np.mean(
                    [level.left_subarticular_space, level.right_subarticular_space]
                )
            }
            
            # Compute proportion scores
            scores = []
            for ratio_name, (min_val, max_val) in expected_ratios.items():
                actual = actual_ratios[ratio_name]
                if actual < min_val:
                    score = actual / min_val
                elif actual > max_val:
                    score = max_val / actual
                else:
                    score = 1.0
                scores.append(score)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"Error checking anatomical proportions: {str(e)}")
            raise
    
    def _check_measurement_ranges(self, level: VertebralLevel) -> float:
        """Check if measurements fall within expected ranges"""
        try:
            # Define expected ranges (in mm)
            expected_ranges = {
                'canal_diameter': (10, 20),
                'foraminal_width': (5, 15),
                'subarticular_space': (3, 8)
            }
            
            scores = []
            
            # Check canal diameter
            scores.append(
                self._compute_range_score(
                    level.canal_diameter,
                    *expected_ranges['canal_diameter']
                )
            )
            
            # Check foraminal widths
            for width in [level.left_foraminal_width, level.right_foraminal_width]:
                scores.append(
                    self._compute_range_score(
                        width,
                        *expected_ranges['foraminal_width']
                    )
                )
            
            # Check subarticular spaces
            for space in [level.left_subarticular_space, level.right_subarticular_space]:
                scores.append(
                    self._compute_range_score(
                        space,
                        *expected_ranges['subarticular_space']
                    )
                )
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"Error checking measurement ranges: {str(e)}")
            raise
    
    def _compute_range_score(self, 
                           value: float, 
                           min_val: float, 
                           max_val: float) -> float:
        """Compute score for value within expected range"""
        if value < min_val:
            return value / min_val
        elif value > max_val:
            return max_val / value
        return 1.0
    
    def _compute_coverage_ratio(self, levels: Dict[str, VertebralLevel]) -> float:
        """
        Compute ratio of successfully measured features.
        
        Args:
            levels: Dictionary of vertebral level features
            
        Returns:
            Coverage ratio (0-1 scale)
        """
        try:
            expected_measurements = len(levels) * 5  # 5 measurements per level
            actual_measurements = sum(
                np.isfinite([
                    level.canal_diameter,
                    level.left_foraminal_width,
                    level.right_foraminal_width,
                    level.left_subarticular_space,
                    level.right_subarticular_space
                ]).sum()
                for level in levels.values()
            )
            
            return actual_measurements / expected_measurements
            
        except Exception as e:
            logger.error(f"Error computing coverage ratio: {str(e)}")
            raise
    
    def _compute_landmark_confidence(self, 
                                  levels: Dict[str, VertebralLevel]
                                  ) -> Dict[str, float]:
        """
        Compute confidence scores for anatomical landmarks.
        
        Args:
            levels: Dictionary of vertebral level features
            
        Returns:
            Dictionary of landmark confidence scores
        """
        try:
            confidence_scores = {}
            
            for level_id, level in levels.items():
                # Check landmark stability
                stability = self._compute_landmark_stability(level.coordinates)
                
                # Check anatomical consistency
                consistency = self._check_landmark_consistency(level)
                
                # Combine scores
                confidence_scores[level_id] = np.mean([stability, consistency])
            
            return confidence_scores
            
        except Exception as e:
            logger.error(f"Error computing landmark confidence: {str(e)}")
            raise
    
    def _attempt_feature_refinement(self,
                                  global_shape: np.ndarray,
                                  levels: Dict[str, VertebralLevel]) -> None:
        """
        Attempt to refine low-quality features.
        
        Args:
            global_shape: Global shape features to refine
            levels: Dictionary of vertebral level features to refine
        """
        try:
            logger.info("Attempting feature refinement")
            
            # 1. Apply smoothing to global shape features
            if self._compute_noise_level(global_shape) > 0.3:
                global_shape = gaussian_filter(global_shape, 
                                            sigma=self.config.gaussian_sigma)
            
            # 2. Adjust inconsistent measurements
            for level_id, level in levels.items():
                self._refine_level_measurements(level)
            
            logger.info("Feature refinement completed")
            
        except Exception as e:
            logger.error(f"Error during feature refinement: {str(e)}")
            raise
    
    def _refine_level_measurements(self, level: VertebralLevel) -> None:
        """Refine measurements for a single vertebral level"""
        try:
            # 1. Enforce bilateral symmetry constraints
            mean_foraminal = np.mean([
                level.left_foraminal_width,
                level.right_foraminal_width
            ])
            mean_subarticular = np.mean([
                level.left_subarticular_space,
                level.right_subarticular_space
            ])
            
            # Adjust measurements towards mean if difference is too large
            max_asymmetry = 0.2  # 20% maximum allowed asymmetry
            
            for attr in ['foraminal_width', 'subarticular_space']:
                left_val = getattr(level, f'left_{attr}')
                right_val = getattr(level, f'right_{attr}')
                mean_val = np.mean([left_val, right_val])
                
                if abs(left_val - right_val) / mean_val > max_asymmetry:
                    setattr(level, f'left_{attr}', 
                           mean_val * (1 + max_asymmetry/2))
                    setattr(level, f'right_{attr}', 
                           mean_val * (1 - max_asymmetry/2))
            
            logger.debug(f"Refined measurements for level {level.level_id}")
            
        except Exception as e:
            logger.error(f"Error refining level measurements: {str(e)}")
            raise