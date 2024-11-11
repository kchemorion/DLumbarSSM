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
from scipy.ndimage import gaussian_filter1d
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
        try:
            required_keys = ['labels', 'coordinates', 'series']
            if not all(key in study_data for key in required_keys):
                raise ValueError(f"Missing required keys in study_data: {required_keys}")
                
            if not segmentations:
                raise ValueError("Empty segmentations dictionary")
                
            # Validate segmentation masks
            for series_id, mask in segmentations.items():
                if not isinstance(mask, np.ndarray):
                    raise ValueError(f"Invalid segmentation type for series {series_id}")
                
                # Check dimensions (D, C, H, W)
                if mask.ndim != 4:
                    raise ValueError(
                        f"Invalid segmentation dimensions for series {series_id}. "
                        f"Expected 4D array (D, C, H, W), got shape {mask.shape}"
                    )
                    
                # Check that dimensions make sense
                D, C, H, W = mask.shape
                if D < 1 or C < 1 or H < 16 or W < 16:
                    raise ValueError(
                        f"Invalid segmentation dimensions for series {series_id}: {mask.shape}"
                    )
                    
                # Check value range
                if not (0 <= mask.min() <= mask.max() <= 1):
                    raise ValueError(
                        f"Invalid segmentation values for series {series_id}. "
                        f"Expected range [0,1], got [{mask.min()}, {mask.max()}]"
                    )
                    
        except Exception as e:
            logger.error(f"Error validating inputs: {str(e)}")
            raise
    
    def _extract_global_shape(self, 
                        segmentations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Extract global spine shape features from segmentations.
        """
        try:
            combined_features = []
            processed_series = 0
            
            for series_id, seg in segmentations.items():
                logger.info(f"Processing series {series_id} with shape {seg.shape}")
                series_features = []
                
                # 1. Extract centerline
                centerline = self._extract_centerline(seg)
                if len(centerline) < 3:
                    logger.warning(f"Insufficient centerline points for series {series_id}")
                    continue
                    
                # 2. Compute geometric features
                if FeatureType.GEOMETRIC in self.config.feature_types:
                    try:
                        curvature = self._compute_curvature(centerline)
                        torsion = self._compute_torsion(centerline)
                        
                        # Normalize and flatten geometric features
                        geometric_features = np.concatenate([
                            centerline.flatten(),
                            curvature.flatten(),
                            torsion.flatten()
                        ])
                        series_features.append(geometric_features)
                        logger.debug(f"Added geometric features of length {len(geometric_features)}")
                    except Exception as e:
                        logger.warning(f"Failed to compute geometric features: {e}")
                
                # 3. Extract morphological features
                if FeatureType.MORPHOLOGICAL in self.config.feature_types:
                    try:
                        morph_features = self._extract_morphological_features(seg)
                        if morph_features is not None and len(morph_features) > 0:
                            series_features.append(morph_features.flatten())
                            logger.debug(f"Added morphological features of length {len(morph_features)}")
                    except Exception as e:
                        logger.warning(f"Failed to compute morphological features: {e}")
                
                # Only add features if we got valid ones
                if series_features:
                    try:
                        # Combine features for this series
                        series_combined = np.concatenate(series_features)
                        combined_features.append(series_combined)
                        processed_series += 1
                        logger.info(f"Successfully processed series {series_id}")
                    except Exception as e:
                        logger.warning(f"Failed to combine features for series {series_id}: {e}")
            
            if processed_series == 0:
                raise ValueError(f"No valid features extracted from any of the {len(segmentations)} series")
            
            # Combine features from all series
            all_features = np.concatenate(combined_features)
            
            # Normalize combined features
            normalized_features = self._normalize_features(all_features)
            
            logger.info(f"Successfully extracted features from {processed_series} series")
            logger.debug(f"Final feature vector shape: {normalized_features.shape}")
            
            return normalized_features
            
        except Exception as e:
            logger.error(f"Error extracting global shape features: {str(e)}")
            raise

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vector to zero mean and unit variance"""
        try:
            mean = np.mean(features)
            std = np.std(features)
            
            if std < 1e-10:  # Avoid division by very small numbers
                logger.warning("Very small standard deviation in features")
                return features - mean
                
            normalized = (features - mean) / std
            
            # Check for invalid values
            if np.any(~np.isfinite(normalized)):
                logger.warning("Invalid values after normalization")
                # Replace invalid values with 0
                normalized[~np.isfinite(normalized)] = 0
                
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return features
    
    def _extract_centerline(self, 
                        segmentation: np.ndarray,
                        smooth: bool = True) -> np.ndarray:
        """
        Extract spine centerline from segmentation mask.
        
        Args:
            segmentation: 3D segmentation mask of shape (D, C, H, W) or (D, H, W)
            smooth: Whether to apply smoothing to the centerline
        """
        try:
            centerline = []
            
            # If segmentation is (D, C, H, W), take maximum probability class
            if segmentation.ndim == 4:
                mask_3d = np.argmax(segmentation, axis=1)
            else:
                mask_3d = segmentation
                
            logger.debug(f"Processing mask of shape {mask_3d.shape}")
            
            for slice_idx in range(mask_3d.shape[0]):
                # Convert to binary mask
                mask = (mask_3d[slice_idx] > 0).astype(np.uint8)
                
                # Check if mask is empty
                if not np.any(mask):
                    logger.debug(f"Empty mask in slice {slice_idx}")
                    continue
                    
                # Compute area to filter out noise
                area = np.sum(mask)
                if area < 100:  # Minimum area threshold
                    logger.debug(f"Mask too small in slice {slice_idx}: {area} pixels")
                    continue
                
                try:
                    coords = self._compute_weighted_centroid(mask)
                    if coords is not None:
                        centerline.append([slice_idx, coords[0], coords[1]])
                        logger.debug(f"Added centerline point at slice {slice_idx}")
                except Exception as e:
                    logger.debug(f"Failed to compute centroid for slice {slice_idx}: {e}")
                    continue
            
            if len(centerline) < 3:
                logger.warning(f"Not enough valid centerline points: {len(centerline)}")
                return np.array([])
                
            centerline = np.array(centerline)
            
            if smooth:
                try:
                    # Smooth each coordinate separately
                    for i in range(1, 3):  # Only smooth y and x coordinates
                        centerline[:, i] = gaussian_filter1d(centerline[:, i], 
                                                        sigma=self.config.gaussian_sigma)
                except Exception as e:
                    logger.warning(f"Failed to smooth centerline: {e}")
            
            logger.info(f"Successfully extracted centerline with {len(centerline)} points")
            return centerline
            
        except Exception as e:
            logger.error(f"Error in centerline extraction: {str(e)}")
            return np.array([])
    
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
            # Add small epsilon to avoid division by zero
            eps = 1e-6
            
            # Compute derivatives
            dx = np.gradient(centerline[:, 1]) + eps
            dy = np.gradient(centerline[:, 2]) + eps
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            
            # Compute curvature using differential geometry formula
            curvature = np.abs(d2x * dy - dx * d2y) / np.power(dx * dx + dy * dy + eps, 1.5)
            
            # Remove infinity values and smooth
            curvature = np.nan_to_num(curvature)
            curvature = gaussian_filter1d(curvature, sigma=self.config.gaussian_sigma)
            
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
            # Add small epsilon to avoid division by zero
            eps = 1e-6
            
            # Compute derivatives up to third order
            dx = np.gradient(centerline[:, 1]) + eps
            dy = np.gradient(centerline[:, 2]) + eps
            d2x = np.gradient(dx)
            d2y = np.gradient(dy)
            d3x = np.gradient(d2x)
            d3y = np.gradient(d2y)
            
            # Compute torsion
            numerator = (d3y * dx - d3x * dy)
            denominator = ((dx * dx + dy * dy + eps) * 
                        np.sqrt(d2x * d2x + d2y * d2y + eps))
            
            torsion = numerator / denominator
            
            # Clean up and smooth
            torsion = np.nan_to_num(torsion)
            torsion = gaussian_filter1d(torsion, sigma=self.config.gaussian_sigma)
            
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
            
            # If segmentation is (D, C, H, W), take argmax along channel dimension
            if segmentation.ndim == 4:
                mask_3d = np.argmax(segmentation, axis=1)
            else:
                mask_3d = segmentation
            
            for slice_idx in range(mask_3d.shape[0]):
                # Convert to binary mask
                mask = (mask_3d[slice_idx] > 0).astype(np.uint8)
                
                if np.any(mask):
                    # Compute basic shape descriptors
                    area = np.sum(mask)
                    perimeter = self._compute_perimeter(mask)
                    
                    # Compute shape descriptors
                    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
                    
                    # Compute oriented bounding box features
                    bbox_features = self._compute_bbox_features(mask)
                    
                    features.append([
                        area, perimeter, circularity, *bbox_features
                    ])
            
            if not features:
                raise ValueError("No valid features extracted from any slice")
                
            # Take mean across slices
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

    def _get_level_coordinates(self, study_data: Dict[str, Any], level_id: str) -> np.ndarray:
        """
        Get coordinates for a specific vertebral level from the study data.
        
        Args:
            study_data: Dictionary containing study information and annotations
            level_id: Identifier for the vertebral level (e.g., 'L1_L2')
            
        Returns:
            Array of coordinates for the specified level
        """
        try:
            # Get coordinates for this level from the study data
            level_coords = study_data['coordinates']
            
            # Filter coordinates for the specific level
            level_mask = level_coords['level'] == level_id
            if not any(level_mask):
                logger.warning(f"No coordinates found for level {level_id}")
                return np.zeros((0, 3))  # Return empty array if no coordinates found
                
            # Extract x, y coordinates and combine with slice information
            coords = np.array([
                level_coords.loc[level_mask, 'x'].values,
                level_coords.loc[level_mask, 'y'].values,
                level_coords.loc[level_mask, 'instance_number'].values
            ]).T
            
            logger.debug(f"Extracted {len(coords)} coordinates for level {level_id}")
            
            # Sort by instance number
            coords = coords[np.argsort(coords[:, 2])]
            
            return coords
        
        except Exception as e:
            logger.error(f"Error getting coordinates for level {level_id}: {str(e)}")
            raise

    def _extract_level_features(self,
                         study_data: Dict[str, Any],
                         segmentations: Dict[str, np.ndarray]
                         ) -> Dict[str, 'VertebralLevel']:
        """
        Extract features for each vertebral level.
        
        Args:
            study_data: Study metadata and annotations
            segmentations: Dictionary of segmentation masks
            
        Returns:
            Dictionary of VertebralLevel objects
        """
        try:
            levels = {}
            level_ids = ['L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1']
            
            for level_id in level_ids:
                logger.debug(f"Processing level {level_id}")
                
                # Get coordinates for this level
                coords = self._get_level_coordinates(study_data, level_id)
                
                if len(coords) == 0:
                    logger.warning(f"Skipping level {level_id} due to missing coordinates")
                    continue
                    
                # Extract measurements
                measurements = {}
                for series_id, seg in segmentations.items():
                    series_measurements = self._measure_level_features(
                        seg, coords, level_id
                    )
                    # Update measurements with better values if available
                    measurements.update(series_measurements)
                
                # Get severity grade from labels
                severity_key = f'spinal_canal_stenosis_{level_id.lower()}'
                severity = study_data['labels'].get(severity_key, 'Unknown')
                
                # Create VertebralLevel instance
                levels[level_id] = VertebralLevel(
                    level_id=level_id,
                    canal_diameter=measurements.get('canal_diameter', 0.0),
                    left_foraminal_width=measurements.get('left_foraminal_width', 0.0),
                    right_foraminal_width=measurements.get('right_foraminal_width', 0.0),
                    left_subarticular_space=measurements.get('left_subarticular_space', 0.0),
                    right_subarticular_space=measurements.get('right_subarticular_space', 0.0),
                    coordinates=coords,
                    severity_grade=severity
                )
                
                logger.debug(f"Completed feature extraction for level {level_id}")
            
            return levels
            
        except Exception as e:
            logger.error(f"Error extracting level features: {str(e)}")
            raise

    def _measure_level_features(self,
                         segmentation: np.ndarray,
                         coords: np.ndarray,
                         level_id: str) -> Dict[str, float]:
        """
        Measure features for a specific vertebral level from segmentation.
        
        Args:
            segmentation: Segmentation mask
            coords: Coordinates for the level
            level_id: Identifier for the vertebral level
            
        Returns:
            Dictionary of measurements
        """
        try:
            # Find the relevant slices using instance numbers
            instance_numbers = coords[:, 2]
            min_instance = int(np.min(instance_numbers))
            max_instance = int(np.max(instance_numbers))
            
            # Extract relevant slices from segmentation
            relevant_slices = segmentation[min_instance:max_instance+1]
            
            # Compute measurements
            measurements = {
                'canal_diameter': self._measure_canal_diameter(relevant_slices, coords),
                'left_foraminal_width': self._measure_foraminal_width(relevant_slices, coords, 'left'),
                'right_foraminal_width': self._measure_foraminal_width(relevant_slices, coords, 'right'),
                'left_subarticular_space': self._measure_subarticular_space(relevant_slices, coords, 'left'),
                'right_subarticular_space': self._measure_subarticular_space(relevant_slices, coords, 'right')
            }
            
            return measurements
            
        except Exception as e:
            logger.error(f"Error measuring features for level {level_id}: {str(e)}")
            return {}
    def _extract_condition_states(self, study_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract overall condition states from study data.
        
        Args:
            study_data: Dictionary containing study metadata and labels
            
        Returns:
            Dictionary mapping condition types to their severities
        """
        try:
            labels = study_data['labels']
            conditions = {}
            
            # Extract stenosis states
            stenosis_cols = [col for col in labels if 'stenosis' in col 
                            and not ('foraminal' in col or 'subarticular' in col)]
            if stenosis_cols:
                conditions['stenosis'] = max(str(labels[col]) for col in stenosis_cols)
                
            # Extract foraminal narrowing states
            foraminal_cols = [col for col in labels if 'foraminal' in col]
            if foraminal_cols:
                conditions['foraminal_narrowing'] = max(str(labels[col]) for col in foraminal_cols)
                
            # Extract subarticular stenosis states
            subarticular_cols = [col for col in labels if 'subarticular' in col]
            if subarticular_cols:
                conditions['subarticular_stenosis'] = max(str(labels[col]) for col in subarticular_cols)
                
            # Validate condition states
            valid_states = {'Normal/Mild', 'Moderate', 'Severe'}
            for condition, state in conditions.items():
                if state not in valid_states:
                    logger.warning(f"Invalid state '{state}' for condition '{condition}', "
                                f"defaulting to 'Normal/Mild'")
                    conditions[condition] = 'Normal/Mild'
            
            # Ensure all condition types are present
            required_conditions = {
                'stenosis', 
                'foraminal_narrowing', 
                'subarticular_stenosis'
            }
            
            for condition in required_conditions:
                if condition not in conditions:
                    logger.warning(f"Missing condition '{condition}', "
                                f"defaulting to 'Normal/Mild'")
                    conditions[condition] = 'Normal/Mild'
            
            logger.debug(f"Extracted condition states: {conditions}")
            return conditions
            
        except Exception as e:
            logger.error(f"Error extracting condition states: {str(e)}")
            raise

    def _compute_bbox_features(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute oriented bounding box features.
        
        Args:
            mask: Binary mask
            
        Returns:
            Array of bounding box features
        """
        try:
            # Ensure mask is binary
            mask = (mask > 0).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(
                mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return np.zeros(5)  # Return zero features if no contours
                
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get rotated rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            
            # Extract features
            center, (width, height), angle = rect
            area = width * height
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            
            return np.array([
                center[0], center[1],  # Center coordinates
                width, height,         # Width and height
                angle                  # Orientation
            ])
            
        except Exception as e:
            logger.error(f"Error computing bbox features: {str(e)}")
            raise

    def _compute_perimeter(self, mask: np.ndarray) -> float:
        """
        Compute perimeter of a binary mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Perimeter length
        """
        try:
            # Ensure mask is binary
            mask = (mask > 0).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(
                mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Sum up contour lengths
            perimeter = sum(cv2.arcLength(contour, closed=True) 
                        for contour in contours)
            
            return float(perimeter)
            
        except Exception as e:
            logger.error(f"Error computing perimeter: {str(e)}")
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