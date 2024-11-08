import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from .models import SpineSegmentationModel
from scipy import ndimage
from src.utils.gpu_utils import setup_gpu

logger = logging.getLogger(__name__)

class SpineSegmentation:
    """Handles the complete segmentation pipeline"""
    
    def __init__(self, model_path: str = None):
        self.device = setup_gpu()
        self.model = SpineSegmentationModel().to(self.device)
        
        if model_path and Path(model_path).exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
        
        self.model.eval()
        
        # Log device and model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model initialized with {total_params:,} parameters on {self.device}")
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
            logger.info(f"GPU memory allocated: {memory_allocated:.2f}GB")
        
    def segment_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Segment a volume of spine MRI images.
        
        Args:
            volume: Input volume of shape (D, H, W) where D is the number of slices
            
        Returns:
            Segmentation mask of shape (D, C, H, W) where C is number of classes
        """
        try:
            if volume.ndim != 3:
                raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
                
            segmentations = []
            depth = volume.shape[0]
            
            logger.info(f"Processing volume of shape {volume.shape}")
            
            with torch.no_grad():
                for slice_idx in range(depth):
                    # Normalize slice
                    slice_data = volume[slice_idx].astype(np.float32)
                    slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
                    
                    # Prepare input tensor
                    img = torch.from_numpy(slice_data).float()
                    img = img.unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    # Get prediction
                    pred = self.model(img)
                    pred = torch.softmax(pred, dim=1)
                    pred = pred.cpu().numpy()
                    
                    segmentations.append(pred[0])  # Remove batch dimension
            
            # Stack along depth dimension
            segmentation_volume = np.stack(segmentations, axis=0)
            logger.info(f"Generated segmentation of shape {segmentation_volume.shape}")
            
            return segmentation_volume
            
        except Exception as e:
            logger.error(f"Error during segmentation: {str(e)}")
            raise
    
    def post_process(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to the segmentation.
        
        Args:
            segmentation: Raw segmentation of shape (D, C, H, W)
            
        Returns:
            Processed segmentation of shape (D, H, W)
        """
        try:
            # Convert probabilities to labels
            labels = np.argmax(segmentation, axis=1)
            
            # Apply morphological operations to clean up the segmentation
            cleaned = []
            for slice_idx in range(labels.shape[0]):
                # Remove small objects
                slice_cleaned = ndimage.binary_opening(labels[slice_idx])
                # Fill holes
                slice_cleaned = ndimage.binary_closing(slice_cleaned)
                cleaned.append(slice_cleaned)
            
            cleaned_volume = np.stack(cleaned, axis=0)
            logger.info(f"Post-processed segmentation shape: {cleaned_volume.shape}")
            
            return cleaned_volume
            
        except Exception as e:
            logger.error(f"Error during post-processing: {str(e)}")
            raise
    
    def extract_vertebrae_coordinates(self, segmentation: np.ndarray) -> List[Dict]:
        """
        Extract vertebrae coordinates from segmentation.
        
        Args:
            segmentation: Binary segmentation mask of shape (D, H, W)
            
        Returns:
            List of coordinate dictionaries
        """
        try:
            coordinates = []
            
            for slice_idx in range(segmentation.shape[0]):
                slice_seg = segmentation[slice_idx]
                # Find connected components
                labeled, num_features = ndimage.label(slice_seg)
                
                for label in range(1, num_features + 1):
                    # Get center of mass for each vertebra
                    center = ndimage.center_of_mass(slice_seg == label)
                    
                    coordinates.append({
                        'slice_idx': slice_idx,
                        'y': float(center[0]),
                        'x': float(center[1]),
                        'label': int(label)
                    })
            
            logger.info(f"Extracted {len(coordinates)} vertebrae coordinates")
            return coordinates
            
        except Exception as e:
            logger.error(f"Error extracting coordinates: {str(e)}")
            raise