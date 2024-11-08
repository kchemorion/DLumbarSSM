import numpy as np
from scipy import ndimage
import cv2
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
from pathlib import Path

class SpineImagePreprocessor:
    """Handles preprocessing of spine MRI images"""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        self.intensity_scaler = StandardScaler()
        
    def preprocess_series(self, image_series: List[Dict]) -> List[np.ndarray]:
        """Preprocess a series of images"""
        processed_series = []
        
        for image_data in image_series:
            processed = self.preprocess_single_image(image_data['image'])
            processed_series.append(processed)
            
        return processed_series
    
    def preprocess_single_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image"""
        # 1. Basic normalization
        img_normalized = self._normalize_intensity(image)
        
        # 2. Resize to target size
        img_resized = cv2.resize(img_normalized, self.target_size)
        
        # 3. Enhance contrast
        img_enhanced = self._enhance_contrast(img_resized)
        
        # 4. Denoise
        img_denoised = self._denoise_image(img_enhanced)
        
        return img_denoised
    
    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity"""
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max > img_min:
            img_normalized = (image - img_min) / (img_max - img_min)
        else:
            img_normalized = image
            
        return img_normalized
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_enhanced = clahe.apply((image * 255).astype(np.uint8))
        return img_enhanced.astype(float) / 255
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising"""
        return cv2.fastNlMeansDenoising(
            (image * 255).astype(np.uint8),
            None,
            h=10,
            searchWindowSize=21,
            templateWindowSize=7
        ).astype(float) / 255

class SpineSegmentationPreprocessor:
    """Prepares images specifically for segmentation"""
    
    def __init__(self, preprocessor: SpineImagePreprocessor):
        self.preprocessor = preprocessor
        
    def prepare_for_segmentation(self, image_series: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """Prepare image series for segmentation"""
        # 1. Preprocess images
        processed_images = self.preprocessor.preprocess_series(image_series)
        
        # 2. Stack images into volume
        volume = np.stack(processed_images, axis=0)
        
        # 3. Extract metadata
        metadata = [{
            'instance_number': img_data['instance_number'],
            'slice_location': img_data['slice_location']
        } for img_data in image_series]
        
        return volume, metadata
    
    def extract_spine_region(self, volume: np.ndarray) -> np.ndarray:
        """Extract the spine region from the volume"""
        # This is a placeholder for more sophisticated region extraction
        return volume