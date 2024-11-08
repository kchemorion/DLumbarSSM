import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from .models import SpineSegmentationModel
from scipy import ndimage

class SpineSegmentation:
    """Handles the complete segmentation pipeline"""
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpineSegmentationModel().to(self.device)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval()
        
    def segment_volume(self, volume: np.ndarray) -> np.ndarray:
        """Segment a volume of spine MRI images"""
        segmentations = []
        
        with torch.no_grad():
            for slice_idx in range(volume.shape[0]):
                # Prepare input
                img = torch.from_numpy(volume[slice_idx]).float()
                img = img.unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Get prediction
                pred = self.model(img)
                pred = torch.softmax(pred, dim=1)
                pred = pred.cpu().numpy()[0]
                
                segmentations.append(pred)
        
        return np.stack(segmentations, axis=0)
    
    def post_process(self, segmentation: np.ndarray) -> np.ndarray:
        """Apply post-processing to the segmentation"""
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
        
        return np.stack(cleaned, axis=0)
    
    def extract_vertebrae_coordinates(self, segmentation: np.ndarray) -> List[Dict]:
        """Extract vertebrae coordinates from segmentation"""
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
                    'y': center[0],
                    'x': center[1],
                    'label': label
                })
        
        return coordinates