# src/training/datasets.py

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

class SpineSegmentationDataset(Dataset):
    """Dataset for training spine segmentation model"""
    
    def __init__(self, 
                 data_root: Path,
                 series_type: str = "Sagittal T2/STIR",
                 mode: str = 'train',
                 transform=None):
        """
        Initialize dataset
        
        Args:
            data_root: Root directory containing data
            series_type: Type of series to use
            mode: 'train' or 'val'
            transform: Optional transforms to apply
        """
        self.data_root = Path(data_root)
        self.series_type = series_type
        self.mode = mode
        
        # Define augmentations
        if mode == 'train':
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                ], p=0.3),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
        
        # Load annotations
        self.data = self._load_and_prepare_data()
        logger.info(f"Loaded {len(self.data)} samples for {mode}")
        
    def _load_and_prepare_data(self) -> List[Dict]:
        """Load and prepare dataset entries"""
        # Load CSV files
        series_df = pd.read_csv(self.data_root / 'annotations' / 'train_series_descriptions.csv')
        coords_df = pd.read_csv(self.data_root / 'annotations' / 'train_label_coordinates.csv')
        
        # Filter for desired series type
        series_info = series_df[series_df['series_description'] == self.series_type]
        
        prepared_data = []
        for _, row in series_info.iterrows():
            study_id = str(row['study_id'])
            series_id = str(row['series_id'])
            
            # Get coordinates for this series
            series_coords = coords_df[
                (coords_df['study_id'] == int(study_id)) &
                (coords_df['series_id'] == int(series_id))
            ]
            
            if len(series_coords) > 0:
                image_dir = self.data_root / 'raw' / 'train_images' / study_id / series_id
                if image_dir.exists():
                    prepared_data.append({
                        'study_id': study_id,
                        'series_id': series_id,
                        'coordinates': series_coords.to_dict('records'),
                        'image_dir': image_dir
                    })
        
        return prepared_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        item = self.data[idx]
        
        # Load image
        image_paths = sorted(item['image_dir'].glob('*.dcm'))
        image = self._load_dicom_image(image_paths[0])
        
        # Create mask
        mask = self._create_mask_from_coordinates(
            item['coordinates'],
            image.shape
        )
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask,
            'study_id': item['study_id'],
            'series_id': item['series_id']
        }
    
    def _load_dicom_image(self, path: Path) -> np.ndarray:
        """Load and preprocess DICOM image"""
        import pydicom
        dcm = pydicom.dcmread(str(path))
        image = dcm.pixel_array.astype(float)
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image
    
    def _create_mask_from_coordinates(self, 
                                    coordinates: List[Dict],
                                    image_shape: Tuple[int, int]) -> np.ndarray:
        """Create segmentation mask from coordinates"""
        mask = np.zeros(image_shape, dtype=np.float32)
        
        for coord in coordinates:
            x, y = int(coord['x']), int(coord['y'])
            cv2.circle(mask, (x, y), radius=3, color=1, thickness=-1)
        
        # Apply Gaussian smoothing to create soft masks
        mask = gaussian_filter(mask, sigma=2)
        
        return mask