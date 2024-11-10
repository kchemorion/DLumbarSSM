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

# src/training/datasets.py

class SpineSegmentationDataset(Dataset):
    def __init__(self, 
                 data_root: Path,
                 mode: str = 'train',
                 series_types: List[str] = ["Sagittal T2/STIR", "Sagittal T1", "Axial T2"],
                 target_size: Tuple[int, int] = (256, 256),
                 max_samples: int = 5000):
        self.data_root = Path(data_root)
        self.mode = mode
        self.series_types = series_types
        self.target_size = target_size
        self.max_samples = max_samples
        
        # Define transformations for each view type
        self.transforms = {
            view_type: self._get_transform(mode, target_size, view_type)
            for view_type in series_types
        }
        
        self.data = self._load_and_prepare_data()
        
        # Subsample data if needed
        if len(self.data) > max_samples:
            indices = np.random.choice(len(self.data), max_samples, replace=False)
            self.data = [self.data[i] for i in indices]
        
        logger.info(f"Loaded {len(self.data)} samples for {mode}")
    
    def _load_and_prepare_data(self) -> List[Dict]:
        prepared_data = []
        series_df = pd.read_csv(self.data_root / 'annotations' / 'train_series_descriptions.csv')
        coords_df = pd.read_csv(self.data_root / 'annotations' / 'train_label_coordinates.csv')
        
        for series_type in self.series_types:
            series_info = series_df[series_df['series_description'] == series_type]
            for _, row in series_info.iterrows():
                study_id = str(row['study_id'])
                series_id = str(row['series_id'])
                series_coords = coords_df[
                    (coords_df['study_id'] == int(study_id)) &
                    (coords_df['series_id'] == int(series_id))
                ]
                if len(series_coords) > 0:
                    image_dir = self.data_root / 'raw' / 'train_images' / study_id / series_id
                    if image_dir.exists():
                        dicom_files = sorted(image_dir.glob('*.dcm'))
                        for dcm_file in dicom_files:
                            prepared_data.append({
                                'study_id': study_id,
                                'series_id': series_id,
                                'series_type': series_type,
                                'coordinates': series_coords.to_dict('records'),
                                'image_path': str(dcm_file)
                            })
        return prepared_data
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        images = {}
        masks = {}
        for series_type in self.series_types:
            image = self._load_dicom_image(item['image_path'])
            image = image.astype(np.float32)
            mask = self._create_mask_from_coordinates(item['coordinates'], image.shape)
            mask = mask.astype(np.float32)
            
            transformed = self.transforms[series_type](image=image, mask=mask)
            images[series_type] = transformed['image']
            masks[series_type] = transformed['mask'].unsqueeze(0).repeat(4, 1, 1)
        
        return {
            'images': images,
            'masks': masks,
            'study_id': item['study_id'],
            'series_id': item['series_id']
        }
    
    def _get_transform(self, mode: str, target_size: Tuple[int, int], series_type: str) -> A.Compose:
        if mode == 'train':
            return A.Compose([
                A.Resize(height=target_size[0], width=target_size[1], always_apply=True),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                
                # Removed heavy augmentations
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                ], p=0.3),
                
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=target_size[0], width=target_size[1], always_apply=True),
                A.Normalize(mean=[0.485], std=[0.229]),
                ToTensorV2(),
            ])
    

    
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_dicom_image(self, path: Path) -> np.ndarray:
        """Memory-efficient DICOM loading"""
        try:
            import pydicom
            dcm = pydicom.dcmread(str(path))
            
            # Convert to uint8 to save memory
            image = dcm.pixel_array.astype(np.float32)
            image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading DICOM {path}: {e}")
            # Return blank image of target size
            return np.zeros(self.target_size, dtype=np.uint8)
    
    def _create_mask_from_coordinates(self, coordinates: List[Dict], 
                                    image_shape: Tuple[int, int]) -> np.ndarray:
        """Create segmentation mask from coordinates"""
        mask = np.zeros(image_shape, dtype=np.float32)
        
        for coord in coordinates:
            x, y = int(coord['x']), int(coord['y'])
            # Skip if coordinates are outside image
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                cv2.circle(mask, (x, y), radius=3, color=1, thickness=-1)
        
        return mask