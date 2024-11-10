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

logger = logging.getLogger(__name__)

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
        
        # Load all annotations at initialization
        self.train_df = pd.read_csv(self.data_root / 'annotations' / 'train.csv')
        self.coords_df = pd.read_csv(self.data_root / 'annotations' / 'train_label_coordinates.csv')
        self.series_df = pd.read_csv(self.data_root / 'annotations' / 'train_series_descriptions.csv')
        
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
        
        # Handle missing data and NaN values in DataFrames
        self.train_df = self.train_df.fillna('Normal/Mild')  # Fill NaN with Normal/Mild
        
        for series_type in self.series_types:
            series_info = self.series_df[self.series_df['series_description'] == series_type]
            for _, row in series_info.iterrows():
                try:
                    study_id = str(row['study_id'])
                    series_id = str(row['series_id'])
                    
                    # Get coordinates for this series
                    series_coords = self.coords_df[
                        (self.coords_df['study_id'] == int(study_id)) &
                        (self.coords_df['series_id'] == int(series_id))
                    ]
                    
                    # Get conditions for this study
                    study_conditions = self.train_df[
                        self.train_df['study_id'] == int(study_id)
                    ].iloc[0]
                    
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
                                    'conditions': study_conditions,
                                    'image_path': str(dcm_file)
                                })
                                
                except Exception as e:
                    logger.warning(f"Error preparing data for study {study_id}: {str(e)}")
                    continue
                    
        return prepared_data
    
    def _extract_landmarks(self, coordinates: List[Dict]) -> torch.Tensor:
        """Extract landmark coordinates for each vertebral level"""
        landmarks = []
        level_mapping = {
            'L1/L2': 0, 'L2/L3': 1, 'L3/L4': 2, 'L4/L5': 3, 'L5/S1': 4
        }
        
        # Initialize empty landmarks
        level_coords = {level: [] for level in level_mapping.keys()}
        
        # Group coordinates by level
        for coord in coordinates:
            level = coord['level']
            if level in level_mapping:
                level_coords[level].append((coord['x'], coord['y']))
        
        # Process each level
        for level in level_mapping.keys():
            if level_coords[level]:
                # Average coordinates if multiple points exist
                x_coords, y_coords = zip(*level_coords[level])
                x = float(np.mean(x_coords))
                y = float(np.mean(y_coords))
            else:
                # Use zeros for missing levels
                x, y = 0.0, 0.0
            landmarks.extend([x, y])
        
        return torch.tensor(landmarks, dtype=torch.float32)
    
    def _extract_levels(self, conditions: pd.Series) -> torch.Tensor:
        """Extract vertebral level labels with NaN handling"""
        # Initialize tensor for 5 levels x 3 conditions (stenosis, foraminal, subarticular)
        levels = torch.zeros(5, 3)
        
        # Mapping for severity scores
        severity_map = {
            'Normal/Mild': 0,
            'Moderate': 1,
            'Severe': 2
        }
        
        # Process each level
        for i, level in enumerate(['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']):
            try:
                # Spinal canal stenosis
                stenosis_key = f'spinal_canal_stenosis_{level}'
                stenosis_val = conditions.get(stenosis_key, 'Normal/Mild')
                levels[i, 0] = severity_map.get(stenosis_val, 0)  # Default to Normal/Mild
                
                # Neural foraminal narrowing (use maximum of left/right)
                left_key = f'left_neural_foraminal_narrowing_{level}'
                right_key = f'right_neural_foraminal_narrowing_{level}'
                
                left_val = conditions.get(left_key, 'Normal/Mild')
                right_val = conditions.get(right_key, 'Normal/Mild')
                
                # Handle NaN values
                left_foraminal = severity_map.get(left_val if pd.notna(left_val) else 'Normal/Mild', 0)
                right_foraminal = severity_map.get(right_val if pd.notna(right_val) else 'Normal/Mild', 0)
                levels[i, 1] = max(left_foraminal, right_foraminal)
                
                # Subarticular stenosis (use maximum of left/right)
                left_sub_key = f'left_subarticular_stenosis_{level}'
                right_sub_key = f'right_subarticular_stenosis_{level}'
                
                left_sub_val = conditions.get(left_sub_key, 'Normal/Mild')
                right_sub_val = conditions.get(right_sub_key, 'Normal/Mild')
                
                # Handle NaN values
                left_sub = severity_map.get(left_sub_val if pd.notna(left_sub_val) else 'Normal/Mild', 0)
                right_sub = severity_map.get(right_sub_val if pd.notna(right_sub_val) else 'Normal/Mild', 0)
                levels[i, 2] = max(left_sub, right_sub)
                
            except Exception as e:
                logger.warning(f"Error processing level {level}: {str(e)}. Using default values.")
                # Keep zeros for this level
                continue
        
        return levels
    
    def _extract_conditions(self, conditions: pd.Series) -> torch.Tensor:
        """Extract overall condition severity with NaN handling"""
        severity_map = {
            'Normal/Mild': 0,
            'Moderate': 1,
            'Severe': 2
        }
        
        try:
            # Get maximum severity for each condition type across all levels
            stenosis = max(
                severity_map.get(conditions[col] if pd.notna(conditions[col]) else 'Normal/Mild', 0)
                for col in conditions.index if 'spinal_canal_stenosis' in col
            )
            
            foraminal = max(
                severity_map.get(conditions[col] if pd.notna(conditions[col]) else 'Normal/Mild', 0)
                for col in conditions.index if 'neural_foraminal_narrowing' in col
            )
            
            subarticular = max(
                severity_map.get(conditions[col] if pd.notna(conditions[col]) else 'Normal/Mild', 0)
                for col in conditions.index if 'subarticular_stenosis' in col
            )
            
        except Exception as e:
            logger.warning(f"Error extracting conditions: {str(e)}. Using default values.")
            stenosis, foraminal, subarticular = 0, 0, 0
        
        return torch.tensor([stenosis, foraminal, subarticular], dtype=torch.long)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Load all modalities
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
        
        # Extract multi-task labels
        landmarks = self._extract_landmarks(item['coordinates'])
        vertebral_levels = self._extract_levels(item['conditions'])
        conditions = self._extract_conditions(item['conditions'])
        
        return {
            'images': images,
            'masks': masks,
            'landmarks': landmarks,
            'vertebral_levels': vertebral_levels,
            'conditions': conditions,
            'study_id': item['study_id'],
            'series_id': item['series_id']
        }
    
    def _get_transform(self, mode: str, target_size: Tuple[int, int], series_type: str) -> A.Compose:
        if mode == 'train':
            return A.Compose([
                A.Resize(height=target_size[0], width=target_size[1], always_apply=True),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
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
            image = dcm.pixel_array.astype(np.float32)
            image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
            return image
        except Exception as e:
            logger.error(f"Error loading DICOM {path}: {e}")
            return np.zeros(self.target_size, dtype=np.uint8)
    
    def _create_mask_from_coordinates(self, coordinates: List[Dict], 
                                    image_shape: Tuple[int, int]) -> np.ndarray:
        """Create segmentation mask from coordinates"""
        mask = np.zeros(image_shape, dtype=np.float32)
        
        for coord in coordinates:
            x, y = int(coord['x']), int(coord['y'])
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                cv2.circle(mask, (x, y), radius=3, color=1, thickness=-1)
        
        return mask