#datasets.py
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
from tqdm import tqdm


logger = logging.getLogger(__name__)

class SpineSegmentationDataset(Dataset):
    def __init__(self, 
                 data_root: Path,
                 mode: str = 'train',
                 series_types: List[str] = ["Sagittal T2/STIR", "Sagittal T1", "Axial T2"],
                 target_size: Tuple[int, int] = (256, 256),
                 max_samples: int = 5000):
        try:
            self.data_root = Path(data_root)
            if not self.data_root.exists():
                raise ValueError(f"Data root path does not exist: {self.data_root}")
            
            self.mode = mode
            self.series_types = series_types
            self.target_size = target_size
            self.max_samples = max_samples
            
            # Verify annotation files exist
            annotations_path = self.data_root / 'annotations'
            if not annotations_path.exists():
                raise ValueError(f"Annotations directory not found: {annotations_path}")
            
            required_files = ['train.csv', 'train_series_descriptions.csv', 'train_label_coordinates.csv']
            for file in required_files:
                if not (annotations_path / file).exists():
                    raise ValueError(f"Required annotation file not found: {annotations_path / file}")
            
            # Load annotations first
            self.train_df = pd.read_csv(annotations_path / 'train.csv')
            self.series_df = pd.read_csv(annotations_path / 'train_series_descriptions.csv')
            self.coords_df = pd.read_csv(annotations_path / 'train_label_coordinates.csv')
            
            logger.info(f"Loaded annotations: {len(self.train_df)} studies, "
                       f"{len(self.series_df)} series, {len(self.coords_df)} coordinates")
            
            # Define transformations for each view type
            self.transforms = {
                view_type: self._get_transform(mode, target_size, view_type)
                for view_type in series_types
            }
            
            # Load and prepare data
            self.data = self._load_and_prepare_data()
            
            if len(self.data) == 0:
                raise ValueError("No valid samples found in the dataset")
            
            # Subsample data if needed
            if len(self.data) > max_samples:
                indices = np.random.choice(len(self.data), max_samples, replace=False)
                self.data = [self.data[i] for i in indices]
            
            logger.info(f"Loaded {len(self.data)} samples for {mode}")
            
        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            raise
    
    def _load_and_prepare_data(self) -> List[Dict]:
        """Load and prepare data with detailed error checking"""
        try:
            prepared_data = []
            
            # Get unique study IDs
            study_ids = self.train_df['study_id'].unique()
            logger.info(f"Processing {len(study_ids)} unique studies")
            
            for study_id in tqdm(study_ids, desc="Loading studies"):
                try:
                    study_id_str = str(study_id)
                    
                    # Get series for this study
                    study_series = self.series_df[
                        (self.series_df['study_id'] == study_id) & 
                        (self.series_df['series_description'].isin(self.series_types))
                    ]
                    
                    if len(study_series) == 0:
                        continue
                    
                    for _, series_row in study_series.iterrows():
                        series_id = str(series_row['series_id'])
                        series_type = series_row['series_description']
                        
                        # Get coordinates for this series
                        series_coords = self.coords_df[
                            (self.coords_df['study_id'] == study_id) &
                            (self.coords_df['series_id'] == int(series_id))
                        ]
                        
                        if len(series_coords) == 0:
                            continue
                        
                        # Check if images exist
                        image_dir = self.data_root / 'raw' / study_id_str / series_id
                        if not image_dir.exists():
                            logger.warning(f"Image directory not found: {image_dir}")
                            continue
                        
                        dicom_files = sorted(image_dir.glob('*.dcm'))
                        if not dicom_files:
                            logger.warning(f"No DICOM files found in {image_dir}")
                            continue
                        
                        # Get study conditions
                        study_conditions = self.train_df[
                            self.train_df['study_id'] == study_id
                        ].iloc[0]
                        
                        # Add valid samples
                        for dcm_file in dicom_files:
                            prepared_data.append({
                                'study_id': study_id_str,
                                'series_id': series_id,
                                'series_type': series_type,
                                'coordinates': series_coords.to_dict('records'),
                                'conditions': study_conditions,
                                'image_path': str(dcm_file)
                            })
                            
                except Exception as e:
                    logger.error(f"Error processing study {study_id}: {str(e)}")
                    continue
            
            if not prepared_data:
                raise ValueError("No valid samples found after data preparation")
            
            logger.info(f"Successfully prepared {len(prepared_data)} samples")
            return prepared_data
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
    
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
        landmarks = self._extract_landmarks(item['coordinates'])  # Should return [10] tensor
        landmarks = landmarks.view(-1)  # Ensure flat tensor of size 10
        
        vertebral_levels = self._extract_levels(item['conditions'])
        conditions = self._extract_conditions(item['conditions'])
        levels = self._extract_levels(item['conditions'])  # Should return integers 0-2

        return {
            'images': images,
            'masks': masks,
            'landmarks': landmarks,
            'vertebral_levels': levels.long(),  # Ensure long type
            'conditions': conditions.long(),     # Ensure long type
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