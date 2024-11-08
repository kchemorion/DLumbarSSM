import os
import pandas as pd
import numpy as np
from pathlib import Path
import pydicom
from typing import Dict, List, Tuple

class SpineDataLoader:
    """
    Handles loading and initial processing of spine imaging data and annotations.
    """
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.train_csv = None
        self.train_coords = None
        self.series_desc = None
        self.image_data = {}
        
    def load_annotations(self) -> None:
        """Load all annotation files"""
        # Load main training labels
        self.train_csv = pd.read_csv(self.data_root / 'annotations' / 'train.csv')
        
        # Load coordinate annotations
        self.train_coords = pd.read_csv(self.data_root / 'annotations' / 'train_label_coordinates.csv')
        
        # Load series descriptions
        self.series_desc = pd.read_csv(self.data_root / 'annotations' / 'train_series_descriptions.csv')
        
    def get_patient_data(self, study_id: str) -> Dict:
        """
        Retrieve all data for a specific patient/study
        """
        patient_data = {
            'labels': self.train_csv[self.train_csv['study_id'] == study_id].iloc[0].to_dict(),
            'coordinates': self.train_coords[self.train_coords['study_id'] == study_id],
            'series': self.series_desc[self.series_desc['study_id'] == study_id],
            'images': self._load_patient_images(study_id)
        }
        return patient_data
    
    def _load_patient_images(self, study_id: str) -> Dict:
        """Load all image series for a patient"""
        image_path = self.data_root / 'raw' / 'train_images' / str(study_id)
        series_data = {}
        
        if not image_path.exists():
            return series_data
            
        for series_folder in image_path.iterdir():
            if series_folder.is_dir():
                series_id = series_folder.name
                series_images = []
                
                # Load all DICOM files in series
                for dcm_file in sorted(series_folder.glob('*.dcm')):
                    try:
                        ds = pydicom.dcmread(dcm_file)
                        series_images.append({
                            'image': ds.pixel_array,
                            'instance_number': ds.InstanceNumber,
                            'slice_location': getattr(ds, 'SliceLocation', None)
                        })
                    except Exception as e:
                        print(f"Error loading {dcm_file}: {e}")
                        
                series_data[series_id] = series_images
                
        return series_data
    
    def get_condition_distribution(self) -> pd.DataFrame:
        """Analyze the distribution of conditions in the dataset"""
        condition_cols = [col for col in self.train_csv.columns 
                         if col not in ['study_id']]
        
        distribution = {}
        for col in condition_cols:
            distribution[col] = self.train_csv[col].value_counts()
            
        return pd.DataFrame(distribution)

    def get_level_statistics(self) -> Dict:
        """Get statistics for each vertebral level"""
        levels = ['L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1']
        stats = {}
        
        for level in levels:
            level_cols = [col for col in self.train_csv.columns if level in col]
            level_data = self.train_csv[level_cols]
            
            stats[level] = {
                'stenosis_count': level_data[[col for col in level_cols 
                                            if 'stenosis' in col]].value_counts(),
                'foraminal_count': level_data[[col for col in level_cols 
                                             if 'foraminal' in col]].value_counts(),
                'subarticular_count': level_data[[col for col in level_cols 
                                                if 'subarticular' in col]].value_counts()
            }
            
        return stats