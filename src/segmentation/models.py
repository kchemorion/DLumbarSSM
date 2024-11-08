import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SpineSegmentationModel(nn.Module):
    """U-Net based model for spine segmentation"""
    
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        # Bridge
        self.bridge = ConvBlock(512, 1024)
        
        # Decoder
        self.dec4 = ConvBlock(1024 + 512, 512)
        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, 1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bridge
        bridge = self.bridge(self.pool(enc4))
        
        # Decoder
        dec4 = self.dec4(torch.cat([F.interpolate(bridge, enc4.shape[2:]), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.shape[2:]), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.shape[2:]), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.shape[2:]), enc1], 1))
        
        return self.final(dec1)

class SpineSegmentation:
    """Handles the segmentation pipeline"""
    
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
                pred = F.softmax(pred, dim=1)
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
            slice_cleaned = ndimage.binary_opening(labels[slice_idx])
            slice_cleaned = ndimage.binary_closing(slice_cleaned)
            cleaned.append(slice_cleaned)
        
        return