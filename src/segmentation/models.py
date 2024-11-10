import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpineSegmentationModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=4):
        super().__init__()
        self.enc1 = self._make_encoder_block(in_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)

        self.dec4 = self._make_decoder_block(512, 256)
        self.dec3 = self._make_decoder_block(256 + 256, 128)
        self.dec2 = self._make_decoder_block(128 + 128, 64)
        self.dec1 = self._make_decoder_block(64 + 64, out_channels)

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        dec4 = self.dec4(enc4)
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))

        return dec1

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