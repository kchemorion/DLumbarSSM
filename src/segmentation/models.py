"""
MultiView-SpineNet: A Novel Architecture for Multi-view Spine Analysis with Cross-attention and Anatomical Constraints

This implementation introduces several key innovations:
1. Cross-view attention fusion for multi-sequence MRI analysis
2. Anatomically-aware architecture with vertebral level detection
3. Multi-task learning for comprehensive spine analysis
4. Progressive feature fusion with anatomical constraints

Author: Francis Kiptengwer Chemorion
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

logger = logging.getLogger(__name__)

class CrossViewAttention(nn.Module):
    """Cross-view attention module for fusing information from multiple MRI sequences"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-head self-attention for each view
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout)
        
        # Cross-view attention
        self.cross_q = nn.Linear(dim, dim)
        self.cross_k = nn.Linear(dim, dim)
        self.cross_v = nn.Linear(dim, dim)
        
        # Adaptive fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(3))  # One per view
        self.softmax = nn.Softmax(dim=0)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.LayerNorm(dim)
        )
    
    def forward(self, views: List[torch.Tensor]) -> torch.Tensor:
        """
        Process multiple views through self and cross attention
        
        Args:
            views: List of tensors [B, C, H, W] for each view
            
        Returns:
            Fused feature tensor
            
        Raises:
            ValueError: If input views are invalid
            RuntimeError: If tensor operations fail
        """
        try:
            # Input validation
            if not views or not isinstance(views, list):
                raise ValueError("Views must be a non-empty list of tensors")
            
            if not all(isinstance(v, torch.Tensor) for v in views):
                raise ValueError("All views must be torch tensors")
            
            B = views[0].shape[0]
            
            # Reshape views for attention
            try:
                views = [view.flatten(2).permute(2, 0, 1) for view in views]  # [HW, B, C]
            except Exception as e:
                raise RuntimeError(f"Failed to reshape views: {str(e)}")
            
            # 1. Self-attention per view
            attended_views = []
            for i, view in enumerate(views):
                try:
                    attended, _ = self.self_attention(view, view, view)
                    attended_views.append(attended)
                except Exception as e:
                    raise RuntimeError(f"Self-attention failed for view {i}: {str(e)}")
            
            # 2. Cross-view attention
            try:
                fusion_weights = self.softmax(self.fusion_weights)
                queries = self.cross_q(torch.stack(attended_views, dim=0))  # [V, HW, B, C]
                keys = self.cross_k(torch.stack(attended_views, dim=0))
                values = self.cross_v(torch.stack(attended_views, dim=0))
                
                # Compute cross-attention scores
                scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
                attn = F.softmax(scores, dim=-1)
                
                # Weight and combine views
                cross_attended = torch.matmul(attn, values)  # [V, HW, B, C]
                fused = torch.sum(cross_attended * fusion_weights.view(-1, 1, 1, 1), dim=0)
                
                # Project output
                fused = self.out_proj(fused)
                
                # Reshape back to spatial dimensions
                H = W = int(math.sqrt(fused.size(0)))
                fused = fused.permute(1, 2, 0).view(B, -1, H, W)
                
                return fused
                
            except Exception as e:
                raise RuntimeError(f"Cross-attention processing failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise


class AnatomicalConstraintModule(nn.Module):
    """Enforces anatomical constraints on vertebral predictions"""
    
    def __init__(self, num_levels: int = 5):
        super().__init__()
        self.num_levels = num_levels
        
        # Learned adjacency matrix for vertebral levels
        self.adjacency = nn.Parameter(torch.eye(num_levels))
        
        # Vertebral shape prior
        self.shape_prior = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_levels, 1)
        )
        
        # Spatial transformer for alignment
        self.stn = SpatialTransformer(256)
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply anatomical constraints to features
        
        Returns:
            Tuple of (constrained features, adjacency matrix)
        """
        try:
            # Generate shape prior
            shape_attention = self.shape_prior(features)
            
            # Apply spatial transformer
            aligned_features = self.stn(features)
            
            # Apply adjacency constraints
            adj_matrix = F.softmax(self.adjacency, dim=1)
            constrained = torch.matmul(adj_matrix, aligned_features.view(*aligned_features.shape[:2], -1))
            constrained = constrained.view_as(aligned_features)
            
            return constrained, shape_attention
            
        except Exception as e:
            logger.error(f"Error in AnatomicalConstraintModule forward pass: {str(e)}")
            raise

class SpatialTransformer(nn.Module):
    """Spatial transformer network for feature alignment"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            xs = self.localization(x)
            xs = xs.view(-1, 64 * 4 * 4)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)
            
            grid = F.affine_grid(theta, x.size())
            x = F.grid_sample(x, grid)
            return x
            
        except Exception as e:
            logger.error(f"Error in SpatialTransformer forward pass: {str(e)}")
            raise

class MultiViewSpineNet(nn.Module):
    """
    Complete multi-view spine analysis architecture with anatomical awareness
    and multi-task capabilities.
    """
    
    def __init__(self, 
                 in_channels: int = 1,
                 num_classes: int = 4,
                 num_levels: int = 5):
        super().__init__()
        try:
            self.num_levels = num_levels
            
            # View-specific encoders
            self.encoders = nn.ModuleDict({
                'Sagittal T2/STIR': self._make_encoder(in_channels),
                'Sagittal T1': self._make_encoder(in_channels),
                'Axial T2': self._make_encoder(in_channels)
            })
            
            # Cross-view attention modules
            self.cross_attention = nn.ModuleList([
                CrossViewAttention(dim=64),
                CrossViewAttention(dim=128),
                CrossViewAttention(dim=256)
            ])
            
            # Anatomical constraint module
            self.anatomical_module = AnatomicalConstraintModule(num_levels)
            
            # Task-specific decoders
            self.segmentation_decoder = self._make_segmentation_decoder(256, num_classes)
            self.level_classifier = self._make_level_classifier()
            self.landmark_detector = self._make_landmark_detector()
            self.condition_classifier = self._make_condition_classifier()
            
        except Exception as e:
            logger.error(f"Error initializing MultiViewSpineNet: {str(e)}")
            raise
    
    def _make_encoder(self, in_channels: int) -> nn.ModuleList:
        """Create progressive encoder with residual blocks"""
        try:
            return nn.ModuleList([
                nn.Sequential(
                    ResBlock(in_channels, 64),
                    ResBlock(64, 64)
                ),
                nn.Sequential(
                    ResBlock(64, 128),
                    ResBlock(128, 128)
                ),
                nn.Sequential(
                    ResBlock(128, 256),
                    ResBlock(256, 256)
                )
            ])
        except Exception as e:
            logger.error(f"Error creating encoder: {str(e)}")
            raise
    
    def _make_segmentation_decoder(self, in_channels: int, num_classes: int) -> nn.Module:
        """Create segmentation decoder"""
        try:
            return nn.Sequential(
                UpBlock(in_channels, 128),
                UpBlock(128, 64),
                UpBlock(64, 32),
                nn.Conv2d(32, num_classes, 1),
                nn.BatchNorm2d(num_classes)
            )
        except Exception as e:
            logger.error(f"Error creating segmentation decoder: {str(e)}")
            raise
    
    def _make_level_classifier(self) -> nn.Module:
        """Create vertebral level classifier"""
        try:
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_levels),
                nn.BatchNorm1d(self.num_levels)
            )
        except Exception as e:
            logger.error(f"Error creating level classifier: {str(e)}")
            raise
    
    def _make_landmark_detector(self) -> nn.Module:
        """Create anatomical landmark detector"""
        try:
            return nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(128, self.num_levels * 2, 1),
                nn.BatchNorm2d(self.num_levels * 2)
            )
        except Exception as e:
            logger.error(f"Error creating landmark detector: {str(e)}")
            raise
    
    def _make_condition_classifier(self) -> nn.Module:
        """Create degenerative condition classifier"""
        try:
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 128),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(128, 3),  # 3 condition types
                nn.BatchNorm1d(3)
            )
        except Exception as e:
            logger.error(f"Error creating condition classifier: {str(e)}")
            raise
    
    def forward(self, views: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            views: Dictionary of input views {view_name: tensor}
            
        Returns:
            Dictionary of outputs for each task
        """
        try:
            # Input validation
            if not views or not isinstance(views, dict):
                raise ValueError("Views must be a non-empty dictionary of tensors")
            
            # 1. View-specific Processing
            features = {name: [] for name in views.keys()}
            
            # Validate input views match expected encoders
            for name in views.keys():
                if name not in self.encoders:
                    raise ValueError(f"Received unexpected view type: {name}. "
                                f"Expected one of {list(self.encoders.keys())}")
            
            # Process each view
            for name, view in views.items():
                for i, encoder in enumerate(self.encoders[name]):
                    if i == 0:
                        features[name].append(encoder(view))
                    else:
                        features[name].append(
                            encoder(F.max_pool2d(features[name][-1], 2))
                        )
            
            # 2. Progressive Feature Fusion
            fused_features = []
            for level in range(len(self.cross_attention)):
                level_features = [
                    features[name][level] for name in views.keys()
                ]
                fused = self.cross_attention[level](level_features)
                fused_features.append(fused)
            
            # 3. Apply Anatomical Constraints
            constrained_features, shape_attention = self.anatomical_module(fused_features[-1])
            
            # 4. Multi-task Predictions
            outputs = {
                'segmentation': {
                    name: self.segmentation_decoder(constrained_features) 
                    for name in views.keys()
                },
                'vertebral_levels': self.level_classifier(constrained_features),
                'landmarks': self.landmark_detector(constrained_features),
                'conditions': self.condition_classifier(constrained_features),
                'shape_attention': shape_attention
            }
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error in MultiViewSpineNet forward pass: {str(e)}")
            raise

class ResBlock(nn.Module):
    """Residual block with pre-activation"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        try:
            # First conv block
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(in_channels)  # BN before conv
            
            # Second conv block
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)  # BN on out_channels
            
            # Shortcut connection
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()
                
            # Initialize weights
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    
        except Exception as e:
            logger.error(f"Error initializing ResBlock: {str(e)}")
            raise
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            identity = self.shortcut(x)
            
            # Pre-activation pattern
            out = self.bn1(x)
            out = F.relu(out)
            out = self.conv1(out)
            
            out = self.bn2(out)
            out = F.relu(out)
            out = self.conv2(out)
            
            out += identity
            return out
            
        except Exception as e:
            logger.error(f"Error in ResBlock forward pass: {str(e)}")
            raise

class UpBlock(nn.Module):
    """Upsampling block with skip connections"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        try:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            self.bn = nn.BatchNorm2d(out_channels)
            self.conv = ResBlock(out_channels * 2, out_channels)
            
            # Initialize weights
            for m in self.modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    m.running_mean.zero_()
                    m.running_var.fill_(1)
                    
        except Exception as e:
            logger.error(f"Error initializing UpBlock: {str(e)}")
            raise
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            x = self.up(x)
            x = self.bn(x)
            
            if skip is not None:
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
            
            x = self.conv(x)
            return x
            
        except Exception as e:
            logger.error(f"Error in UpBlock forward pass: {str(e)}")
            raise

def initialize_weights(model: nn.Module) -> None:
    """Initialize model weights properly"""
    try:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.running_mean.zero_()  # Initialize running mean
                m.running_var.fill_(1)  # Initialize running variance
    except Exception as e:
        logger.error(f"Error initializing weights: {str(e)}")
        raise