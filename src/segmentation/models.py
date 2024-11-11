"""
MultiView-SpineNet: A Novel Architecture for Multi-view Spine Analysis with Cross-attention and Anatomical Constraints

This implementation introduces several key innovations:
1. Cross-view attention fusion for multi-sequence MRI analysis
2. Anatomically-aware architecture with vertebral level detection
3. Multi-task learning for comprehensive spine analysis
4. Progressive feature fusion with anatomical constraints

Author: Francis Kiptengwer Chemorion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
from torch.utils.checkpoint import checkpoint
import logging
import torch._dynamo
torch._dynamo.config.suppress_errors = True

logger = logging.getLogger(__name__)  
logger.info("Initializing MultiView-SpineNet module")

class CrossViewAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, chunk_size: int = 1024):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.chunk_size = chunk_size
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    
    def _chunk_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, chunk_size: int):
        B, H, L, D = q.shape  # B: batch, H: heads, L: sequence length, D: head dimension
        
        out = []
        for i in range(0, L, chunk_size):
            chunk_q = q[:, :, i:i+chunk_size]
            
            # Compute attention scores for this chunk
            attn = torch.matmul(chunk_q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention to values
            chunk_out = torch.matmul(attn, v)
            out.append(chunk_out)
        
        return torch.cat(out, dim=2)
    
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        try:
            B, C, H, W = features_list[0].shape
            V = len(features_list)
            N = H * W
            
            # Process features and reshape
            features_flat = []
            for feat in features_list:
                # Process in smaller chunks to save memory
                feat = feat.reshape(B, C, -1).permute(0, 2, 1)
                features_flat.append(feat)
            
            x = torch.cat(features_flat, dim=1)
            
            # Project to q, k, v
            q = self.to_q(x)
            k = self.to_k(x)
            v = self.to_v(x)
            
            # Split heads
            q = q.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Chunked attention computation
            x = self._chunk_attention(q, k, v, self.chunk_size)
            
            # Reshape and project
            x = x.transpose(1, 2).reshape(B, V*H*W, self.dim)
            x = self.proj(x)
            x = self.norm(x)
            
            # Reshape to spatial form
            x = x.reshape(B, V, H*W, self.dim).mean(dim=1)
            x = x.reshape(B, H, W, self.dim).permute(0, 3, 1, 2)
            
            return x
            
        except Exception as e:
            logger.error(f"Error in CrossViewAttention forward pass: {str(e)}")
            raise


class AnatomicalConstraintModule(nn.Module):
    def __init__(self, num_levels: int = 5, in_channels: int = 128):
        super().__init__()
        self.num_levels = num_levels
        
        # Shape prior
        self.shape_prior = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, num_levels, 1)
        )
        
        self.stn = SpatialTransformer(in_channels)
        self.level_projection = nn.Conv2d(in_channels, num_levels * 16, 1)
        self.feature_projection = nn.Conv2d(num_levels * 16, in_channels, 1)
        self.register_parameter('adjacency', nn.Parameter(torch.eye(num_levels)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            B = x.size(0)
            H, W = x.shape[-2:]
            
            # Generate shape attention
            shape_attention = self.shape_prior(x)
            
            # Apply spatial transform
            aligned_features = self.stn(x)
            
            # Project features
            level_features = self.level_projection(aligned_features)  # [B, num_levels*16, H, W]
            
            # Reshape with contiguous tensors
            level_features = level_features.contiguous()
            level_features = level_features.view(B, self.num_levels, 16, H, W)
            level_features = level_features.permute(0, 1, 3, 4, 2).contiguous()  # [B, num_levels, H, W, 16]
            level_features = level_features.view(B, self.num_levels, -1)  # [B, num_levels, H*W*16]
            
            # Apply adjacency
            adj_matrix = F.softmax(self.adjacency, dim=1)
            adj_matrix = adj_matrix.expand(B, -1, -1)
            constrained = torch.bmm(adj_matrix, level_features)  # [B, num_levels, H*W*16]
            
            # Restore spatial dimensions
            constrained = constrained.view(B, self.num_levels * 16, H, W)
            constrained_features = self.feature_projection(constrained)
            
            return constrained_features, shape_attention
            
        except Exception as e:
            logger.error(f"Error in AnatomicalConstraintModule forward pass: {str(e)}")
            logger.error(f"Input shape: {x.shape}")
            if 'level_features' in locals():
                logger.error(f"Level features shapes:")
                logger.error(f"- After projection: {level_features.shape}")
                logger.error(f"- Strides: {level_features.stride()}")
            if 'constrained' in locals():
                logger.error(f"Constrained shape: {constrained.shape}")
            raise

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        self.fc_loc = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 16, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            B = x.size(0)
            
            # Initial identity transformation
            theta = torch.zeros(B, 2, 3, device=x.device)
            theta[:, 0, 0] = 1
            theta[:, 1, 1] = 1
            
            # Predict transformation parameters
            xs = self.localization(x)
            theta_update = self.fc_loc(xs).view(-1, 2, 3)
            theta = theta + theta_update
            
            # Generate grid and apply transform
            size = x.size()[-2:]
            grid = F.affine_grid(theta, [B, x.size(1), *size], align_corners=True)
            transformed = F.grid_sample(x, grid, align_corners=True)
            
            return transformed
            
        except Exception as e:
            logger.error(f"Error in SpatialTransformer forward pass: {str(e)}")
            raise


class MultiViewSpineNet(nn.Module):
    def __init__(self, 
                 in_channels: int = 1,
                 num_classes: int = 4,
                 num_levels: int = 5):
        super().__init__()
        self.num_classes = num_classes
        # Reduced feature dimensions
        self.dims = [32, 64, 128]
        self.num_levels = num_levels
        
        # View-specific encoders
        self.encoders = nn.ModuleDict({
            'Sagittal T2/STIR': self._make_encoder(in_channels),
            'Sagittal T1': self._make_encoder(in_channels),
            'Axial T2': self._make_encoder(in_channels)
        })
        
        # Memory-efficient cross-attention
        self.cross_attention = nn.ModuleList([
            CrossViewAttention(dim=dim, num_heads=4, chunk_size=1024) 
            for dim in self.dims
        ])
        
        self.anatomical_module = AnatomicalConstraintModule(
            num_levels=num_levels,
            in_channels=self.dims[-1]  # Pass the final dimension
        )

        # Task-specific decoders with reduced dimensions
        self.segmentation_decoder = self._make_segmentation_decoder(self.dims[-1], num_classes)
        self.level_classifier = self._make_level_classifier()
        self.landmark_detector = self._make_landmark_detector()
        self.condition_classifier = self._make_condition_classifier()
        
        # Enable gradient checkpointing
        self.use_checkpointing = True

    def _make_encoder(self, in_channels: int) -> nn.ModuleList:
        layers = []
        current_channels = in_channels
        
        for dim in self.dims:
            block = nn.Sequential(
                ResBlock(current_channels, dim),
                ResBlock(dim, dim)
            )
            layers.append(block)
            current_channels = dim
        
        return nn.ModuleList(layers)

    def _make_segmentation_decoder(self, in_channels: int, num_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            UpBlock(in_channels, 64),
            UpBlock(64, 32),
            UpBlock(32, 16),
            nn.Conv2d(16, self.num_classes, 1)  # Use self.num_classes here
        )

    def _make_level_classifier(self) -> nn.Module:
        """Create level classifier to output correct shape"""
        return nn.Sequential(
            nn.Conv2d(self.dims[-1], 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 5 * 3)  # Output 5 levels x 3 conditions
        )


    def _make_landmark_detector(self) -> nn.Module:
        """Create landmark detector with proper dimension handling"""
        return nn.Sequential(
            # Initial convolution
            nn.Conv2d(self.dims[-1], 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Reduce spatial dimensions
            nn.AdaptiveAvgPool2d((1, 1)),
            # Flatten and project to landmarks
            nn.Flatten(),
            nn.Linear(128, self.num_levels * 2)  # 5 levels * 2 coordinates
        )

    def _make_condition_classifier(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(self.dims[-1], 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 3)
        )

    def forward(self, views: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MultiViewSpineNet.
        
        Args:
            views: Dictionary of input views, each of shape [B, C, H, W]
        
        Returns:
            Dictionary containing model outputs for all tasks
        """
        try:
            # Clear GPU memory and get input dimensions
            torch.cuda.empty_cache()
            first_view = next(iter(views.values()))
            B = first_view.size(0)
            H, W = first_view.size(-2), first_view.size(-1)
            
            # Process each view through encoders
            features = {name: [] for name in views.keys()}
            for name, view in views.items():
                x = view
                for i, encoder in enumerate(self.encoders[name]):
                    if self.use_checkpointing:
                        x = checkpoint(encoder, x)
                    else:
                        x = encoder(x)
                    features[name].append(x)
                    if i < len(self.encoders[name]) - 1:
                        x = F.avg_pool2d(x, 2)
            
            # Cross-attention fusion across views
            fused_features = []
            for level in range(len(self.cross_attention)):
                level_features = [features[name][level] for name in views.keys()]
                if self.use_checkpointing:
                    fused = checkpoint(self.cross_attention[level], level_features)
                else:
                    fused = self.cross_attention[level](level_features)
                fused_features.append(fused)
            
            # Get final features
            final_features = fused_features[-1]
            
            # Apply anatomical constraints
            if self.use_checkpointing:
                constrained_features, shape_attention = checkpoint(
                    self.anatomical_module, final_features
                )
            else:
                constrained_features, shape_attention = self.anatomical_module(final_features)
            
            # Generate segmentation outputs for each view
            segmentation_outputs = {}
            for name in views.keys():
                seg_output = self.segmentation_decoder(constrained_features)
                if seg_output.size(-2) != H or seg_output.size(-1) != W:
                    seg_output = F.interpolate(
                        seg_output, 
                        size=(H, W),
                        mode='bilinear', 
                        align_corners=True
                    )
                segmentation_outputs[name] = seg_output
            
            # Generate predictions for all tasks
            level_preds = self.level_classifier(constrained_features)
            level_preds = level_preds.view(B, 5, 3)  # Reshape to [B, 5 levels, 3 conditions]
            
            landmarks = self.landmark_detector(constrained_features)
            landmarks = landmarks.view(B, -1)  # Ensure [B, 10] shape
            
            condition_preds = self.condition_classifier(constrained_features)
            
            # Return actual tensors instead of shape specifications
            outputs = {
                'segmentation': segmentation_outputs,  # Dictionary of actual tensors
                'landmarks': landmarks,                # [B, 10] tensor
                'vertebral_levels': level_preds,       # [B, 5, 3] tensor
                'conditions': condition_preds          # [B, 3] tensor
            }
                        
            return outputs 
            
        except Exception as e:
            logger.error(f"Error in MultiViewSpineNet forward pass: {str(e)}")
            logger.error(f"Input shapes: {[(name, view.shape) for name, view in views.items()]}")
            logger.error("Output shapes at error:")
            for name, tensor in outputs.items() if 'outputs' in locals() else []:
                if isinstance(tensor, dict):
                    logger.error(f"{name}: {[(k, v.shape) for k, v in tensor.items()]}")
                else:
                    logger.error(f"{name}: {tensor.shape}")
            raise


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Memory-efficient implementation with reduced parameters
        self.bn1 = nn.BatchNorm2d(in_channels)
        # Use groups to reduce parameters while maintaining performance
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, 
                              groups=math.gcd(in_channels, out_channels))
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                              groups=4)  # Use grouped convolutions
        
        # Efficient shortcut with 1x1 convolution
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
        # Enable memory-efficient forward pass
        self.use_checkpointing = True
    
    def _forward_impl(self, x):
        identity = self.shortcut(x)
        
        out = self.bn1(x)
        out = F.relu(out, inplace=True)  # Use inplace operations
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        
        out += identity
        return out
    
    def forward(self, x):
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, x)
        return self._forward_impl(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv = nn.Sequential(
            ResBlock(out_channels, out_channels),
            ResBlock(out_channels, out_channels)
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            # Upsample
            x = self.up(x)
            x = self.bn(x)
            
            # Handle skip connection if present
            if skip is not None:
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
            
            # Process through residual blocks
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

def memory_efficient_inference(model: MultiViewSpineNet, 
                             views: Dict[str, torch.Tensor],
                             chunk_size: int = 4) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient inference for large batches
    """
    model.eval()
    B = next(iter(views.values())).size(0)
    outputs_list = []
    
    for i in range(0, B, chunk_size):
        chunk_views = {
            name: tensor[i:i+chunk_size] 
            for name, tensor in views.items()
        }
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                chunk_outputs = model(chunk_views)
                outputs_list.append(chunk_outputs)
    
    # Combine chunk outputs
    combined_outputs = {
        'segmentation': {},
        'vertebral_levels': [],
        'landmarks': [],
        'conditions': [],
        'shape_attention': []
    }
    
    for outputs in outputs_list:
        for key in outputs:
            if key == 'segmentation':
                for view in outputs[key]:
                    if view not in combined_outputs[key]:
                        combined_outputs[key][view] = []
                    combined_outputs[key][view].append(outputs[key][view])
            else:
                combined_outputs[key].append(outputs[key])
    
    # Concatenate tensors
    final_outputs = {
        'segmentation': {
            view: torch.cat(tensors) 
            for view, tensors in combined_outputs['segmentation'].items()
        },
        'vertebral_levels': torch.cat(combined_outputs['vertebral_levels']),
        'landmarks': torch.cat(combined_outputs['landmarks']),
        'conditions': torch.cat(combined_outputs['conditions']),
        'shape_attention': torch.cat(combined_outputs['shape_attention'])
    }
    
    return final_outputs

def enable_memory_efficient_mode(model: MultiViewSpineNet, enable: bool = True) -> None:
    """
    Enable or disable memory-efficient mode for the entire model
    """
    def _set_checkpointing(module):
        if hasattr(module, 'use_checkpointing'):
            module.use_checkpointing = enable
    
    model.apply(_set_checkpointing)
    model.use_checkpointing = enable
    
    # Reduce feature dimensions if needed
    if enable:
        for attention in model.cross_attention:
            attention.chunk_size = 1024  # Adjust chunk size for memory efficiency
    
    # Clear CUDA cache
    torch.cuda.empty_cache()