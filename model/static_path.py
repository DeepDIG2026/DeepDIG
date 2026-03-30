"""
Static Path.

Extracts static spatial features from current frame.
Also provides descriptor extraction for DBA alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNet_CBAM import DoubleConv, Down, Up_CBAM, OutConv


class XFeatDescriptorHead(nn.Module):
    """Multi-scale feature fusion for descriptor extraction."""
    
    def __init__(self):
        super().__init__()
        self.reduce2 = nn.Conv2d(128, 64, 1)
        self.reduce3 = nn.Conv2d(256, 64, 1)
        self.reduce4 = nn.Conv2d(512, 64, 1)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, padding=0)
        )
    
    def forward(self, x2, x3, x4):
        """
        Args:
            x2: [B, 128, H/8, W/8]
            x3: [B, 256, H/16, W/16]
            x4: [B, 512, H/32, W/32]
        Returns:
            [B, 64, H/8, W/8]
        """
        p2 = self.reduce2(x2)
        p3 = F.interpolate(self.reduce3(x3), scale_factor=2, mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.reduce4(x4), scale_factor=4, mode='bilinear', align_corners=False)
        return self.fusion(p2 + p3 + p4)


class XFeatKeypointHead(nn.Module):
    """Keypoint detection head with 8x8 grid classification."""
    
    def __init__(self, in_channels=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1, padding=0, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, padding=0, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, padding=0, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 65, 1)
        )
    
    def _unfold2d(self, x, ws=8):
        B, C, H, W = x.shape
        x = x.unfold(2, ws, ws).unfold(3, ws, ws)
        x = x.reshape(B, C, H//ws, W//ws, ws**2)
        return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)
    
    def forward(self, x):
        x_unfold = self._unfold2d(x, ws=8)
        return self.head(x_unfold)


class XFeatHeatmapHead(nn.Module):
    """Keypoint reliability heatmap head."""
    
    def __init__(self, in_channels=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1, padding=0, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 1, padding=0, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.head(x)


class DualTaskUNet(nn.Module):
    """
    Dual-task UNet for detection and descriptor extraction.
    
    Encoder: Progressive 2x downsampling to H/32
    Decoder: Transposed conv upsampling with CBAM attention
    XFeat branch: Descriptor extraction at H/8 resolution
    
    Args:
        n_channels: Input channels (default: 1)
        detection_channels: Detection output channels (default: 64)
        bilinear: Use bilinear upsampling (default: False)
        enable_xfeat: Enable XFeat branch (default: True)
    """
    
    def __init__(self, n_channels=1, detection_channels=64, bilinear=False, enable_xfeat=True):
        super().__init__()
        self.n_channels = n_channels
        self.detection_channels = detection_channels
        self.bilinear = bilinear
        self.enable_xfeat = enable_xfeat
        
        # Encoder
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32, kernel_size=2)
        self.down2 = Down(32, 64, kernel_size=2)
        self.down3 = Down(64, 128, kernel_size=2)
        self.down4 = Down(128, 256, kernel_size=2)
        self.down5 = Down(256, 512, kernel_size=2)
        
        # Decoder with CBAM
        self.up1 = Up_CBAM(512 + 256, 256, bilinear, x1_channels=512)
        self.up2 = Up_CBAM(256 + 128, 128, bilinear, x1_channels=256)
        self.up3 = Up_CBAM(128 + 64, 64, bilinear, x1_channels=128)
        self.up4 = Up_CBAM(64 + 32, 32, bilinear, x1_channels=64)
        self.up5 = Up_CBAM(32 + 16, 16, bilinear, x1_channels=32)
        self.outc = OutConv(16, detection_channels, activation=True)
        
        # Multi-scale detection head
        self.detection_mask_head = nn.ModuleDict({
            'upsample_d3': nn.Sequential(
                nn.ConvTranspose2d(64, 48, kernel_size=2, stride=2),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            'upsample_d4': nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            'reduce_d5': nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            'fusion': nn.Sequential(
                nn.Conv2d(32 + 32 + 32 + detection_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, detection_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(detection_channels),
                nn.ReLU(inplace=True)
            ),
            'mask_proj': nn.Conv2d(detection_channels, 1, kernel_size=1)
        })
        
        # XFeat branch
        if enable_xfeat:
            self.descriptor_head = XFeatDescriptorHead()
            self.keypoint_head = XFeatKeypointHead(in_channels=64)
            self.heatmap_head = XFeatHeatmapHead(in_channels=64)
            
            self.fine_matcher = nn.Sequential(
                nn.Linear(128, 512),
                nn.BatchNorm1d(512, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(512, 64),
            )
            
            self.norm = nn.InstanceNorm2d(n_channels)
    
    def forward(self, x, mode='detection', return_features=False):
        """
        Args:
            x: [B, 1, H, W] input image
            mode: 'detection' | 'xfeat' | 'both'
            return_features: Return intermediate features
        
        Returns:
            mode='detection': [B, C, H, W] detection features
            mode='xfeat': dict with descriptors, keypoints, heatmap
            mode='both': dict with all outputs
        """
        if mode in ['xfeat', 'both'] and self.enable_xfeat:
            x = x.mean(dim=1, keepdim=True) if x.shape[1] > 1 else x
            x = self.norm(x)
        
        x_input = x
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        
        # Decoder
        if mode in ['detection', 'both']:
            d1 = self.up1(x6, x5)
            d2 = self.up2(d1, x4)
            d3 = self.up3(d2, x3)
            d4 = self.up4(d3, x2)
            d5 = self.up5(d4, x1)
            detection_features = self.outc(d5)
        
        # XFeat branch
        if mode in ['xfeat', 'both'] and self.enable_xfeat:
            descriptors = self.descriptor_head(x4, x5, x6)
            keypoints = self.keypoint_head(x_input)
            heatmap = self.heatmap_head(descriptors)
        
        # Return
        if mode == 'detection':
            if return_features:
                return detection_features, {
                    'enc1': x1, 'enc2': x2, 'enc3': x3, 'enc4': x4, 'enc5': x5, 'enc6': x6,
                    'dec1': d1, 'dec2': d2, 'dec3': d3, 'dec4': d4, 'dec5': d5
                }
            return detection_features
        
        elif mode == 'xfeat':
            return {
                'descriptors': descriptors,
                'keypoints': keypoints,
                'heatmap': heatmap
            }
        
        elif mode == 'both':
            feat_d3_up = self.detection_mask_head['upsample_d3'](d3)
            feat_d4_up = self.detection_mask_head['upsample_d4'](d4)
            feat_d5_proc = self.detection_mask_head['reduce_d5'](d5)
            
            multi_scale_feat = torch.cat([
                feat_d3_up, feat_d4_up, feat_d5_proc, detection_features
            ], dim=1)
            
            enhanced_static_features = self.detection_mask_head['fusion'](multi_scale_feat)
            detection_mask = self.detection_mask_head['mask_proj'](enhanced_static_features)
            
            outputs = {
                'detection_features': enhanced_static_features,
                'detection_mask': detection_mask,
                'descriptors': descriptors,
                'keypoints': keypoints,
                'heatmap': heatmap
            }
            if return_features:
                outputs['intermediate_features'] = {
                    'enc1': x1, 'enc2': x2, 'enc3': x3, 'enc4': x4, 'enc5': x5,
                    'dec1': d1, 'dec2': d2, 'dec3': d3, 'dec4': d4
                }
            return outputs
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def extract_descriptors(self, image, threshold=0.1, top_k=1024):
        """
        Extract descriptors and keypoints for DBA.
        
        Args:
            image: [B, 1, H, W]
            threshold: Keypoint threshold
            top_k: Maximum keypoints
        
        Returns:
            dict with 'descriptors', 'keypoints', 'scores', 'heatmap'
        """
        with torch.no_grad():
            outputs = self(image, mode='xfeat')
            
            heatmap = outputs['heatmap']
            descriptors = outputs['descriptors']
            
            B, C, H, W = heatmap.shape
            heatmap_flat = heatmap.view(B, -1)
            
            keypoints_list = []
            scores_list = []
            
            for b in range(B):
                mask = heatmap_flat[b] > threshold
                indices = torch.nonzero(mask).squeeze(1)
                
                if len(indices) == 0:
                    keypoints_list.append(torch.zeros(0, 2, device=image.device))
                    scores_list.append(torch.zeros(0, device=image.device))
                    continue
                
                scores_all = heatmap_flat[b, indices]
                if len(scores_all) > top_k:
                    topk_scores, topk_idx = torch.topk(scores_all, top_k)
                    indices = indices[topk_idx]
                    scores_all = topk_scores
                
                y_coords = (indices // W).float() * 8.0 + 4.0
                x_coords = (indices % W).float() * 8.0 + 4.0
                keypoints = torch.stack([x_coords, y_coords], dim=1)
                
                keypoints_list.append(keypoints)
                scores_list.append(scores_all)
            
            return {
                'descriptors': descriptors,
                'keypoints': keypoints_list,
                'scores': scores_list,
                'heatmap': heatmap
            }


class StaticPath(nn.Module):
    """
    Static feature extraction branch.
    
    Uses DualTaskUNet for detection features and XFeat descriptors.
    
    Args:
        detection_channels: Output feature channels (default: 64)
        enable_xfeat: Enable descriptor extraction (default: True)
    """
    
    def __init__(self, detection_channels=64, enable_xfeat=True):
        super().__init__()
        
        self.detection_channels = detection_channels
        self.enable_xfeat = enable_xfeat
        
        self.unet = DualTaskUNet(
            n_channels=1,
            detection_channels=detection_channels,
            bilinear=True,
            enable_xfeat=enable_xfeat
        )
    
    def forward(self, x_curr, mode='detection'):
        """
        Args:
            x_curr: [B, 1, H, W] current frame
            mode: 'detection' | 'xfeat' | 'both'
        
        Returns:
            mode='detection': [B, C, H, W] static features
            mode='xfeat': dict with descriptors, keypoints, heatmap
            mode='both': dict with both outputs
        """
        return self.unet(x_curr, mode=mode)
    
    def extract_descriptors(self, image, threshold=0.1, top_k=1024):
        """Extract descriptors and keypoints for DBA."""
        return self.unet.extract_descriptors(image, threshold=threshold, top_k=top_k)


__all__ = ['StaticPath', 'DualTaskUNet']
