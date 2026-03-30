"""
Detection Network.

Lightweight Encoder-Decoder detection network based on MSHNet architecture.
Takes fused features as input and outputs detection mask.

Architecture:
    Input: [B, C, H, W] fused features (default C=64)
    Encoder: 4 downsampling layers (H → H/16)
    Decoder: 4 upsampling layers (H/16 → H)
    Output: [B, 1, H, W] detection mask
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention module (CBAM)."""
    
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention module (CBAM)."""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResBlock(nn.Module):
    """Residual block with CBAM attention."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        out += residual
        out = self.relu(out)
        return out


class DetectionNetwork(nn.Module):
    """
    Detection Network.
    
    Lightweight Encoder-Decoder based on MSHNet architecture.
    
    Args:
        input_channels: Input feature channels (default: 64)
        deep_supervision: Enable deep supervision (default: False)
        block: Basic residual block type (default: ResBlock)
    """
    
    def __init__(self, input_channels=64, deep_supervision=False, block=ResBlock):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)

        # Encoder
        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])
     
        # Middle Layer
        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])
        
        # Decoder
        self.decoder_3 = self._make_layer(param_channels[3] + param_channels[4], param_channels[3], block, param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2] + param_channels[3], param_channels[2], block, param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1] + param_channels[2], param_channels[1], block, param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0] + param_channels[1], param_channels[0], block)

        # Output Layers
        if self.deep_supervision:
            self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
            self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
            self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
            self.output_3 = nn.Conv2d(param_channels[3], 1, 1)
            self.final = nn.Conv2d(4, 1, 3, 1, 1)
        else:
            self.final = nn.Conv2d(param_channels[0], 1, 1)

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layers = []
        layers.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] fused features
            
        Returns:
            Deep supervision: List[mask0, mask1, mask2, mask3, output]
            Single output: output [B, 1, H, W]
        """
        # Encoding Path
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        # Middle Layer
        x_m = self.middle_layer(self.pool(x_e3))

        # Decoding Path
        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))
        
        # Output
        if self.deep_supervision:
            mask0 = self.output_0(x_d0).sigmoid()
            mask1 = self.up(self.output_1(x_d1)).sigmoid()
            mask2 = self.up_4(self.output_2(x_d2)).sigmoid()
            mask3 = self.up_8(self.output_3(x_d3)).sigmoid()
            output = self.final(torch.cat([mask0, mask1, mask2, mask3], dim=1)).sigmoid()
            return [mask0, mask1, mask2, mask3, output]
        else:
            output = self.final(x_d0).sigmoid()
            return output
