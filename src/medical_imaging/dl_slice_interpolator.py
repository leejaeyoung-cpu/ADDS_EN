"""
Deep Learning Slice Interpolator

Neural network-based CT slice interpolation using 2D U-Net architecture.
Predicts intermediate slices from pairs of adjacent slices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DoubleConv(nn.Module):
    """Two consecutive convolution layers with instance norm and LeakyReLU"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Transposed convolution for upsampling
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Upsampled feature map
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SliceInterpolatorUNet(nn.Module):
    """
    U-Net for CT slice interpolation
    
    Takes two adjacent slices as input (2 channels) and outputs the interpolated
    middle slice (1 channel).
    """
    
    def __init__(self, base_channels: int = 64):
        super().__init__()
        
        # Input: 2 channels (two adjacent slices)
        self.inc = DoubleConv(2, base_channels)
        
        # Encoder (downsampling path)
        self.down1 = Down(base_channels, base_channels * 2)       # 64 -> 128
        self.down2 = Down(base_channels * 2, base_channels * 4)   # 128 -> 256
        self.down3 = Down(base_channels * 4, base_channels * 8)   # 256 -> 512
        
        # Bottleneck
        self.down4 = Down(base_channels * 8, base_channels * 16)  # 512 -> 1024
        
        # Decoder (upsampling path)
        self.up1 = Up(base_channels * 16, base_channels * 8)      # 1024 -> 512
        self.up2 = Up(base_channels * 8, base_channels * 4)       # 512 -> 256
        self.up3 = Up(base_channels * 4, base_channels * 2)       # 256 -> 128
        self.up4 = Up(base_channels * 2, base_channels)           # 128 -> 64
        
        # Output: 1 channel (interpolated slice)
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 2, H, W)
               Channel 0: Slice before
               Channel 1: Slice after
        
        Returns:
            Interpolated slice of shape (B, 1, H, W)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        output = self.outc(x)
        
        return output
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SliceInterpolatorLightweight(nn.Module):
    """
    Lightweight version for faster training and inference
    
    Reduced depth and channels for resource-constrained environments.
    """
    
    def __init__(self, base_channels: int = 32):
        super().__init__()
        
        self.inc = DoubleConv(2, base_channels)
        
        # Shallower encoder
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        
        # Decoder
        self.up1 = Up(base_channels * 8, base_channels * 4)
        self.up2 = Up(base_channels * 4, base_channels * 2)
        self.up3 = Up(base_channels * 2, base_channels)
        
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        output = self.outc(x)
        return output
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_interpolator(model_type: str = 'standard', device: str = 'cuda') -> nn.Module:
    """
    Factory function to create interpolator model
    
    Args:
        model_type: 'standard' or 'lightweight'
        device: 'cuda' or 'cpu'
    
    Returns:
        Initialized model on specified device
    """
    if model_type == 'standard':
        model = SliceInterpolatorUNet(base_channels=64)
    elif model_type == 'lightweight':
        model = SliceInterpolatorLightweight(base_channels=32)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    print(f"Created {model_type} interpolator")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {device}")
    
    return model


if __name__ == "__main__":
    # Test model
    print("="*80)
    print("Slice Interpolator Model Test")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create models
    print("\n--- Standard Model ---")
    model_std = create_interpolator('standard', device)
    
    print("\n--- Lightweight Model ---")
    model_lite = create_interpolator('lightweight', device)
    
    # Test forward pass
    print("\n--- Forward Pass Test ---")
    batch_size = 4
    height, width = 256, 256
    
    # Create dummy input (2 adjacent slices)
    x = torch.randn(batch_size, 2, height, width).to(device)
    print(f"Input shape: {x.shape}")
    
    # Standard model
    with torch.no_grad():
        output_std = model_std(x)
    print(f"Standard output shape: {output_std.shape}")
    
    # Lightweight model
    with torch.no_grad():
        output_lite = model_lite(x)
    print(f"Lightweight output shape: {output_lite.shape}")
    
    # Memory usage
    if device == 'cuda':
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    print("\n" + "="*80)
    print("✓ Model test passed!")
    print("="*80)
