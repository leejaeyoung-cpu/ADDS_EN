import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dl_slice_interpolator import (
    SliceInterpolatorUNet,
    SliceInterpolatorLightweight,
    create_interpolator
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CTSliceTripletDataset(Dataset):
    """
    Dataset for training slice interpolation
    
    Creates triplets (slice_before, slice_middle, slice_after) from CT volumes.
    Network learns to predict slice_middle from (slice_before, slice_after).
    """
    
    def __init__(
        self,
        ct_volumes_dir: Path,
        normalize: bool = True,
        random_crop: bool = False,
        crop_size: Tuple[int, int] = (256, 256)
    ):
        """
        Args:
            ct_volumes_dir: Directory containing .nii.gz CT volumes
            normalize: Whether to normalize slices to [0, 1]
            random_crop: Whether to randomly crop slices
            crop_size: Target size for cropping
        """
        self.ct_volumes_dir = Path(ct_volumes_dir)
        self.normalize = normalize
        self.random_crop = random_crop
        self.crop_size = crop_size
        
        # Find all CT volumes
        self.volume_files = list(self.ct_volumes_dir.glob("*.nii.gz"))
        
        # Build triplet indices
        self.triplets = self._build_triplet_indices()
        
        logger.info(f"Found {len(self.volume_files)} CT volumes")
        logger.info(f"Generated {len(self.triplets)} training triplets")
    
    def _build_triplet_indices(self) -> List[Tuple[int, int]]:
        """Build list of (volume_idx, slice_idx) for all valid triplets"""
        triplets = []
        
        for vol_idx, vol_path in enumerate(self.volume_files):
            try:
                nii = nib.load(vol_path)
                volume = nii.get_fdata()
                num_slices = volume.shape[0]
                
                # Can only create triplets from slice 1 to n-2
                # (need one before and one after)
                for slice_idx in range(1, num_slices - 1):
                    triplets.append((vol_idx, slice_idx))
                    
            except Exception as e:
                logger.warning(f"Skipping {vol_path.name}: {e}")
        
        return triplets
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_pair: Tensor of shape (2, H, W) - two adjacent slices
            target: Tensor of shape (1, H, W) - middle slice to predict
        """
        vol_idx, slice_idx = self.triplets[idx]
        
        # Load volume
        volume_path = self.volume_files[vol_idx]
        nii = nib.load(volume_path)
        volume = nii.get_fdata()
        
        # Extract triplet
        slice_before = volume[slice_idx - 1]
        slice_middle = volume[slice_idx]
        slice_after = volume[slice_idx + 1]
        
        # Normalize to [0, 1] or [-1, 1]
        if self.normalize:
            # Window for CT (soft tissue window)
            hu_min, hu_max = -100, 400
            slice_before = np.clip((slice_before - hu_min) / (hu_max - hu_min), 0, 1)
            slice_middle = np.clip((slice_middle - hu_min) / (hu_max - hu_min), 0, 1)
            slice_after = np.clip((slice_after - hu_min) / (hu_max - hu_min), 0, 1)
        
        # Random crop
        if self.random_crop:
            h, w = slice_before.shape
            th, tw = self.crop_size
            
            if h > th and w > tw:
                top = np.random.randint(0, h - th)
                left = np.random.randint(0, w - tw)
                
                slice_before = slice_before[top:top+th, left:left+tw]
                slice_middle = slice_middle[top:top+th, left:left+tw]
                slice_after = slice_after[top:top+th, left:left+tw]
        
        # Convert to tensors
        input_pair = torch.from_numpy(
            np.stack([slice_before, slice_after], axis=0)
        ).float()
        
        target = torch.from_numpy(slice_middle[np.newaxis, ...]).float()
        
        return input_pair, target


class CombinedLoss(nn.Module):
    """Combined L1 + SSIM loss for better perceptual quality"""
    
    def __init__(self, alpha: float = 0.7):
        """
        Args:
            alpha: Weight for L1 loss (1-alpha for SSIM)
        """
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
    
    def ssim(self, x, y, window_size=11):
        """Simplified SSIM calculation"""
        # Constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Use simple average pooling
        kernel = torch.ones(1, 1, window_size, window_size).to(x.device) / (window_size ** 2)
        
        def pool2d(tensor):
            return F.conv2d(tensor, kernel, padding=window_size//2)
        
        # Mean
        mu_x = pool2d(x)
        mu_y = pool2d(y)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # Variance and covariance
        sigma_x_sq = pool2d(x ** 2) - mu_x_sq
        sigma_y_sq = pool2d(y ** 2) - mu_y_sq
        sigma_xy = pool2d(x * y) - mu_xy
        
        # SSIM
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return ssim_map.mean()
    
    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - self.ssim(pred, target)
        
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for input_pair, target in pbar:
        input_pair = input_pair.to(device)
        target = target.to(device)
        
        # Forward
        optimizer.zero_grad()
        output = model(input_pair)
        loss = criterion(output, target)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_pair, target in tqdm(dataloader, desc="Validating"):
            input_pair = input_pair.to(device)
            target = target.to(device)
            
            output = model(input_pair)
            loss = criterion(output, target)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_interpolator(
    ct_volumes_dir: Path,
    output_dir: Path,
    model_type: str = 'lightweight',
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    """
    Main training function
    
    Args:
        ct_volumes_dir: Directory with CT .nii.gz files
        output_dir: Where to save models and logs
        model_type: 'standard' or 'lightweight'
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: 'cuda' or 'cpu'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training: {model_type} model")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Device: {device}")
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = CTSliceTripletDataset(
        ct_volumes_dir,
        normalize=True,
        random_crop=True,
        crop_size=(256, 256)
    )
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_interpolator(model_type, device)
    
    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.7)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'model_type': model_type
            }, output_dir / f'best_interpolator_{model_type}.pth')
            logger.info(f"  [BEST] Model saved!")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, output_dir / f'checkpoint_epoch{epoch+1}.pth')
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"\nTraining complete!")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")
    logger.info(f"  Model saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    
    # Check for CT volumes
    ct_dir = Path("outputs/inha_ct_analysis")
    
    if not ct_dir.exists() or not list(ct_dir.glob("*.nii.gz")):
        print(f"[ERROR] No CT volumes found in {ct_dir}")
        print("Please ensure CT analysis has been run")
        sys.exit(1)
    
    # Start training
    train_interpolator(
        ct_volumes_dir=ct_dir,
        output_dir=Path("models/slice_interpolator"),
        model_type='lightweight',  # Start with lightweight for faster training
        num_epochs=30,
        batch_size=16,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
