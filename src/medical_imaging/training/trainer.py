"""
SOTA Training Pipeline - Trainer Class
Manages training loop, validation, checkpointing, and logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.warning("TensorBoard not available")

logger = logging.getLogger(__name__)


class SOTATrainer:
    """
    Trainer for SOTA medical segmentation models
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - TensorBoard logging
    - Best model tracking
    - Early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[_LRScheduler] = None,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        log_dir: str = "runs",
        checkpoint_dir: str = "models/checkpoints"
    ):
        """
        Args:
            model: Neural network model
            optimizer: Optimizer
            criterion: Loss function
            device: Device (cuda/cpu)
            scheduler: Learning rate scheduler
            use_amp: Use automatic mixed precision
            gradient_accumulation_steps: Steps for gradient accumulation
            log_dir: TensorBoard log directory
            checkpoint_dir: Checkpoint save directory
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # AMP scaler
        if self.use_amp:
            self.scaler = GradScaler()
        
        # TensorBoard
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        self.global_step = 0
        
        logger.info(f"Trainer initialized")
        logger.info(f"Device: {device}")
        logger.info(f"AMP: {use_amp}")
        logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Normalize loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track loss
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
            
            # Log to TensorBoard
            if self.writer and (batch_idx % 10 == 0):
                self.writer.add_scalar(
                    'train/batch_loss',
                    loss.item() * self.gradient_accumulation_steps,
                    self.global_step
                )
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        
        metrics = {
            'loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # Log epoch metrics
        if self.writer:
            self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)
            self.writer.add_scalar('train/learning_rate', metrics['learning_rate'], epoch)
        
        logger.info(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
        
        return metrics
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
        
        Returns:
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(val_loader)
        
        pbar = tqdm(val_loader, desc=f"Validation")
        
        for images, labels in pbar:
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Calculate Dice score
            dice = self._calculate_dice(outputs, labels)
            
            total_loss += loss.item()
            total_dice += dice
            
            pbar.set_postfix({'loss': loss.item(), 'dice': dice})
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        metrics = {
            'loss': avg_loss,
            'dice': avg_dice
        }
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
            self.writer.add_scalar('val/dice', avg_dice, epoch)
        
        logger.info(f"Epoch {epoch} - Val Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
        
        return metrics
    
    def _calculate_dice(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-5
    ) -> float:
        """
        Calculate Dice coefficient
        
        Args:
            pred: Predictions (B, C, D, H, W)
            target: Ground truth (B, D, H, W)
            smooth: Smoothing factor
        
        Returns:
            dice: Dice score
        """
        # Get predicted classes
        pred_classes = torch.argmax(pred, dim=1)
        
        # Calculate intersection and union (exclude background)
        intersection = (pred_classes * target).sum()
        union = pred_classes.sum() + target.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        return dice.item()
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: Optional[str] = None
    ):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            metrics: Metrics dictionary
            is_best: Whether this is the best model so far
            filename: Custom filename (optional)
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            epoch: Epoch number to resume from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        
        epoch = checkpoint['epoch']
        
        logger.info(f"Checkpoint loaded from epoch {epoch}")
        
        return epoch
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        start_epoch: int = 0,
        early_stopping_patience: Optional[int] = None
    ):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Total number of epochs
            start_epoch: Starting epoch (for resuming)
            early_stopping_patience: Patience for early stopping (optional)
        """
        patience_counter = 0
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                # ReduceLROnPlateau needs the metric
                if hasattr(self.scheduler, 'mode'):  # It's ReduceLROnPlateau
                    self.scheduler.step(val_metrics['dice'])
                else:  # CosineAnnealingLR or other schedulers
                    self.scheduler.step()
            
            # Track best model
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_val_dice = val_metrics['dice']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {patience_counter} epochs")
                break
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        logger.info(f"Training completed. Best Dice: {self.best_val_dice:.4f}")
        
        if self.writer:
            self.writer.close()


# Test trainer
if __name__ == "__main__":
    print("Testing SOTATrainer...")
    
    # Dummy model
    model = nn.Conv3d(1, 2, kernel_size=3, padding=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizer and loss
    from .losses import DiceCELoss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = DiceCELoss()
    
    # Trainer
    trainer = SOTATrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_amp=True
    )
    
    print("✓ Trainer initialized successfully!")
