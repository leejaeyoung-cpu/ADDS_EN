"""
SOTA Medical Imaging Pipeline - Main Training Script
Train Swin-UNETR model on Medical Decathlon Task010_Colon
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import yaml
import argparse
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medical_imaging.data.dataset import ColonCancerDataset
from medical_imaging.models.sota_model import SOTAModelWrapper
from medical_imaging.training.trainer import SOTATrainer
from medical_imaging.training.losses import DiceCELoss, FocalDiceCELoss, ComboLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, device: torch.device) -> nn.Module:
    """Create model from config"""
    model_config = config['model']
    
    model_wrapper = SOTAModelWrapper(
        model_type=model_config['type'],
        img_size=tuple(model_config['img_size']),
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        use_pretrained=model_config['use_pretrained'],
        device=str(device)
    )
    
    return model_wrapper.model


def create_dataloaders(config: dict, fold: int) -> tuple:
    """Create train and validation dataloaders"""
    data_config = config['data']
    training_config = config['training']
    
    # Training dataset
    train_dataset = ColonCancerDataset(
        data_root=data_config['root'],
        fold=fold,
        n_folds=data_config['n_folds'],
        mode='train',
        target_spacing=tuple(data_config['target_spacing']),
        patch_size=tuple(data_config['patch_size']),
        use_augmentation=True,
        use_cache=data_config.get('use_cache', False),
        tumor_focused_sampling=data_config.get('tumor_focused_sampling', False)
    )
    
    # Validation dataset
    val_dataset = ColonCancerDataset(
        data_root=data_config['root'],
        fold=fold,
        n_folds=data_config['n_folds'],
        mode='val',
        target_spacing=tuple(data_config['target_spacing']),
        patch_size=tuple(data_config['patch_size']),
        use_augmentation=False,
        use_cache=data_config.get('use_cache', False)
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        persistent_workers=True if config.get('num_workers', 4) > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        persistent_workers=True if config.get('num_workers', 4) > 0 else False
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def create_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer from config"""
    opt_config = config['optimizer']
    training_config = config['training']
    
    # Convert to float to handle scientific notation
    lr = float(training_config['learning_rate'])
    weight_decay = float(training_config.get('weight_decay', 1e-5))
    eps = float(opt_config.get('eps', 1e-8))
    
    if opt_config['type'] == 'AdamW':
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=opt_config.get('betas', [0.9, 0.999]),
            eps=eps
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['type']}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Create learning rate scheduler"""
    scheduler_config = config['scheduler']
    
    if scheduler_config['type'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config['T_max']),
            eta_min=float(scheduler_config.get('eta_min', 1e-6))
        )
    elif scheduler_config['type'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'max'),  # 'max' for Dice
            factor=float(scheduler_config.get('factor', 0.5)),
            patience=int(scheduler_config.get('patience', 10)),
            min_lr=float(scheduler_config.get('min_lr', 1e-6))
        )
    else:
        scheduler = None
    
    return scheduler


def create_criterion(config: dict) -> nn.Module:
    """Create loss criterion"""
    loss_config = config['loss']
    
    if loss_config['type'] == 'DiceCELoss':
        criterion = DiceCELoss(
            dice_weight=loss_config.get('dice_weight', 0.5),
            ce_weight=loss_config.get('ce_weight', 0.5),
            smooth=loss_config.get('smooth', 1e-5)
        )
    elif loss_config['type'] == 'FocalDiceCELoss':
        # Focal + Dice + CE combination
        criterion = FocalDiceCELoss(
            focal_weight=loss_config.get('focal_weight', 0.4),
            dice_weight=loss_config.get('dice_weight', 0.4),
            ce_weight=loss_config.get('ce_weight', 0.2),
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            dice_smooth=loss_config.get('dice_smooth', 1e-5)
        )
        logger.info(f"Using FocalDiceCELoss:")
        logger.info(f"  - focal_weight={loss_config.get('focal_weight', 0.4)}")
        logger.info(f"  - dice_weight={loss_config.get('dice_weight', 0.4)}")
        logger.info(f"  - ce_weight={loss_config.get('ce_weight', 0.2)}")
        logger.info(f"  - focal_alpha={loss_config.get('focal_alpha', 0.25)}")
        logger.info(f"  - focal_gamma={loss_config.get('focal_gamma', 2.0)}")
    elif loss_config['type'] == 'ComboLoss':
        # NEW: 2026 SOTA ComboLoss (Dice + Focal + Boundary)
        criterion = ComboLoss(
            dice_weight=loss_config.get('dice_weight', 0.5),
            focal_weight=loss_config.get('focal_weight', 0.3),
            boundary_weight=loss_config.get('boundary_weight', 0.2),
            dice_smooth=loss_config.get('dice_smooth', 1e-5),
            focal_alpha=loss_config.get('focal_alpha', 0.25),
            focal_gamma=loss_config.get('focal_gamma', 2.0),
            boundary_strength=loss_config.get('boundary_strength', 10.0)
        )
        logger.info(f"Using ComboLoss (2026 SOTA):")
        logger.info(f"  - dice_weight={loss_config.get('dice_weight', 0.5)}")
        logger.info(f"  - focal_weight={loss_config.get('focal_weight', 0.3)}")
        logger.info(f"  - boundary_weight={loss_config.get('boundary_weight', 0.2)}")
        logger.info(f"  - focal_alpha={loss_config.get('focal_alpha', 0.25)}")
        logger.info(f"  - focal_gamma={loss_config.get('focal_gamma', 2.0)}")
        logger.info(f"  - boundary_strength={loss_config.get('boundary_strength', 10.0)}")
    else:
        raise ValueError(f"Unknown loss: {loss_config['type']}")
    
    return criterion


def main():
    parser = argparse.ArgumentParser(description='Train SOTA CT Segmentation Model')
    parser.add_argument('--config', type=str, default='configs/sota_training_config.yaml',
                        help='Path to config file')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold number for cross-validation (0-4)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Set device
    if config.get('device') == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, device)
    model = model.to(device)
    
    # Create dataloaders
    logger.info(f"Creating dataloaders for fold {args.fold}...")
    train_loader, val_loader = create_dataloaders(config, args.fold)
    
    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    logger.info("Creating scheduler...")
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss criterion
    logger.info("Creating loss criterion...")
    criterion = create_criterion(config)
    criterion = criterion.to(device)
    
    # Create trainer
    logger.info("Creating trainer...")
    logging_config = config.get('logging', {})
    
    trainer = SOTATrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        use_amp=config['training'].get('use_amp', True),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        log_dir=f"{logging_config.get('log_dir', 'runs/sota')}/fold_{args.fold}",
        checkpoint_dir=f"{logging_config.get('checkpoint_dir', 'models/sota')}/fold_{args.fold}"
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume) + 1
    
    # Training loop
    logger.info("Starting training...")
    logger.info(f"Fold: {args.fold}")
    logger.info(f"Epochs: {config['training']['num_epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        start_epoch=start_epoch,
        early_stopping_patience=config['training'].get('early_stopping_patience')
    )
    
    logger.info("Training completed!")
    logger.info(f"Best validation Dice: {trainer.best_val_dice:.4f}")


if __name__ == "__main__":
    main()
