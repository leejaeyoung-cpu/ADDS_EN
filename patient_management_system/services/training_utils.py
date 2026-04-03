"""
Training Utilities for CDSS Deep Learning

Provides PyTorch Dataset, data preprocessing, and metric calculation
for the pharmacodynamics prediction model.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import logging

logger = logging.getLogger(__name__)


class PatientMetadataDataset(Dataset):
    """
    PyTorch Dataset for patient metadata + treatment outcome pairs
    
    Each sample contains:
    - Patient tumor characteristics (volume, HU stats, radiomics)
    - Treatment information (drug cocktail)
    - Outcome (response type, survival)
    """
    
    def __init__(self, samples: List[Dict[str, Any]], feature_names: List[str] = None):
        """
        Args:
            samples: List of sample dictionaries from MetadataAggregator
            feature_names: Optional list of feature names to use
        """
        self.samples = samples
        self.feature_names = feature_names
        
        # Extract features and labels
        self.X_features, self.y_response, self.y_survival = self._parse_samples()
        
        # Normalize features
        self.scaler = StandardScaler()
        self.X_normalized = self.scaler.fit_transform(self.X_features)
    
    def _parse_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse samples into feature matrix and label arrays"""
        X_list = []
        y_response_list = []
        y_survival_list = []
        
        for sample in self.samples:
            # Extract numeric features
            features = sample.get('features', {})
            feature_vector = [
                features.get('tumor_volume_ml', 0),
                features.get('max_diameter_mm', 0),
                features.get('hu_mean', 0),
                features.get('hu_std', 0),
                # Add more features as needed
            ]
            
            # Extract outcome
            outcome = sample.get('outcome', {})
            
            # Map response type to numeric (CR=0, PR=1, SD=2, PD=3)
            response_map = {'CR': 0, 'PR': 1, 'SD': 2, 'PD': 3}
            response = response_map.get(outcome.get('response_type', 'SD'), 2)
            
            # Survival days (default to 0 if missing)
            survival = outcome.get('pfs_days', 0)
            
            X_list.append(feature_vector)
            y_response_list.append(response)
            y_survival_list.append(survival)
        
        return (
            np.array(X_list, dtype=np.float32),
            np.array(y_response_list, dtype=np.int64),
            np.array(y_survival_list, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: Normalized feature tensor
            response_label: Response type (0-3)
            survival_days: Survival in days
        """
        return (
            torch.from_numpy(self.X_normalized[idx]),
            torch.tensor(self.y_response[idx], dtype=torch.long),
            torch.tensor(self.y_survival[idx], dtype=torch.float32)
        )


def create_data_loaders(
    dataset: PatientMetadataDataset,
    train_split: float = 0.7,
    val_split: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders from dataset
    
    Args:
        dataset: PatientMetadataDataset
        train_split: Fraction for training
        val_split: Fraction for validation
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        train_loader, val_loader, test_loader
    """
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created data loaders: train={train_size}, val={val_size}, test={test_size}")
    
    return train_loader, val_loader, test_loader


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    task: str = 'classification'
) -> Dict[str, float]:
    """
    Calculate metrics for model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)
        task: 'classification' or 'regression'
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if task == 'classification':
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted'))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # AUC-ROC if probabilities provided
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
            except:
                metrics['auc_roc'] = 0.0
    
    elif task == 'regression':
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics['r2'] = float(r2_score(y_true, y_pred))
    
    return metrics


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' to minimize metric, 'max' to maximize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
                return True
        
        return False


class ModelCheckpoint:
    """Save best model during training"""
    
    def __init__(self, save_path: str, mode: str = 'min'):
        """
        Args:
            save_path: Path to save model
            mode: 'min' or 'max'
        """
        self.save_path = save_path
        self.mode = mode
        self.best_score = None
    
    def __call__(self, model: nn.Module, score: float) -> bool:
        """
        Save model if score improved
        
        Returns:
            True if model was saved
        """
        if self.best_score is None:
            self.best_score = score
            self._save_model(model)
            return True
        
        if self.mode == 'min':
            improved = score < self.best_score
        else:
            improved = score > self.best_score
        
        if improved:
            self.best_score = score
            self._save_model(model)
            logger.info(f"Model saved with score: {score:.4f}")
            return True
        
        return False
    
    def _save_model(self, model: nn.Module):
        """Save model to disk"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_score': self.best_score
        }, self.save_path)


if __name__ == "__main__":
    # Test dataset creation
    print("="*80)
    print("Training Utilities Test")
    print("="*80)
    
    # Mock data
    mock_samples = [
        {
            'features': {
                'tumor_volume_ml': 5.2,
                'max_diameter_mm': 25.0,
                'hu_mean': 45.0,
                'hu_std': 12.0
            },
            'outcome': {
                'response_type': 'PR',
                'pfs_days': 180
            }
        },
        {
            'features': {
                'tumor_volume_ml': 8.5,
                'max_diameter_mm': 35.0,
                'hu_mean': 50.0,
                'hu_std': 15.0
            },
            'outcome': {
                'response_type': 'SD',
                'pfs_days': 120
            }
        }
    ]
    
    dataset = PatientMetadataDataset(mock_samples)
    print(f"\nDataset size: {len(dataset)}")
    print(f"Feature dimension: {dataset.X_features.shape[1]}")
    
    # Test data loader
    train_loader, val_loader, test_loader = create_data_loaders(dataset, batch_size=1)
    print(f"\nLoaders created successfully")
