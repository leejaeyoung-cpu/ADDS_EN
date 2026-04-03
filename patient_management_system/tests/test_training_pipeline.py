"""
Test suite for CDSS training pipeline

Run with: pytest patient_management_system/tests/test_training_pipeline.py -v
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from patient_management_system.services.training_utils import (
    PatientMetadataDataset,
    create_data_loaders,
    calculate_metrics,
    EarlyStopping,
    ModelCheckpoint
)
from patient_management_system.services.daily_ml_trainer import (
    PharmacodynamicsPredictor,
    DailyMLTrainer
)


class TestPatientDataset:
    """Test PyTorch Dataset creation"""
    
    @pytest.fixture
    def mock_samples(self):
        """Create mock patient samples"""
        return [
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
            },
            {
                'features': {
                    'tumor_volume_ml': 3.1,
                    'max_diameter_mm': 18.0,
                    'hu_mean': 42.0,
                    'hu_std': 10.0
                },
                'outcome': {
                    'response_type': 'CR',
                    'pfs_days': 365
                }
            }
        ]
    
    def test_dataset_creation(self, mock_samples):
        """Test dataset initialization"""
        dataset = PatientMetadataDataset(mock_samples)
        
        assert len(dataset) == 3
        assert dataset.X_features.shape == (3, 4)  # 3 samples, 4 features
        assert dataset.y_response.shape == (3,)
        assert dataset.y_survival.shape == (3,)
        
        print(f"✓ Dataset created: {len(dataset)} samples")
    
    def test_dataset_getitem(self, mock_samples):
        """Test __getitem__ method"""
        dataset = PatientMetadataDataset(mock_samples)
        
        features, response, survival = dataset[0]
        
        assert isinstance(features, torch.Tensor)
        assert isinstance(response, torch.Tensor)
        assert isinstance(survival, torch.Tensor)
        assert features.shape == (4,)
        
        print(f"✓ Sample retrieval works correctly")


class TestDataLoaders:
    """Test data loader creation"""
    
    def test_create_loaders(self):
        """Test creating train/val/test loaders"""
        # Create larger mock dataset
        samples = [
            {
                'features': {'tumor_volume_ml': i, 'max_diameter_mm': i*2, 
                            'hu_mean': 45, 'hu_std': 10},
                'outcome': {'response_type': 'PR', 'pfs_days': 150}
            }
            for i in range(100)
        ]
        
        dataset = PatientMetadataDataset(samples)
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset, train_split=0.7, val_split=0.15, batch_size=16
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
        
        print(f"✓ Data loaders created successfully")


class TestPharmacodynamicsModel:
    """Test the neural network model"""
    
    def test_model_forward_pass(self):
        """Test model forward propagation"""
        model = PharmacodynamicsPredictor(input_dim=10, hidden_dim=64)
        
        # Create dummy input
        x = torch.randn(8, 10)  # Batch of 8, 10 features
        
        response_pred, survival_pred = model(x)
        
        assert response_pred.shape == (8, 4)  # 4 response classes
        assert survival_pred.shape == (8, 1)  # 1 survival value
        
        print(f"✓ Model forward pass: {response_pred.shape}, {survival_pred.shape}")
    
    def test_model_training_step(self):
        """Test one training step"""
        model = PharmacodynamicsPredictor(input_dim=10, hidden_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        criterion_response = torch.nn.CrossEntropyLoss()
        criterion_survival = torch.nn.MSELoss()
        
        # Dummy batch
        x = torch.randn(4, 10)
        y_response = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        y_survival = torch.tensor([[150.0], [200.0], [100.0], [180.0]])
        
        # Forward
        response_pred, survival_pred = model(x)
        
        # Loss
        loss_response = criterion_response(response_pred, y_response)
        loss_survival = criterion_survival(survival_pred, y_survival)
        loss = loss_response + 0.5 * loss_survival
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        print(f"✓ Training step completed, loss: {loss.item():.4f}")


class TestTrainingUtilities:
    """Test utility functions"""
    
    def test_calculate_metrics(self):
        """Test metric calculation"""
        y_true = np.array([0, 1, 2, 1, 0, 2])
        y_pred = np.array([0, 1, 2, 2, 0, 2])
        
        metrics = calculate_metrics(y_true, y_pred, task='classification')
        
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert metrics['accuracy'] > 0.5
        
        print(f"✓ Metrics calculated: accuracy={metrics['accuracy']:.3f}")
    
    def test_early_stopping(self):
        """Test early stopping logic"""
        early_stop = EarlyStopping(patience=3, mode='min')
        
        # Simulate improving losses
        assert early_stop(1.0) == False
        assert early_stop(0.9) == False
        assert early_stop(0.8) == False
        
        # Simulate plateau
        assert early_stop(0.85) == False
        assert early_stop(0.86) == False
        assert early_stop(0.87) == False  # 3rd non-improvement
        assert early_stop.early_stop == True
        
        print(f"✓ Early stopping works correctly")
    
    def test_model_checkpoint(self):
        """Test model checkpointing"""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint = ModelCheckpoint(f.name, mode='min')
            
            # Create dummy model
            model = PharmacodynamicsPredictor(input_dim=5, hidden_dim=32)
            
            # Simulate improving scores
            assert checkpoint(model, 1.0) == True  # First save
            assert checkpoint(model, 0.9) == True  # Improvement
            assert checkpoint(model, 0.95) == False  # No improvement
            
            # Verify file exists
            assert Path(f.name).exists()
            Path(f.name).unlink()  # Cleanup
        
        print(f"✓ Model checkpointing works correctly")


class TestDailyMLTrainer:
    """Test daily training workflow"""
    
    def test_trainer_init(self):
        """Test trainer initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DailyMLTrainer(model_dir=Path(tmpdir))
            
            assert trainer.model_dir.exists()
            assert trainer.aggregator is not None
        
        print(f"✓ Trainer initialized successfully")


if __name__ == "__main__":
    print("="*80)
    print("CDSS Training Pipeline Test Suite")
    print("="*80)
    
    pytest.main([__file__, "-v", "-s"])
