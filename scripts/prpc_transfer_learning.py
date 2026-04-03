#!/usr/bin/env python3
"""
ADDS Transfer Learning for PrPc Prediction
Phase 2 of v3.0 AI-First Computational Discovery

Leverages pre-trained ADDS models (Cellpose, CT analysis) 
for PrPc biomarker prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATA_DIR = Path("data/analysis/prpc_validation")
MODEL_DIR = Path("models/prpc_predictor")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class ADDSEncoder(nn.Module):
    """
    Simplified version of ADDS encoder
    In production, load actual ADDS Swin-UNETR weights
    """
    def __init__(self, input_dim=768, hidden_dims=[512, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        return self.encoder(x)


class MultiModalPrPcPredictor(nn.Module):
    """
    Multi-modal AI model for PrPc prediction
    
    Inputs:
    - Genomic features (PRNP + cancer genes)
    - Clinical features
    - Imaging features (from ADDS encoder)
    - Proteomics (if available)
    
    Outputs:
    - PrPc level prediction (regression)
    - Cancer classification (multi-class)
    - Risk score
    """
    
    def __init__(self, 
                 genomic_dim=100, 
                 clinical_dim=10,
                 imaging_dim=768,
                 proteomics_dim=50,
                 use_imaging=False,
                 use_proteomics=False):
        
        super().__init__()
        
        self.use_imaging = use_imaging
        self.use_proteomics = use_proteomics
        
        # Genomic pathway encoder
        self.genomic_encoder = nn.Sequential(
            nn.Linear(genomic_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Clinical pathway encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Imaging pathway encoder (ADDS transfer)
        if use_imaging:
            self.imaging_encoder = ADDSEncoder(
                input_dim=imaging_dim,
                hidden_dims=[512, 256, 128]
            )
            imaging_output_dim = 128
        else:
            imaging_output_dim = 0
        
        # Proteomics pathway encoder
        if use_proteomics:
            self.proteomics_encoder = nn.Sequential(
                nn.Linear(proteomics_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            proteomics_output_dim = 64
        else:
            proteomics_output_dim = 0
        
        # Fusion layer - combine all modalities
        fusion_input_dim = 128 + 32 + imaging_output_dim + proteomics_output_dim
        
        # Multi-head attention for cross-modality
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_input_dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        # Shared representation
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Task-specific heads
        
        # 1. PrPc level prediction (regression)
        self.prpc_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # 2. Cancer classification (4 classes: Cancer, Inflammation, Cell Death, Healthy)
        self.cancer_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
        
        # 3. Risk score (binary)
        self.risk_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, genomic, clinical, imaging=None, proteomics=None):
        """
        Forward pass through multi-modal network
        
        Args:
            genomic: (batch, genomic_dim)
            clinical: (batch, clinical_dim)
            imaging: (batch, imaging_dim) - optional
            proteomics: (batch, proteomics_dim) - optional
        
        Returns:
            prpc_pred: PrPc level prediction
            cancer_class: Cancer classification logits
            risk_score: Risk score (0-1)
        """
        
        # Encode each modality
        genomic_emb = self.genomic_encoder(genomic)
        clinical_emb = self.clinical_encoder(clinical)
        
        embeddings = [genomic_emb, clinical_emb]
        
        if self.use_imaging and imaging is not None:
            imaging_emb = self.imaging_encoder(imaging)
            embeddings.append(imaging_emb)
        
        if self.use_proteomics and proteomics is not None:
            proteomics_emb = self.proteomics_encoder(proteomics)
            embeddings.append(proteomics_emb)
        
        # Concatenate all modalities
        combined = torch.cat(embeddings, dim=1)
        
        # Self-attention (cross-modality interaction)
        # Reshape for attention: (batch, seq_len=1, features)
        combined_seq = combined.unsqueeze(1)
        attn_out, _ = self.attention(combined_seq, combined_seq, combined_seq)
        attn_out = attn_out.squeeze(1)
        
        # Residual connection
        combined = combined + attn_out
        
        # Fusion
        shared_repr = self.fusion(combined)
        
        # Task-specific predictions
        prpc_pred = self.prpc_head(shared_repr)
        cancer_class = self.cancer_classifier(shared_repr)
        risk_score = self.risk_head(shared_repr)
        
        return prpc_pred, cancer_class, risk_score


class PrPcTrainer:
    """Training pipeline for PrPc predictor"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Multi-task loss weights
        self.loss_weights = {
            'prpc': 1.0,
            'cancer': 0.5,
            'risk': 0.3
        }
        
        # Optimizers
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': []
        }
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            genomic = batch['genomic'].to(self.device)
            clinical = batch['clinical'].to(self.device)
            
            # Targets
            prpc_target = batch['prpc'].to(self.device)
            cancer_target = batch['cancer_class'].to(self.device)
            risk_target = batch['risk'].to(self.device)
            
            # Forward
            prpc_pred, cancer_pred, risk_pred = self.model(genomic, clinical)
            
            # Multi-task loss
            loss_prpc = self.mse_loss(prpc_pred.squeeze(), prpc_target)
            loss_cancer = self.ce_loss(cancer_pred, cancer_target)
            loss_risk = self.bce_loss(risk_pred.squeeze(), risk_target)
            
            loss = (self.loss_weights['prpc'] * loss_prpc +
                   self.loss_weights['cancer'] * loss_cancer +
                   self.loss_weights['risk'] * loss_risk)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        all_risk_preds = []
        all_risk_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                genomic = batch['genomic'].to(self.device)
                clinical = batch['clinical'].to(self.device)
                
                prpc_target = batch['prpc'].to(self.device)
                cancer_target = batch['cancer_class'].to(self.device)
                risk_target = batch['risk'].to(self.device)
                
                prpc_pred, cancer_pred, risk_pred = self.model(genomic, clinical)
                
                loss_prpc = self.mse_loss(prpc_pred.squeeze(), prpc_target)
                loss_cancer = self.ce_loss(cancer_pred, cancer_target)
                loss_risk = self.bce_loss(risk_pred.squeeze(), risk_target)
                
                loss = (self.loss_weights['prpc'] * loss_prpc +
                       self.loss_weights['cancer'] * loss_cancer +
                       self.loss_weights['risk'] * loss_risk)
                
                total_loss += loss.item()
                
                # Collect for AUC
                all_risk_preds.extend(risk_pred.cpu().numpy())
                all_risk_targets.extend(risk_target.cpu().numpy())
        
        # Calculate AUC
        val_auc = roc_auc_score(all_risk_targets, all_risk_preds)
        
        return total_loss / len(val_loader), val_auc
    
    def fit(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        """Full training loop"""
        print("=" * 80)
        print("🚀 Training Multi-Modal PrPc Predictor")
        print("=" * 80)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_auc = self.validate(val_loader)
            
            # Record
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                }, MODEL_DIR / 'best_prpc_model.pth')
                
                print(f"   ✓ Best model saved! (AUC: {val_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                    break
        
        print("\n" + "=" * 80)
        print("✅ Training Complete!")
        print(f"   Best Val AUC: {max(self.history['val_auc']):.4f}")
        print("=" * 80)
        
        # Plot training history
        self.plot_history()
        
        return self.history
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC
        axes[1].plot(self.history['val_auc'], label='Val AUC', color='green')
        axes[1].axhline(y=0.75, color='r', linestyle='--', label='Target (0.75)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(MODEL_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Training history saved: {MODEL_DIR / 'training_history.png'}")


def load_unified_data():
    """Load harmonized 10K cohort"""
    print("\n📂 Loading unified cohort...")
    
    data_file = DATA_DIR / "unified_cohort_harmonized.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Unified cohort not found: {data_file}")
    
    df = pd.read_csv(data_file)
    print(f"   ✓ Loaded {len(df)} samples")
    
    return df


def prepare_features(df):
    """Prepare features for ML"""
    print("\n🔧 Preparing features...")
    
    # For this demo, create synthetic genomic and clinical features
    # In production, extract from integrated databases
    
    n_samples = len(df)
    
    # Genomic features (100-dim: PRNP + 99 cancer genes)
    genomic = np.random.randn(n_samples, 100)
    genomic[:, 0] = df['PRNP_value'].values  # First feature is PRNP itself
    
    # Clinical features (10-dim: age, sex, stage, etc.)
    clinical = np.random.randn(n_samples, 10)
    
    # Targets
    prpc_target = df['PRNP_zscore'].values
    
    # Cancer classification (4 classes)
    cancer_mapping = {'Colorectal': 0, 'Pancreatic': 1, 'Breast': 2, 'Other': 3}
    cancer_class = df['cancer_group'].map(cancer_mapping).fillna(3).astype(int).values
    
    # Risk score (binary: high PRNP = 1)
    risk_target = df['PRNP_high'].values.astype(float)
    
    print(f"   ✓ Features prepared")
    print(f"      Genomic: {genomic.shape}")
    print(f"      Clinical: {clinical.shape}")
    print(f"      Targets: PrPc={len(prpc_target)}, Cancer={len(cancer_class)}, Risk={len(risk_target)}")
    
    return genomic, clinical, prpc_target, cancer_class, risk_target


def main():
    """Main execution"""
    
    # Load data
    df = load_unified_data()
    
    # Prepare features
    genomic, clinical, prpc_target, cancer_class, risk_target = prepare_features(df)
    
    # Train/val split
    indices = np.arange(len(genomic))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=cancer_class,
        random_state=42
    )
    
    print(f"\n✂️  Data split:")
    print(f"   Train: {len(train_idx)} samples")
    print(f"   Val: {len(val_idx)} samples")
    
    # Create model
    model = MultiModalPrPcPredictor(
        genomic_dim=100,
        clinical_dim=10,
        use_imaging=False,  # No imaging data in public databases
        use_proteomics=False  # Can enable if CPTAC data available
    )
    
    print(f"\n🤖 Model created:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create data loaders (simplified - in production use PyTorch Dataset/DataLoader)
    # For now, just demonstrate the architecture
    
    print("\n✅ Model architecture ready!")
    print(f"   Next: Implement DataLoader and run training")
    print(f"   Target AUC: > 0.75")
    
    return model


if __name__ == "__main__":
    model = main()
