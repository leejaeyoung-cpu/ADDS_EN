"""
Attention-based model for drug combination scoring
Uses Transformer architecture to model drug combinations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from gpu_init import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

try:
    from utils import get_logger, config
except ImportError:
    from src.utils import get_logger, config

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:x.size(0), :]


class DrugAttentionEncoder(nn.Module):
    """
    Attention-based encoder for drug features
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        """
        Initialize attention encoder
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super(DrugAttentionEncoder, self).__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"✓ DrugAttentionEncoder initialized with {num_layers} layers")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, seq_len, input_dim)
            mask: Optional attention mask
        
        Returns:
            Encoded features (batch_size, seq_len, d_model)
        """
        # Project to model dimension
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Transformer encoding
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        return output


class CombinationAttentionModel(nn.Module):
    """
    Full attention-based model for drug combination prediction
    """
    
    def __init__(
        self,
        drug_feature_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_additional_features: int = 0
    ):
        """
        Initialize combination attention model
        
        Args:
            drug_feature_dim: Dimension of drug features
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            num_additional_features: Additional features (concentrations, etc.)
        """
        super(CombinationAttentionModel, self).__init__()
        
        # Shared encoder for drugs
        self.drug_encoder = DrugAttentionEncoder(
            input_dim=drug_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Cross-attention between drug A and drug B
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Combination head
        combined_dim = d_model * 2 + num_additional_features
        self.combination_head = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.num_additional_features = num_additional_features
        logger.info("✓ CombinationAttentionModel initialized")
    
    def forward(
        self,
        drug_a_features: torch.Tensor,
        drug_b_features: torch.Tensor,
        additional_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            drug_a_features: Features for drug A (batch_size, seq_len, feature_dim)
            drug_b_features: Features for drug B (batch_size, seq_len, feature_dim)
            additional_features: Additional features (batch_size, num_features)
        
        Returns:
            Predicted efficacy (batch_size,)
        """
        # Encode both drugs
        encoded_a = self.drug_encoder(drug_a_features)  # (batch, seq_len, d_model)
        encoded_b = self.drug_encoder(drug_b_features)
        
        # Cross-attention: how drug A attends to drug B
        attn_output, _ = self.cross_attention(
            query=encoded_a,
            key=encoded_b,
            value=encoded_b
        )
        
        # Pool encoded representations
        pooled_a = encoded_a.mean(dim=1)  # (batch, d_model)
        pooled_b = attn_output.mean(dim=1)
        
        # Combine
        combined = torch.cat([pooled_a, pooled_b], dim=1)
        
        # Add additional features
        if additional_features is not None:
            combined = torch.cat([combined, additional_features], dim=1)
        elif self.num_additional_features > 0:
            raise ValueError("Model expects additional features")
        
        # Predict
        efficacy = self.combination_head(combined)
        
        return efficacy.squeeze(-1)


class SelfAttentionPooling(nn.Module):
    """
    Self-attention pooling for feature aggregation
    """
    
    def __init__(self, input_dim: int):
        super(SelfAttentionPooling, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention pooling
        
        Args:
            x: Input (batch_size, seq_len, input_dim)
        
        Returns:
            Pooled output (batch_size, input_dim)
        """
        # Compute attention weights
        attn_weights = F.softmax(self.attention(x), dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum
        pooled = (x * attn_weights).sum(dim=1)  # (batch, input_dim)
        
        return pooled


class MultiModalFusionModel(nn.Module):
    """
    Fuse multiple data modalities (images, text, numerical) for prediction
    """
    
    def __init__(
        self,
        image_features_dim: int = 512,
        text_features_dim: int = 768,
        numerical_features_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize multimodal fusion model
        
        Args:
            image_features_dim: Dimension of image features
            text_features_dim: Dimension of text features
            numerical_features_dim: Dimension of numerical features
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super(MultiModalFusionModel, self).__init__()
        
        # Modality-specific projections
        self.image_projection = nn.Sequential(
            nn.Linear(image_features_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_features_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.numerical_projection = nn.Sequential(
            nn.Linear(numerical_features_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        logger.info("✓ MultiModalFusionModel initialized")
    
    def forward(
        self,
        image_features: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        numerical_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with multimodal fusion
        
        Args:
            image_features: Image features (batch, image_dim)
            text_features: Text features (batch, text_dim)
            numerical_features: Numerical features (batch, num_dim)
        
        Returns:
            Prediction (batch,)
        """
        modalities = []
        
        if image_features is not None:
            img_proj = self.image_projection(image_features)
            modalities.append(img_proj)
        
        if text_features is not None:
            text_proj = self.text_projection(text_features)
            modalities.append(text_proj)
        
        if numerical_features is not None:
            num_proj = self.numerical_projection(numerical_features)
            modalities.append(num_proj)
        
        if len(modalities) == 0:
            raise ValueError("At least one modality must be provided")
        
        # Concatenate all modalities
        if len(modalities) == 1:
            fused = modalities[0]
        else:
            fused = torch.cat(modalities, dim=1)
        
        # Predict
        output = self.fusion(fused)
        
        return output.squeeze(-1)
