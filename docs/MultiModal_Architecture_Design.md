# Multi-Modal Learning Architecture Design
## ADDS Precision Oncology Platform

**Document Version:** 1.0  
**Date:** January 15, 2026  
**Status:** Design Specification  
**Target Implementation:** Q1-Q2 2026

---

## 1. Executive Summary

### Vision
Transform ADDS into a **true multi-modal precision oncology platform** by integrating pathology images, genomic data, clinical records, and spatial-temporal features through a unified deep learning architecture.

### Key Objectives
1. **Unified Representation:** Learn joint embeddings across modalities
2. **Cross-Modal Attention:** Enable modalities to inform each other
3. **Superior Performance:** Achieve 25%+ improvement in prognostic prediction
4. **Interpretability:** Maintain explainability through attention visualization

---

## 2. Current vs. Proposed Architecture

### 2.1 Current State (Single-Modal)

```
Input: Pathology Image (512×512×3)
  ↓
Cellpose Segmentation
  ↓
Feature Extraction (22 features)
  ↓
Random Forest / Gradient Boosting
  ↓
Prediction
```

**Limitations:**
- Only uses image data
- Ignores genomic/clinical information
- Features manually engineered
- Limited cross-patient generalization

### 2.2 Proposed Multi-Modal Architecture

```
┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐
│ Pathology Image │  │ Genomic Data │  │ Clinical Record │
└────────┬────────┘  └──────┬───────┘  └────────┬────────┘
         │                  │                   │
         ▼                  ▼                   ▼
   ┌─────────┐       ┌──────────┐        ┌──────────┐
   │ Vision  │       │ Genomic  │        │ Clinical │
   │ Encoder │       │ Encoder  │        │ Encoder  │
   └────┬────┘       └────┬─────┘        └────┬─────┘
        │                 │                    │
        │        ┌────────▼────────┐           │
        └───────►│ Cross-Modal     │◄──────────┘
                 │ Attention Fusion │
                 └────────┬─────────┘
                          ▼
                 ┌─────────────────┐
                 │ Joint Embedding │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │  Prediction Head│
                 │  - Prognosis    │
                 │  - Drug Response│
                 │  - Risk Score   │
                 └─────────────────┘
```

---

## 3. Detailed Component Design

### 3.1 Vision Encoder

**Architecture:** Vision Transformer (ViT) + Cellpose Integration

```python
class VisionEncoder(nn.Module):
    """
    Pathology image encoder combining Cellpose segmentation
    with self-supervised pretrained ViT
    """
    
    def __init__(self, cellpose_features=22, vit_dim=768):
        super().__init__()
        
        # Pretrained Vision Transformer
        self.vit = timm.create_model(
            'vit_base_patch16_224.dino',  # DINO pretrained
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Cellpose feature branch
        self.cellpose_processor = CellposeProcessor()
        self.cellpose_mlp = nn.Sequential(
            nn.Linear(cellpose_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512)
        )
        
        # Feature fusion
        self.fusion = nn.Linear(vit_dim + 512, 768)
        
    def forward(self, image):
        # ViT features (global)
        vit_features = self.vit(image)  # [B, 768]
        
        # Cellpose features (cell-level)
        with torch.no_grad():
            cellpose_output = self.cellpose_processor.segment_image(
                image.cpu().numpy()
            )
        cellpose_features = extract_summary_features(cellpose_output)
        cellpose_embed = self.cellpose_mlp(cellpose_features)  # [B, 512]
        
        # Combine
        combined = torch.cat([vit_features, cellpose_embed], dim=1)
        fused = self.fusion(combined)  # [B, 768]
        
        return fused
```

**Key Design Decisions:**
- **ViT over CNN:** Better for capturing global context in WSI patches
- **DINO Pretraining:** Self-supervised on medical images
- **Cellpose Preservation:** Leverage existing ADDS strengths
- **Late Fusion:** Combine learned + engineered features

### 3.2 Genomic Encoder

**Architecture:** 1D CNN + Transformer for variant sequences

```python
class GenomicEncoder(nn.Module):
    """
    Encode genomic variants and expression data
    
    Inputs:
        - Variant calls (VCF): {Gene, Variant, VAF}
        - Gene expression: RNA-seq TPM values
        - Copy number: Amplifications/deletions
    """
    
    def __init__(self, num_genes=20000, embed_dim=768):
        super().__init__()
        
        # Gene embedding
        self.gene_embed = nn.Embedding(num_genes, 128)
        
        # Variant type embedding
        self.variant_embed = nn.Embedding(
            num_classes=10,  # SNV, Insertion, Deletion, etc.
            embedding_dim=64
        )
        
        # 1D Convolution over gene sequences
        self.conv = nn.Sequential(
            nn.Conv1d(128+64, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # Transformer for long-range dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, 
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Projection to common dimension
        self.projection = nn.Linear(512, embed_dim)
        
    def forward(self, gene_ids, variant_types, expression_values):
        # Embed genes and variants
        gene_emb = self.gene_embed(gene_ids)  # [B, N, 128]
        var_emb = self.variant_embed(variant_types)  # [B, N, 64]
        
        # Concatenate
        x = torch.cat([gene_emb, var_emb], dim=-1)  # [B, N, 192]
        
        # 1D CNN
        x = x.transpose(1, 2)  # [B, 192, N]
        x = self.conv(x)  # [B, 512, N]
        x = x.transpose(1, 2)  # [B, N, 512]
        
        # Transformer
        x = self.transformer(x)  # [B, N, 512]
        
        # Global pooling
        x = x.mean(dim=1)  # [B, 512]
        
        # Project
        x = self.projection(x)  # [B, 768]
        
        return x
```

**Rationale:**
- **1D CNN:** Captures local mutation patterns (e.g., clustered mutations)
- **Transformer:** Models gene-gene interactions
- **Embedding:** Handles discrete variant types
- **Pooling:** Aggregates variable-length variant lists

### 3.3 Clinical Encoder

**Architecture:** BERT-style Transformer for clinical text

```python
class ClinicalEncoder(nn.Module):
    """
    Encode clinical records (demographics, lab values, pathology reports)
    """
    
    def __init__(self, embed_dim=768):
        super().__init__()
        
        # Structured data encoder (age, stage, ECOG, etc.)
        self.structured_encoder = nn.Sequential(
            nn.Linear(20, 256),  # 20 clinical features
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 512)
        )
        
        # Text encoder (pathology reports)
        self.text_encoder = AutoModel.from_pretrained(
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        )
        
        # Fusion
        self.fusion = nn.Linear(512 + 768, embed_dim)
        
    def forward(self, structured_features, report_text_tokens):
        # Structured features
        struct_emb = self.structured_encoder(structured_features)  # [B, 512]
        
        # Text features
        text_output = self.text_encoder(**report_text_tokens)
        text_emb = text_output.last_hidden_state[:, 0, :]  # [CLS] token, [B, 768]
        
        # Combine
        combined = torch.cat([struct_emb, text_emb], dim=1)
        fused = self.fusion(combined)  # [B, 768]
        
        return fused
```

### 3.4 Cross-Modal Attention Fusion

**Core Innovation:** Allow modalities to attend to each other

```python
class CrossModalAttention(nn.Module):
    """
    Multi-head cross-attention between modalities
    """
    
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        
        # Self-attention within each modality
        self.vision_self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.genomic_self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.clinical_self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Cross-attention: Vision ← Genomic/Clinical
        self.vision_cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Cross-attention: Genomic ← Vision/Clinical
        self.genomic_cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Cross-attention: Clinical ← Vision/Genomic
        self.clinical_cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Feed-forward networks
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, embed_dim)
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, vision_emb, genomic_emb, clinical_emb):
        # Add batch and sequence dimensions for attention
        v = vision_emb.unsqueeze(0)  # [1, B, 768]
        g = genomic_emb.unsqueeze(0)
        c = clinical_emb.unsqueeze(0)
        
        # Self-attention
        v_self, _ = self.vision_self_attn(v, v, v)
        g_self, _ = self.genomic_self_attn(g, g, g)
        c_self, _ = self.clinical_self_attn(c, c, c)
        
        # Cross-attention: Vision attends to Genomic + Clinical
        context = torch.cat([g_self, c_self], dim=0)  # [2, B, 768]
        v_cross, attn_weights_v = self.vision_cross_attn(
            query=v_self,
            key=context,
            value=context
        )
        
        # Cross-attention: Genomic attends to Vision + Clinical
        context = torch.cat([v_self, c_self], dim=0)
        g_cross, attn_weights_g = self.genomic_cross_attn(
            query=g_self,
            key=context,
            value=context
        )
        
        # Cross-attention: Clinical attends to Vision + Genomic
        context = torch.cat([v_self, g_self], dim=0)
        c_cross, attn_weights_c = self.clinical_cross_attn(
            query=c_self,
            key=context,
            value=context
        )
        
        # Combine with residuals
        v_out = self.norm(v + v_cross)
        g_out = self.norm(g + g_cross)
        c_out = self.norm(c + c_cross)
        
        # Feed-forward
        v_out = v_out + self.ffn(v_out)
        g_out = g_out + self.ffn(g_out)
        c_out = c_out + self.ffn(c_out)
        
        # Average pool across modalities
        joint_emb = (v_out + g_out + c_out) / 3.0
        joint_emb = joint_emb.squeeze(0)  # [B, 768]
        
        return joint_emb, {
            'vision_attn': attn_weights_v,
            'genomic_attn': attn_weights_g,
            'clinical_attn': attn_weights_c
        }
```

**Attention Interpretation:**
- **Vision → Genomic:** "Which genes explain this morphology?"
- **Genomic → Vision:** "What visual patterns match this mutation?"
- **Clinical → Vision/Genomic:** "How do demographics modify risk?"

---

## 4. Training Strategy

### 4.1 Data Requirements

| Modality | Data Type | Required Format | Sample Size |
|----------|-----------|-----------------|-------------|
| **Vision** | WSI patches | 512×512 PNG/TIFF | 10,000+ |
| **Genomic** | Variant calls | VCF + gene expression | 10,000+ |
| **Clinical** | EHR records | Structured + text | 10,000+ |

**Data Sources:**
- TCGA (The Cancer Genome Atlas): 11,000+ patients
- CPTAC (Clinical Proteomic Tumor Analysis): 1,000+ patients
- Internal hospital database: 500+ patients

### 4.2 Loss Functions

**Multi-Task Learning:**

```python
total_loss = (
    lambda_survival * survival_loss +
    lambda_response * response_loss +
    lambda_recon * reconstruction_loss +
    lambda_align * alignment_loss
)
```

**Components:**

1. **Survival Loss:** Cox Proportional Hazards
   ```python
   survival_loss = CoxPHLoss(predictions, event_times, event_indicators)
   ```

2. **Response Loss:** Binary Cross-Entropy
   ```python
   response_loss = BCELoss(drug_response_pred, actual_response)
   ```

3. **Reconstruction Loss:** Encourage modality-specific understanding
   ```python
   # Reconstruct vision from joint embedding
   recon_loss_v = MSELoss(decoder_v(joint_emb), vision_emb)
   recon_loss_g = MSELoss(decoder_g(joint_emb), genomic_emb)
   recon_loss = recon_loss_v + recon_loss_g
   ```

4. **Alignment Loss:** Contrastive learning across modalities
   ```python
   alignment_loss = InfoNCELoss(vision_emb, genomic_emb, temperature=0.1)
   ```

### 4.3 Training Recipe

```python
# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

# Learning rate schedule
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # First restart after 10 epochs
    T_mult=2,  # Double period after each restart
    eta_min=1e-6
)

# Training loop
for epoch in range(100):
    for batch in dataloader:
        # Forward pass
        joint_emb, attn_weights = model(
            vision=batch['image'],
            genomic=batch['variants'],
            clinical=batch['demographics']
        )
        
        # Predictions
        survival_pred = survival_head(joint_emb)
        response_pred = response_head(joint_emb)
        
        # Loss
        loss = compute_multi_task_loss(...)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    scheduler.step()
```

---

## 5. Implementation Roadmap

### Phase 1: Data Preparation (Weeks 1-2)
- [ ] Curate multi-modal dataset
- [ ] Implement data loaders
- [ ] Create train/val/test splits (70/15/15)

### Phase 2: Encoder Development (Weeks 3-6)
- [ ] Implement Vision Encoder
- [ ] Implement Genomic Encoder
- [ ] Implement Clinical Encoder
- [ ] Unit test each encoder

### Phase 3: Fusion Layer (Weeks 7-8)
- [ ] Implement Cross-Modal Attention
- [ ] Test attention mechanisms
- [ ] Visualize attention weights

### Phase 4: Training (Weeks 9-12)
- [ ] Implement loss functions
- [ ] Train on TCGA data
- [ ] Hyperparameter tuning
- [ ] Ablation studies

### Phase 5: Validation (Weeks 13-14)
- [ ] Evaluate on held-out test set
- [ ] Compare vs. single-modal baselines
- [ ] Clinical validation

---

## 6. Expected Performance

### Baseline (Current ADDS)
- Survival prediction C-index: **0.68**
- Drug response AUC: **0.72**

### Multi-Modal Target
- Survival prediction C-index: **0.85** (+25%)
- Drug response AUC: **0.90** (+25%)

### Ablation Study Predictions

| Configuration | Survival C-index | Response AUC |
|---------------|------------------|--------------|
| Vision only | 0.68 | 0.72 |
| Genomic only | 0.72 | 0.75 |
| Clinical only | 0.65 | 0.68 |
| Vision + Genomic | 0.78 | 0.83 |
| Vision + Clinical | 0.74 | 0.78 |
| All (no cross-attn) | 0.81 | 0.86 |
| **All (cross-attn)** | **0.85** | **0.90** |

---

## 7. XAI Integration

### Attention Visualization

```python
def visualize_cross_modal_attention(attention_weights, modality_names):
    """
    Visualize which modalities attend to each other
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (modality, attn) in enumerate(zip(modality_names, attention_weights)):
        sns.heatmap(
            attn.detach().cpu().numpy(),
            ax=axes[idx],
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'}
        )
        axes[idx].set_title(f'{modality} Attention Pattern')
        axes[idx].set_xlabel('Attending To')
        axes[idx].set_ylabel('Query')
    
    plt.tight_layout()
    return fig
```

### Clinical Interpretation

**Example Output:**
```
Patient ID: GC-12345
Prognosis: High-risk (survival 12 months)

Key Contributing Factors:
1. Vision (40%): High heterogeneity score (0.82)
   - Attention focused on irregular nuclear morphology
2. Genomic (35%): TP53 mutation + ERBB2 amplification
   - Cross-modal attention: mutation correlates with visual features
3. Clinical (25%): Stage IV, ECOG 2
   - Attention: Age and performance status reduce prognosis
```

---

## 8. Deployment Considerations

### Model Size
- **Parameters:** ~300M (ViT: 86M, Genomic: 50M, Clinical: 110M, Fusion: 54M)
- **Model File:** ~1.2 GB (FP32), ~600 MB (FP16)

### Inference Speed
- **GPU (RTX 5070):** 50-100 patients/second
- **CPU (fallback):** 5-10 patients/second

### Memory Requirements
- **Training:** 24 GB VRAM (batch size 16)
- **Inference:** 8 GB VRAM (batch size 1)

---

## 9. Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Insufficient data** | High | Use TCGA + data augmentation |
| **Overfitting** | Medium | Cross-validation, dropout, early stopping |
| **Modality imbalance** | Medium | Weighted loss functions |
| **Computational cost** | Low | Mixed-precision training (AMP) |

---

## 10. Success Metrics

### Technical Metrics
- [ ] C-index ≥ 0.85
- [ ] Response AUC ≥ 0.90
- [ ] Inference time < 500ms per patient
- [ ] Model size < 2 GB

### Clinical Metrics
- [ ] Approved by 3+ oncologists
- [ ] Validated on external dataset
- [ ] Published in top-tier journal (Cell, Nature Med)

---

**Document Status:** Ready for Implementation  
**Next Steps:** Begin Phase 1 data curation  
**Estimated Completion:** Q2 2026
