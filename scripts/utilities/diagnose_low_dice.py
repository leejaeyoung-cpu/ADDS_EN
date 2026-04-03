"""
전문적 진단: 낮은 Dice Score 원인 분석
Low Dice Score Root Cause Analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
import logging

from src.medical_imaging.data.dataset_1ch_v2 import ColonCancerDataset1ChannelV2
from src.medical_imaging.models.model_1ch import SwinUNETR1Channel
from src.medical_imaging.training.losses import FocalDiceCELoss

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_predictions(model, dataloader, device):
    """예측 분포 분석"""
    model.model.eval()
    
    all_predictions = []
    all_labels = []
    
    logger.info("\n[분석 1] 예측 분포 (Prediction Distribution)")
    logger.info("="*60)
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            if idx >= 10:  # 10개 배치만 분석
                break
            
            images, labels = images.to(device), labels.to(device)
            logits = model.forward(images)
            pred = torch.argmax(logits, dim=1)
            
            all_predictions.append(pred.cpu())
            all_labels.append(labels.cpu())
            
            # 배치별 통계
            tumor_pred = (pred == 1).sum().item()
            tumor_true = (labels == 1).sum().item()
            total = pred.numel()
            
            logger.info(f"Batch {idx+1}:")
            logger.info(f"  예측 종양: {tumor_pred:,} / {total:,} ({tumor_pred/total*100:.2f}%)")
            logger.info(f"  실제 종양: {tumor_true:,} / {total:,} ({tumor_true/total*100:.2f}%)")
    
    # 전체 통계
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    pred_tumor_ratio = (all_predictions == 1).sum().float() / all_predictions.numel()
    true_tumor_ratio = (all_labels == 1).sum().float() / all_labels.numel()
    
    logger.info(f"\n전체 통계:")
    logger.info(f"  모델 예측 종양 비율: {pred_tumor_ratio.item()*100:.4f}%")
    logger.info(f"  실제 종양 비율: {true_tumor_ratio.item()*100:.4f}%")
    
    if pred_tumor_ratio < 0.001:
        logger.warning("⚠️ 모델이 거의 모든 것을 배경으로 예측!")
        logger.warning("   → 원인: 극도의 클래스 불균형")
    
    return pred_tumor_ratio.item(), true_tumor_ratio.item()


def analyze_loss_components(model, dataloader, criterion, device):
    """Loss 구성 요소 분해"""
    model.model.eval()
    
    logger.info("\n[분석 2] Loss 구성 요소 분해")
    logger.info("="*60)
    
    # Focal, Dice, CE loss 별도 계산
    focal_losses = []
    dice_losses = []
    ce_losses = []
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(dataloader):
            if idx >= 5:
                break
            
            images, labels = images.to(device), labels.to(device)
            logits = model.forward(images)
            
            # Total loss
            total_loss = criterion(logits, labels)
            
            # Individual components (approximation)
            # Focal loss component
            pred_probs = torch.softmax(logits, dim=1)
            focal_weight = (1 - pred_probs) ** 2.5  # gamma=2.5
            
            # Dice component
            pred = torch.argmax(logits, dim=1)
            tumor_pred = (pred == 1)
            tumor_true = (labels == 1)
            intersection = (tumor_pred & tumor_true).sum().float()
            union = tumor_pred.sum().float() + tumor_true.sum().float()
            dice = (2.0 * intersection / (union + 1e-5))
            dice_loss = 1.0 - dice
            
            focal_losses.append(total_loss.item())
            dice_losses.append(dice_loss.item())
            
            logger.info(f"Batch {idx+1}:")
            logger.info(f"  Total Loss: {total_loss.item():.4f}")
            logger.info(f"  Dice Loss: {dice_loss.item():.4f} (Dice: {dice.item():.4f})")
            logger.info(f"  Intersection: {intersection.item()}, Union: {union.item()}")
    
    avg_dice_loss = np.mean(dice_losses)
    logger.info(f"\n평균 Dice Loss: {avg_dice_loss:.4f}")
    logger.info(f"평균 Dice Score: {1.0 - avg_dice_loss:.4f}")


def analyze_class_imbalance(dataloader):
    """클래스 불균형 정량화"""
    logger.info("\n[분석 3] 클래스 불균형 정량화")
    logger.info("="*60)
    
    total_voxels = 0
    tumor_voxels = 0
    
    tumor_ratios = []
    
    for idx, (images, labels) in enumerate(dataloader):
        if idx >= 20:
            break
        
        batch_total = labels.numel()
        batch_tumor = (labels == 1).sum().item()
        
        total_voxels += batch_total
        tumor_voxels += batch_tumor
        
        ratio = batch_tumor / batch_total
        tumor_ratios.append(ratio)
        
        if idx < 5:
            logger.info(f"Sample {idx+1}: {batch_tumor:,} / {batch_total:,} ({ratio*100:.4f}%)")
    
    overall_ratio = tumor_voxels / total_voxels
    
    logger.info(f"\n전체 통계 (20 samples):")
    logger.info(f"  종양 voxels: {tumor_voxels:,}")
    logger.info(f"  전체 voxels: {total_voxels:,}")
    logger.info(f"  종양 비율: {overall_ratio*100:.4f}%")
    logger.info(f"  불균형 비율: {1/overall_ratio:.1f}:1 (배경:종양)")
    
    if overall_ratio < 0.01:
        logger.warning("⚠️ 극도의 클래스 불균형! (<1% 종양)")
        logger.warning("   → 해결책: 더 강력한 loss weighting 필요")
    
    return overall_ratio


def main():
    logger.info("="*60)
    logger.info("Transfer Learning 1-Channel V2 - 진단 분석")
    logger.info("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    logger.info("\n데이터셋 로딩...")
    dataset = ColonCancerDataset1ChannelV2(
        data_root='data/medical_decathlon/Task10_Colon',
        fold=0,
        mode='train',
        tumor_focused_prob=0.95
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Model
    logger.info("\n모델 로딩...")
    model = SwinUNETR1Channel(
        pretrained_path='models/pretrained/swin_unetr_pretrained.pt',
        out_channels=2,
        device='cuda'
    )
    
    # Loss
    criterion = FocalDiceCELoss(0.6, 0.3, 0.1, 0.8, 3.0)
    
    # 분석 실행
    logger.info("\n" + "="*60)
    logger.info("진단 분석 시작")
    logger.info("="*60)
    
    # 1. 클래스 불균형
    tumor_ratio = analyze_class_imbalance(dataloader)
    
    # 2. 예측 분포
    pred_ratio, true_ratio = analyze_predictions(model, dataloader, device)
    
    # 3. Loss 분해
    analyze_loss_components(model, dataloader, criterion, device)
    
    # 최종 진단
    logger.info("\n" + "="*60)
    logger.info("최종 진단 및 권장 사항")
    logger.info("="*60)
    
    logger.info(f"\n1. 클래스 불균형: {tumor_ratio*100:.4f}% 종양")
    logger.info(f"2. 모델 예측: {pred_ratio*100:.4f}% 종양 예측")
    logger.info(f"3. 예측 차이: {abs(pred_ratio - tumor_ratio)*100:.4f}%p")
    
    if pred_ratio < 0.001 and tumor_ratio > 0.01:
        logger.warning("\n🔴 핵심 문제: 모델이 종양을 거의 예측하지 않음!")
        logger.warning("\n권장 해결책:")
        logger.warning("  1. Loss function 가중치 강화 (Focal weight 0.8+)")
        logger.warning("  2. Positive class weight 추가 (10:1 이상)")
        logger.warning("  3. Learning rate 증가 (0.0005+)")
        logger.warning("  4. Batch size 감소 → gradient 강화")
        logger.warning("  5. Encoder freezing 고려 (decoder만 학습)")


if __name__ == "__main__":
    main()
