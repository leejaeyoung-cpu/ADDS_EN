"""
MONAI Swin-UNETR 사전 학습 모델 다운로드
"""

import torch
from pathlib import Path
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """URL에서 파일 다운로드 (진행 표시)"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_pretrained_swin_unetr():
    """MONAI Swin-UNETR 사전 학습 모델 다운로드"""
    
    print("=" * 60)
    print("MONAI Swin-UNETR Pretrained Model Downloader")
    print("=" * 60)
    
    # 다운로드 URL (MONAI official)
    url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"
    
    # 저장 경로
    save_dir = Path("models/pretrained")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "swin_unetr_pretrained.pt"
    
    # 이미 다운로드되었는지 확인
    if save_path.exists():
        print(f"\n[OK] Model already exists: {save_path}")
        print(f"   Size: {save_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        response = input("\nDownload again? (y/n): ")
        if response.lower() != 'y':
            print("Using existing model.")
            return save_path
    
    # 다운로드
    print(f"\n[Download] Downloading from: {url}")
    print(f"   Save to: {save_path}")
    print(f"   Expected size: ~720 MB")
    print("\nThis may take 3-5 minutes...")
    
    try:
        download_url(url, save_path)
        
        # 검증
        file_size_mb = save_path.stat().st_size / 1024 / 1024
        print(f"\n✅ Download complete!")
        print(f"   Size: {file_size_mb:.1f} MB")
        
        if file_size_mb < 700:
            print("\n⚠️ Warning: File size seems too small. Download may be incomplete.")
            return None
        
        return save_path
        
    except Exception as e:
        print(f"\n❌ Download failed: {str(e)}")
        if save_path.exists():
            save_path.unlink()
        return None

def verify_model(model_path):
    """모델 파일 검증"""
    print("\n" + "=" * 60)
    print("Verifying model...")
    print("=" * 60)
    
    try:
        # PyTorch로 로드 테스트
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print(f"[OK] Model loaded successfully!")
        print(f"   Keys: {len(checkpoint.keys() if isinstance(checkpoint, dict) else 'N/A')}")
        
        if isinstance(checkpoint, dict):
            # 주요 정보 출력
            if 'state_dict' in checkpoint:
                print(f"   State dict found")
            if 'epoch' in checkpoint:
                print(f"   Trained epochs: {checkpoint['epoch']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Model verification failed: {str(e)}")
        return False

def main():
    print("\n[*] Starting pretrained model download...\n")
    
    # 다운로드
    model_path = download_pretrained_swin_unetr()
    
    if model_path is None:
        print("\n[X] Download failed. Please try again.")
        return False
    
    # 검증
    if not verify_model(model_path):
        print("\n[X] Model verification failed.")
        return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS!]")
    print("=" * 60)
    print(f"\nPretrained model ready at:")
    print(f"  {model_path.absolute()}")
    print("\nNext steps:")
    print("  1. Update UI to use pretrained model")
    print("  2. Test with CT images")
    print("  3. Compare with Mock mode")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
