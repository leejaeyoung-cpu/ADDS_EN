"""
Medical Decathlon Task010 Colon Cancer Dataset Downloader
공개 3D CT DICOM 데이터셋 자동 다운로드
"""

import urllib.request
import zipfile
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class MedicalDecathlonDownloader:
    """
    Medical Decathlon Colon Cancer 데이터셋 다운로더
    
    Dataset Info:
    - Task: Colon Cancer Segmentation
    - Modality: CT
    - Size: ~2.5 GB
    - Cases: 190 (126 train + 64 test)
    - Format: NIfTI (.nii.gz)
    """
    
    DATASET_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar"
    DATASET_NAME = "Task10_Colon"
    
    def __init__(self, download_dir: str = "data/medical_decathlon"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.archive_path = self.download_dir / f"{self.DATASET_NAME}.tar"
        self.extract_dir = self.download_dir / self.DATASET_NAME
    
    def download(self):
        """데이터셋 다운로드"""
        
        if self.archive_path.exists():
            logger.info(f"[OK] Archive already exists: {self.archive_path}")
            return
        
        logger.info("=" * 70)
        logger.info("Medical Decathlon Task010: Colon Cancer CT Dataset")
        logger.info("=" * 70)
        logger.info(f"URL: {self.DATASET_URL}")
        logger.info(f"Size: ~2.5 GB")
        logger.info(f"Download to: {self.archive_path}")
        logger.info("")
        
        logger.info("Downloading... (this may take 10-30 minutes)")
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            
            bar_length = 50
            filled = int(bar_length * percent / 100)
            bar = '#' * filled + '-' * (bar_length - filled)
            
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            
            sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
            sys.stdout.flush()
        
        try:
            urllib.request.urlretrieve(
                self.DATASET_URL,
                self.archive_path,
                progress_hook
            )
            print()  # New line after progress bar
            logger.info(f"[OK] Download complete: {self.archive_path}")
            
        except Exception as e:
            logger.error(f"[ERROR] Download failed: {e}")
            if self.archive_path.exists():
                self.archive_path.unlink()
            raise
    
    def extract(self):
        """아카이브 압축 해제"""
        
        if self.extract_dir.exists():
            logger.info(f"[OK] Already extracted: {self.extract_dir}")
            return
        
        logger.info("")
        logger.info("Extracting archive...")
        
        try:
            import tarfile
            
            with tarfile.open(self.archive_path, 'r') as tar:
                tar.extractall(self.download_dir)
            
            logger.info(f"[OK] Extraction complete: {self.extract_dir}")
            
        except Exception as e:
            logger.error(f"[ERROR] Extraction failed: {e}")
            raise
    
    def verify(self):
        """데이터셋 검증"""
        
        logger.info("")
        logger.info("Verifying dataset structure...")
        
        # Check directories
        images_dir = self.extract_dir / "imagesTr"
        labels_dir = self.extract_dir / "labelsTr"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
        
        # Count files
        images = list(images_dir.glob("*.nii.gz"))
        labels = list(labels_dir.glob("*.nii.gz"))
        
        logger.info(f"[OK] Images found: {len(images)}")
        logger.info(f"[OK] Labels found: {len(labels)}")
        
        if len(images) == 0:
            raise ValueError("No image files found!")
        
        # Show sample
        if images:
            logger.info(f"  Sample: {images[0].name}")
        
        return {
            'images_dir': images_dir,
            'labels_dir': labels_dir,
            'num_images': len(images),
            'num_labels': len(labels)
        }
    
    def download_and_prepare(self):
        """전체 프로세스 실행"""
        
        try:
            # 1. Download
            self.download()
            
            # 2. Extract
            self.extract()
            
            # 3. Verify
            info = self.verify()
            
            # Success summary
            logger.info("")
            logger.info("=" * 70)
            logger.info("[OK] Dataset Ready!")
            logger.info("=" * 70)
            logger.info(f"Location: {self.extract_dir}")
            logger.info(f"Images: {info['images_dir']}")
            logger.info(f"Labels: {info['labels_dir']}")
            logger.info(f"Total cases: {info['num_images']}")
            logger.info("")
            logger.info("You can now use this dataset for:")
            logger.info("  1. Training SOTA models (Swin-UNETR, nnU-Net)")
            logger.info("  2. Testing inference pipeline")
            logger.info("  3. Fine-tuning pretrained models")
            logger.info("")
            
            return info
            
        except Exception as e:
            logger.error(f"")
            logger.error("=" * 70)
            logger.error(f"[ERROR] Failed: {e}")
            logger.error("=" * 70)
            raise


def download_sample_case():
    """
    단일 샘플 케이스만 다운로드 (테스트용)
    """
    logger.info("Alternative: Download single sample case for testing")
    logger.info("")
    logger.info("Sample DICOM sources:")
    logger.info("1. The Cancer Imaging Archive (TCIA)")
    logger.info("   URL: https://www.cancerimagingarchive.net/")
    logger.info("   - Colon-specific collections available")
    logger.info("")
    logger.info("2. GitHub sample repositories")
    logger.info("   - Search: 'CT DICOM sample'")
    logger.info("")
    logger.info("3. Medical Segmentation Decathlon (recommended)")
    logger.info("   - Use full downloader above")


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("  Medical Decathlon Colon Cancer Dataset Downloader")
    print("=" * 70)
    print()
    
    # Ask user
    print("This will download ~2.5 GB of data.")
    print("Do you want to proceed? (y/n): ", end='')
    
    response = input().strip().lower()
    
    if response == 'y':
        print()
        downloader = MedicalDecathlonDownloader()
        downloader.download_and_prepare()
        
        print()
        print("Next steps:")
        print("  python train_sota_pipeline.py  # Start training")
        print("  python demo_3d_inference.py    # Test inference")
        
    else:
        print()
        print("Download cancelled.")
        print()
        download_sample_case()
