"""
Global GPU initialization - MUST BE IMPORTED FIRST
Forces NVIDIA RTX GPU (device 0) and blocks AMD integrated graphics
"""
import os
import sys

# CRITICAL: Set BEFORE any CUDA/PyTorch initialization
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Only NVIDIA RTX (GPU 0)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Consistent ordering
os.environ['HIP_VISIBLE_DEVICES'] = '-1'  # Disable AMD ROCm/HIP
os.environ['GPU_MAX_HEAP_SIZE'] = '100'  # NVIDIA optimization

# Windows console UTF-8 encoding
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 > nul 2>&1')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Import and verify
import torch

if torch.cuda.is_available():
    # Force set default device
    torch.cuda.set_device(0)
    
    device_name = torch.cuda.get_device_name(0)
    
    # Strict NVIDIA verification
    if "NVIDIA" not in device_name.upper():
        print("=" * 70)
        print("⚠️ CRITICAL WARNING: AMD GPU DETECTED!")
        print(f"   Current GPU: {device_name}")
        print("   Expected: NVIDIA RTX GPU")
        print()
        print("해결 방법:")
        print("1. Windows 그래픽 설정: Python을 '고성능'으로 설정")
        print("2. NVIDIA 제어판: 전역 설정을 'NVIDIA GPU'로 변경")
        print("=" * 70)
    else:
        print(f"✓ GPU 초기화 완료: {device_name}")
