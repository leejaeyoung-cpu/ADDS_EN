"""
GPU/CUDA 상태 진단 스크립트
"""
import torch
import os

print("=" * 60)
print("CUDA/GPU 진단 보고서")
print("=" * 60)

# 1. PyTorch CUDA 정보
print("\n[1] PyTorch CUDA 정보:")
print(f"  - CUDA Available: {torch.cuda.is_available()}")
print(f"  - PyTorch Version: {torch.__version__}")
print(f"  - CUDA Version: {torch.version.cuda}")
print(f"  - cuDNN Version: {torch.backends.cudnn.version()}")

# 2. GPU 디바이스 정보
if torch.cuda.is_available():
    print(f"\n[2] GPU 디바이스:")
    print(f"  - Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    * Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    * Multi-processor Count: {props.multi_processor_count}")
        print(f"    * CUDA Capability: {props.major}.{props.minor}")
    
    print(f"\n  - Current Device: {torch.cuda.current_device()}")
    print(f"  - Current Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("\n[2] GPU 디바이스: CUDA 사용 불가")

# 3. 환경 변수
print(f"\n[3] 환경 변수:")
print(f"  - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
print(f"  - CUDA_DEVICE_ORDER: {os.environ.get('CUDA_DEVICE_ORDER', 'Not Set')}")

# 4. GPU 메모리 사용량
if torch.cuda.is_available():
    print(f"\n[4] GPU 메모리:")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}:")
        print(f"    * Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"    * Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
        print(f"    * Max Allocated: {torch.cuda.max_memory_allocated(i) / 1024**2:.2f} MB")

# 5. 테스트: 간단한 텐서 연산
print(f"\n[5] GPU 연산 테스트:")
try:
    if torch.cuda.is_available():
        # CPU 테스트
        import time
        size = 5000
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        
        start = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start
        print(f"  - CPU 행렬곱 ({size}x{size}): {cpu_time:.4f}초")
        
        # GPU 테스트
        x_gpu = torch.randn(size, size).cuda()
        y_gpu = torch.randn(size, size).cuda()
        
        # Warm-up
        _ = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        z_gpu = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"  - GPU 행렬곱 ({size}x{size}): {gpu_time:.4f}초")
        print(f"  - 속도 향상: {cpu_time/gpu_time:.2f}배")
        
        print(f"\n  ✅ GPU 연산 정상 작동!")
    else:
        print(f"  ❌ CUDA 사용 불가")
except Exception as e:
    print(f"  ❌ 테스트 실패: {e}")

# 6. Cellpose GPU 설정 확인
print(f"\n[6] Cellpose GPU 설정:")
try:
    from cellpose import core
    print(f"  - Cellpose use_gpu 기본값: {core.use_gpu()}")
    if torch.cuda.is_available():
        print(f"  - GPU 사용 가능 여부: ✅")
    else:
        print(f"  - GPU 사용 가능 여부: ❌")
except ImportError:
    print(f"  - Cellpose 미설치")
except Exception as e:
    print(f"  - 확인 실패: {e}")

print("\n" + "=" * 60)
print("진단 완료")
print("=" * 60)
