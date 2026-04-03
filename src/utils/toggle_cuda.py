"""
CUDA Toggle Utility for ADDS
Easy command-line tool to switch between CPU and GPU modes
"""

import os
import sys
from pathlib import Path

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def update_env_file(device: str, cellpose_gpu: bool):
    """
    Update .env file with new CUDA settings
    
    Args:
        device: 'cuda' or 'cpu'
        cellpose_gpu: True or False
    """
    # Get project root (3 levels up from src/utils/)
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / '.env'
    env_example_path = project_root / '.env.example'
    
    # Read existing .env or create from example
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    elif env_example_path.exists():
        with open(env_example_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        print("❌ Error: Neither .env nor .env.example found!")
        sys.exit(1)
    
    # Update device and cellpose settings
    updated_lines = []
    device_found = False
    cellpose_found = False
    
    for line in lines:
        if line.startswith('ADDS_DEVICE='):
            updated_lines.append(f'ADDS_DEVICE={device}\n')
            device_found = True
        elif line.startswith('ADDS_CELLPOSE_GPU='):
            updated_lines.append(f'ADDS_CELLPOSE_GPU={str(cellpose_gpu).lower()}\n')
            cellpose_found = True
        else:
            updated_lines.append(line)
    
    # Add settings if not found
    if not device_found:
        updated_lines.append(f'\nADDS_DEVICE={device}\n')
    if not cellpose_found:
        updated_lines.append(f'ADDS_CELLPOSE_GPU={str(cellpose_gpu).lower()}\n')
    
    # Write back to .env
    with open(env_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"✅ Updated .env file:")
    print(f"   ADDS_DEVICE={device}")
    print(f"   ADDS_CELLPOSE_GPU={str(cellpose_gpu).lower()}")

def show_current_settings():
    """Display current CUDA settings"""
    from dotenv import load_dotenv
    load_dotenv()
    
    device = os.getenv('ADDS_DEVICE', 'Not set')
    cellpose_gpu = os.getenv('ADDS_CELLPOSE_GPU', 'Not set')
    
    print("\n📊 Current Settings:")
    print(f"   Training Device: {device}")
    print(f"   Cellpose GPU: {cellpose_gpu}")
    
    # Check actual GPU availability
    try:
        import torch
        print(f"\n🔍 System Status:")
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except ImportError:
        print("\n⚠️  PyTorch not installed. Cannot check GPU status.")

def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("🚀 ADDS CUDA Toggle Utility")
        print("\nUsage:")
        print("  python toggle_cuda.py [mode]")
        print("\nModes:")
        print("  cpu    - Use CPU for all operations (Cellpose + Training)")
        print("  gpu    - Use GPU for all operations (Cellpose + Training)")
        print("  mixed  - Use GPU for training, CPU for Cellpose")
        print("  status - Show current settings")
        print("\nExamples:")
        print("  python toggle_cuda.py cpu")
        print("  python toggle_cuda.py gpu")
        print("  python toggle_cuda.py status")
        sys.exit(0)
    
    mode = sys.argv[1].lower()
    
    if mode == 'cpu':
        update_env_file('cpu', False)
        print("\n✨ Configured for CPU mode")
        print("   - Training: CPU")
        print("   - Cellpose: CPU")
        print("   - Best for: RTX 5070 compatibility issues")
    
    elif mode == 'gpu':
        update_env_file('cuda', True)
        print("\n✨ Configured for GPU mode")
        print("   - Training: GPU (CUDA)")
        print("   - Cellpose: GPU")
        print("   - Best for: When PyTorch supports sm_120")
    
    elif mode == 'mixed':
        update_env_file('cuda', False)
        print("\n✨ Configured for Mixed mode")
        print("   - Training: GPU (CUDA)")
        print("   - Cellpose: CPU")
        print("   - Best for: Partial GPU support")
    
    elif mode == 'status':
        show_current_settings()
    
    else:
        print(f"❌ Unknown mode: {mode}")
        print("   Valid modes: cpu, gpu, mixed, status")
        sys.exit(1)

if __name__ == '__main__':
    main()
