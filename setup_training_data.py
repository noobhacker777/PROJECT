"""
Setup training dataset structure and install required packages
Creates symbolic links to dataset/images, copies labels
Detects GPU and installs appropriate packages (GPU or CPU)
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path

PROJECT = Path(__file__).parent
DATASET = PROJECT / 'dataset'

def detect_gpu():
    """Detect if GPU (CUDA) is available"""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        device = "cuda" if has_cuda else "cpu"
        print(f"[GPU CHECK] PyTorch CUDA available: {has_cuda}")
        if has_cuda:
            print(f"[GPU CHECK] CUDA device: {torch.cuda.get_device_name(0)}")
        return has_cuda
    except Exception as e:
        print(f"[GPU CHECK] Error checking GPU: {e}")
        return False

def install_packages(gpu_available=True):
    """Install required packages based on GPU availability"""
    print("\n" + "="*80)
    print("INSTALLING REQUIRED PACKAGES")
    print("="*80)
    
    system = platform.system()
    print(f"[SYSTEM] Detected: {system}")
    
    # Core packages (required for all)
    core_packages = [
        "flask>=2.0.0",
        "pillow>=10.0.0",
        "opencv-python>=4.5.0",
        "pyyaml>=6.0",
        "watchdog>=2.0.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "psutil>=5.9.0",
        "pytest>=7.4.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "open-clip-torch>=2.24.0",
        "faiss-cpu>=1.7.4"
    ]
    
    print("\n[PACKAGES] Installing core packages...")
    for package in core_packages:
        print(f"  - {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
    # PyTorch packages based on GPU
    print("\n[PYTORCH] Installing PyTorch...")
    if gpu_available:
        print("[PYTORCH] GPU detected - installing CUDA-enabled PyTorch...")
        pytorch_packages = [
            "torch==2.9.1",
            "torchvision==0.24.1",
            "torchaudio==2.9.1"
        ]
        
        if system == "Windows":
            # Windows CUDA installation
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1",
                "-i", "https://download.pytorch.org/whl/cu121"
            ])
        elif system == "Darwin":
            # macOS with MPS support
            print("[PYTORCH] Installing for macOS (MPS support)...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1"
            ])
        else:
            # Linux CUDA installation
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1",
                "-i", "https://download.pytorch.org/whl/cu121"
            ])
        print("[PYTORCH] GPU packages installed: torch, torchvision, torchaudio (CUDA)")
    else:
        print("[PYTORCH] No GPU detected - installing CPU-only PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1",
            "-i", "https://download.pytorch.org/whl/cpu"
        ])
        print("[PYTORCH] CPU packages installed: torch, torchvision, torchaudio (CPU)")
    
    print("\n[SUCCESS] All packages installed successfully!")
    return True

def create_symlink_safe(src, dst):
    """Create symlink safely, handles Windows/Unix"""
    if dst.exists() or dst.is_symlink():
        return
    try:
        # Try creating symlink
        dst.symlink_to(src)
    except (OSError, NotImplementedError):
        # Fallback: use junction on Windows
        if platform.system() == "Windows":
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(dst), str(src)], check=False)
        else:
            # On Unix, use shutil.copytree as fallback
            shutil.copytree(src, dst, dirs_exist_ok=True)

def setup_tier1():
    """Setup Tier1 training structure - symlink images, copy labels"""
    print("\n[TIER1] Setting up training structure...")
    
    labels_dir = DATASET / 'labels'
    
    # Create tier1_training directory structure
    tier1_train = DATASET / 'tier1_training'
    tier1_train.mkdir(exist_ok=True)
    
    # Create symlink to dataset/images
    images_src = DATASET / 'images'
    images_link = tier1_train / 'images'
    
    print(f"  Creating symlink to images...")
    create_symlink_safe(images_src, images_link)
    
    # Copy tier1 labels to tier1_training/labels
    tier1_labels_src = labels_dir / 'tier1_router'
    tier1_labels_dest = tier1_train / 'labels'
    tier1_labels_dest.mkdir(exist_ok=True)
    
    print(f"  Copying tier1 labels...")
    for lbl in tier1_labels_src.glob('*.txt'):
        dest_lbl = tier1_labels_dest / lbl.name
        if not dest_lbl.exists():
            shutil.copy2(lbl, dest_lbl)
    
    # Create tier1 YAML
    yaml_content = f"""path: {tier1_train}
train: images
val: images
test: images
nc: 2
names:
  0: elec
  1: hardware
"""
    (DATASET / 'tier1_training.yaml').write_text(yaml_content)
    print(f"  Created tier1_training.yaml")
    
    return True

def setup_tier2():
    """Setup Tier2 training structure for each category - symlink images, copy labels"""
    print("\n[TIER2] Setting up training structures...")
    
    labels_dir = DATASET / 'labels'
    images_src = DATASET / 'images'
    
    # Find all expert directories
    tier2_base = labels_dir / 'tier2_specialists'
    expert_dirs = sorted(tier2_base.glob('*_expert'))
    
    categories = {
        'elec': ['bird', 'cable'],
        'hardware': ['screwdriver']
    }
    
    for expert_dir in expert_dirs:
        expert_name = expert_dir.name.replace('_expert', '')
        print(f"  Setting up {expert_name}_expert...")
        
        # Create tier2_{category}_training directory
        tier2_train = DATASET / f'tier2_{expert_name}_training'
        tier2_train.mkdir(exist_ok=True)
        
        # Create symlink to dataset/images
        images_link = tier2_train / 'images'
        create_symlink_safe(images_src, images_link)
        
        # Copy labels
        tier2_labels_dest = tier2_train / 'labels'
        tier2_labels_dest.mkdir(exist_ok=True)
        
        for lbl in expert_dir.glob('*.txt'):
            dest_lbl = tier2_labels_dest / lbl.name
            if not dest_lbl.exists():
                shutil.copy2(lbl, dest_lbl)
        
        # Create YAML
        names = {str(i): name for i, name in enumerate(categories.get(expert_name, [expert_name]))}
        yaml_content = f"""path: {tier2_train}
train: images
val: images
test: images
nc: {len(names)}
names: {names}
"""
        (DATASET / f'tier2_{expert_name}_training.yaml').write_text(yaml_content)
        print(f"    Created tier2_{expert_name}_training.yaml")
    
    return True

def main():
    print("\n" + "="*80)
    print("SETUP: PACKAGES + TRAINING DATA")
    print("="*80)
    
    try:
        # Step 1: Detect GPU
        print("\n[STEP 1/3] Detecting GPU availability...")
        gpu_available = detect_gpu()
        
        # Step 2: Install packages
        print("\n[STEP 2/3] Installing required packages...")
        install_packages(gpu_available=gpu_available)
        
        # Step 3: Setup training data
        print("\n[STEP 3/3] Setting up training data structure...")
        setup_tier1()
        setup_tier2()
        
        print("\n" + "="*80)
        print("[SUCCESS] SETUP COMPLETE")
        print("="*80)
        print("\nInstalled packages:")
        if gpu_available:
            print("  - PyTorch (GPU/CUDA enabled)")
        else:
            print("  - PyTorch (CPU only)")
        print("  - OpenCV, Flask, OpenCLIP, and all dependencies")
        
        print("\nTraining directories created:")
        print(f"  - {DATASET / 'tier1_training'}")
        print(f"  - {DATASET / 'tier2_*_training'} (for each category)")
        print("\nYAML configs created:")
        print(f"  - {DATASET / 'tier1_training.yaml'}")
        print(f"  - {DATASET / 'tier2_*_training.yaml'} (for each category)")
        print("\nReady to run: python train.py\n")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
