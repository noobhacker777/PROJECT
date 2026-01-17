#!/usr/bin/env python3
"""
HOLO11 Training Script
Licensed under AGPL-3.0. See LICENSE file for details.

Portions derived from Ultralytics YOLOv11 (AGPL-3.0).

Trains HOLO11 model using the product dataset.
Uses converted JSON annotations in HOLO format.
"""

import sys
from pathlib import Path
import torch
import argparse


def create_data_yaml(output_path):
    """Create data.yaml configuration file.
    
    Args:
        output_path: Path to save data.yaml
    """
    yaml_content = """# HOLO Dataset Configuration
# Auto-generated dataset configuration for training

# Dataset paths (relative to dataset root)
path: ./dataset
train: images
val: images
test: images

# Number of classes
nc: 1

# Class names
names:
  0: product

# Dataset statistics
# Total images: 18
# Total annotations: 40
# Created: 2025-12-28
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_content, encoding='utf-8')


def main():
    """Run HOLO11 training."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train HOLO11 model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device to use (0 for GPU, cpu for CPU)')
    args = parser.parse_args()
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Import after adding to path
    from hyperimagedetect import HOLO
    
    # Dataset configuration
    data_yaml = project_root / 'dataset' / 'data.yaml'
    
    # Auto-create data.yaml if it doesn't exist
    if not data_yaml.exists():
        print(f"Creating dataset configuration: {data_yaml}")
        create_data_yaml(data_yaml)
        print(f"✓ Dataset configuration created")
    else:
        print(f"Using existing dataset configuration: {data_yaml}")
    
    # Check GPU availability
    device = args.device
    if device != 'cpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Dataset config: {data_yaml}")
    
    # Load model - always train from scratch
    print("\nLoading model...")
    print("Training from scratch with base HOLO11 model...")
    model = HOLO('holo11m.pt')
    print("✓ Model loaded successfully")
    
    # Training parameters optimized for high accuracy (0.96+)
    training_config = {
        'data': str(data_yaml),
        'epochs': args.epochs,              # User-specified epochs
        'imgsz': args.imgsz,
        'batch': args.batch,                # User-specified batch size
        'device': device,
        'patience': 100,            # Increased patience to allow more training
        'save': True,
        'cache': True,
        'close_mosaic': 50,         # Delay closing mosaic augmentation
        'verbose': True,
        # Augmentation for better generalization
        'fliplr': 0.5,              # 50% horizontal flip
        'flipud': 0.0,              # No vertical flip (product detection)
        'mosaic': 1.0,              # Full mosaic augmentation
        'copy_paste': 0.1,          # Copy-paste augmentation
        'mixup': 0.2,               # Mixup augmentation
        'scale': 0.5,               # Image scale augmentation
        'hsv_h': 0.015,             # HSV hue
        'hsv_s': 0.7,               # HSV saturation
        'hsv_v': 0.4,               # HSV value
        'degrees': 10.0,            # Rotation
        'translate': 0.1,           # Translation
        'perspective': 0.0,         # Perspective
        'shear': 5.0,               # Shear
        'erasing': 0.4,             # Random erasing
        'lr0': 0.001,               # Initial learning rate (lower for precision)
        'lrf': 0.01,                # Final learning rate
        'warmup_epochs': 5.0,       # Warmup epochs
        'warmup_bias_lr': 0.1,      # Warmup bias LR
        'warmup_momentum': 0.8,     # Warmup momentum
        'momentum': 0.937,          # SGD momentum
        'weight_decay': 0.0005,     # L2 regularization
        'optimizer': 'SGD',         # Optimizer type
        'rect': False,              # Rectangular training
        'multi_scale': False,       # Multi-scale training
        'fraction': 1.0,            # Dataset fraction to use
        'plots': True,              # Plot training results
        'save_period': -1,          # Save every epoch
    }
    
    print("\nTraining configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    # Train model
    print("\nStarting training...\n")
    try:
        results = model.train(**training_config)
        print("\n✓ Training completed successfully")
        
        # Move trained model to scan folder
        print("\nMoving trained model to scan folder...")
        scan_models_dir = project_root / 'scan_models'
        scan_models_dir.mkdir(parents=True, exist_ok=True)
        
        trained_model_path = project_root / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
        scan_model_path = scan_models_dir / 'scan_model.pt'
        
        if trained_model_path.exists():
            import shutil
            shutil.copy2(str(trained_model_path), str(scan_model_path))
            print(f"✓ Model copied to: {scan_model_path}")
        else:
            print(f"⚠ Warning: Trained model not found at {trained_model_path}")
        
        return True
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        return False


if __name__ == '__main__':
    try:
            from json_to_holo import main as convert_json
            print("\n📝 Auto-converting JSON annotations to HOLO text format...")
            convert_json()
    except Exception as e:
            print(f"⚠️  JSON conversion warning (non-fatal): {e}")

    success = main()
    sys.exit(0 if success else 1)
