Manual CLI Training Guide

Recommended command for HOLO11m (medium) model training:

python train.py --epochs 100 --batch 8 --imgsz 640 --device (GPU/CPU)

Notes:
- Ensure you use the correct model config for the 'm' scale: holo11m.yaml
- To train from scratch, edit train.py to use HOLO('holo11m.yaml')
- For GPU, use --device 0 (or --device cpu for CPU)
- Make sure your dataset and data.yaml are set up as described in the documentation.

Example for training with the medium model:

python train.py --epochs 100 --batch 8 --imgsz 640 --device (GPU/CPU)