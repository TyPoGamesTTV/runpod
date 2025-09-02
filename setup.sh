#!/bin/bash
# RunPod Setup Script

echo "=========================================="
echo "RunPod Video Classifier Setup"
echo "=========================================="

# Install Python packages
echo "[1/3] Installing Python packages..."
pip install torch torchvision opencv-python numpy tqdm scikit-image gdown

# Verify GPU
echo "[2/3] Checking GPU..."
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"

# Create directories
echo "[3/3] Creating directories..."
mkdir -p /workspace/training
mkdir -p /workspace/models
mkdir -p /workspace/logs

echo "=========================================="
echo "Setup complete! Run: bash run_training.sh"
echo "=========================================="