#!/bin/bash
# RunPod X3D Setup Script
# One-click setup from bare RunPod to ready-to-train

echo "==================================="
echo "RunPod X3D Classifier Setup v1.0"
echo "==================================="

# Update system
echo "[1/7] Updating system packages..."
apt-get update -qq
apt-get install -y ffmpeg htop screen tmux > /dev/null 2>&1

# Install Python packages
echo "[2/7] Installing Python dependencies..."
pip install -q numpy torch torchvision tqdm pathlib

# Create directory structure
echo "[3/7] Creating directories..."
mkdir -p /workspace/organised_dataset/{1_Safe,2_Unsafe,3_Explicit}
mkdir -p /workspace/frames_x3d_clean
mkdir -p /workspace/logs

# Copy scripts to workspace
echo "[4/7] Installing scripts..."
cp x3d_extractor.py /workspace/
cp train_x3d_tuned.py /workspace/
cp train_with_logging.sh /workspace/
chmod +x /workspace/train_with_logging.sh

# Setup SSH (if not already done)
echo "[5/7] Configuring SSH..."
if [ ! -f ~/.ssh/authorized_keys ]; then
    mkdir -p ~/.ssh
    echo "# Add your SSH public key here" > ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys
    echo "⚠️  Add your SSH key to ~/.ssh/authorized_keys"
fi

# Create monitoring script
echo "[6/7] Creating monitor script..."
cat > /workspace/monitor.sh << 'EOF'
#!/bin/bash
echo "Training Status:"
tail -20 /workspace/training_*.log 2>/dev/null || echo "No training running"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
echo ""
echo "Extraction Progress:"
echo "Total frames: $(ls -la /workspace/frames_x3d_clean/*/*.npz 2>/dev/null | wc -l)"
EOF
chmod +x /workspace/monitor.sh

# Create quick test script
echo "[7/7] Creating test script..."
cat > /workspace/test_setup.py << 'EOF'
#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path

print("✓ PyTorch:", torch.__version__)
print("✓ CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✓ GPU:", torch.cuda.get_device_name())
print("✓ NumPy:", np.__version__)

# Check directories
dirs = [
    "/workspace/organised_dataset/1_Safe",
    "/workspace/organised_dataset/2_Unsafe", 
    "/workspace/organised_dataset/3_Explicit",
    "/workspace/frames_x3d_clean"
]
for d in dirs:
    p = Path(d)
    if p.exists():
        count = len(list(p.glob("*")))
        print(f"✓ {d}: {count} files")
    else:
        print(f"✗ {d}: missing")
EOF
chmod +x /workspace/test_setup.py

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Upload videos to /workspace/organised_dataset/{1_Safe,2_Unsafe,3_Explicit}/"
echo "2. Run: python /workspace/x3d_extractor.py"
echo "3. Run: /workspace/train_with_logging.sh"
echo ""
echo "Test setup: python /workspace/test_setup.py"
echo "Monitor: /workspace/monitor.sh"
echo ""