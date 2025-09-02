#!/bin/bash
# Script to package and prepare results for download

echo "=========================================="
echo "Packaging Training Results"
echo "=========================================="

# Create results directory
mkdir -p /workspace/results

# Copy models
echo "[1/4] Copying models..."
cp /workspace/models/*.pth /workspace/results/ 2>/dev/null || echo "No models found yet"

# Copy training log
echo "[2/4] Copying training log..."
cp /workspace/logs/training.log /workspace/results/ 2>/dev/null || echo "No training log found yet"

# Copy extraction metadata
echo "[3/4] Copying metadata..."
cp frames_768/metadata.json /workspace/results/extraction_metadata.json 2>/dev/null || echo "No metadata found"

# Create summary
echo "[4/4] Creating summary..."
cat > /workspace/results/summary.txt << EOF
Training Summary
================
Date: $(date)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
PyTorch: $(python3 -c "import torch; print(torch.__version__)")

Models:
$(ls -lh /workspace/models/*.pth 2>/dev/null || echo "No models yet")

To download locally:
scp -P 22150 root@$(hostname -I | awk '{print $1}'):/workspace/results/* ./
EOF

# Create zip
cd /workspace
zip -r training_results.zip results/

echo "=========================================="
echo "Results packaged!"
echo "Download: /workspace/training_results.zip"
echo "=========================================="
echo ""
echo "To download to your PC:"
echo "scp -P 22150 root@YOUR_RUNPOD_IP:/workspace/training_results.zip ./"