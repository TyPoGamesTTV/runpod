#!/bin/bash
# Training ONLY - assumes dataset already downloaded and extracted

echo "=========================================="
echo "Training Pipeline (Dataset Already Present)"
echo "=========================================="

cd /workspace/training

# Verify dataset exists
if [ ! -d "sample_dataset" ]; then
    echo "ERROR: sample_dataset directory not found!"
    echo "Run download_dataset.sh first"
    exit 1
fi

# Count videos
TOTAL_VIDEOS=$(find sample_dataset -name "*.mp4" -type f | wc -l)
echo "Found $TOTAL_VIDEOS videos in dataset"

if [ $TOTAL_VIDEOS -lt 100 ]; then
    echo "ERROR: Too few videos found ($TOTAL_VIDEOS)"
    exit 1
fi

# Extract frames
echo "[1/2] Extracting frames at 768x768..."
python3 extract_frames.py

# Verify frame extraction
if [ ! -d "frames_768" ]; then
    echo "ERROR: Frame extraction failed"
    exit 1
fi

# Train model
echo "[2/2] Training model..."
python3 train_model.py

echo "=========================================="
echo "Training Complete!"
echo "Model saved to: /workspace/models/"
echo "=========================================="