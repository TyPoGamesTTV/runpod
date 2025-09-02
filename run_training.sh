#!/bin/bash
# Main Training Pipeline

echo "=========================================="
echo "Starting Training Pipeline"
echo "=========================================="

cd /workspace/training

# Download dataset (update GOOGLE_DRIVE_ID with your file ID)
echo "[1/4] Downloading dataset from Google Drive..."
echo "Please set GOOGLE_DRIVE_ID environment variable:"
echo "export GOOGLE_DRIVE_ID='your_file_id_here'"

if [ -z "$GOOGLE_DRIVE_ID" ]; then
    echo "ERROR: Please set GOOGLE_DRIVE_ID first!"
    echo "Get ID from your Google Drive share link"
    exit 1
fi

gdown "https://drive.google.com/uc?id=${GOOGLE_DRIVE_ID}" -O sample_dataset.zip

# Extract dataset
echo "[2/4] Extracting dataset..."
unzip -q sample_dataset.zip
rm sample_dataset.zip  # Save space

# Extract frames
echo "[3/4] Extracting frames at 768x768..."
python3 extract_frames.py

# Train model
echo "[4/4] Training model..."
python3 train_model.py

echo "=========================================="
echo "Training Complete!"
echo "Model saved to: /workspace/models/"
echo "=========================================="