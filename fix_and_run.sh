#!/bin/bash
# Emergency fix script - run this to fix and continue!

echo "=========================================="
echo "Fixing setup and continuing training"
echo "=========================================="

# Install unzip
echo "[1/5] Installing unzip..."
apt-get update && apt-get install -y unzip

# Extract the dataset (it's already downloaded!)
echo "[2/5] Extracting dataset..."
cd /workspace/training
unzip -q sample_dataset.zip
echo "Dataset extracted!"

# Copy our Python scripts to the right location
echo "[3/5] Copying training scripts..."
cp /workspace/runpod/*.py /workspace/training/
ls -la /workspace/training/*.py

# Now run extraction
echo "[4/5] Extracting frames at 768x768..."
cd /workspace/training
python3 extract_frames.py

# Train the model
echo "[5/5] Training model..."
python3 train_model.py

echo "=========================================="
echo "Training Complete!"
echo "Model saved to: /workspace/models/"
echo "=========================================="