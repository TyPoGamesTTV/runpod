# RunPod Video Classifier Training

768x768 Grayscale Video Classifier for A40 GPU

## Quick Start

```bash
# On RunPod:
git clone https://github.com/TyPoGamesTTV/runpod.git
cd runpod
bash setup.sh
bash run_training.sh
```

## What This Does

1. Installs all required packages
2. Downloads dataset from Google Drive
3. Extracts frames at 768x768 grayscale
4. Trains Vision Transformer model
5. Saves best model for download

## Requirements

- RunPod instance with GPU (A40/A100/4090)
- ~50GB disk space
- Dataset uploaded to Google Drive