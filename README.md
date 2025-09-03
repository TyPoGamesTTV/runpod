# RunPod X3D Video Classifier Setup

Quick deployment for video classification on RunPod A40/A100 instances.

## Quick Start (Web Console)

```bash
# Clone and setup
git clone https://github.com/TyPoGamesTTV/runpod.git
cd runpod
chmod +x setup.sh
./setup.sh

# Extract frames
python x3d_extractor.py

# Train model
./train_with_logging.sh
```

## What This Does

Trains an X3D-M model to classify videos into Safe/Unsafe/Explicit categories with ~70% accuracy.

## Directory Structure

```
/workspace/
├── organised_dataset/      # Your video files
│   ├── 1_Safe/
│   ├── 2_Unsafe/
│   └── 3_Explicit/
├── frames_x3d_clean/       # Extracted frames (auto-created)
└── best_x3d_tuned.pth     # Trained model
```

## Monitoring Training

```bash
# Watch live output
tail -f training_*.log

# Check from another session
ssh root@[POD_IP] -p [PORT] "tail -50 /workspace/training_*.log"
```

## Requirements

- RunPod with A40/A100 GPU (40GB+ VRAM)
- ~100GB storage
- 2500+ labeled videos for good results