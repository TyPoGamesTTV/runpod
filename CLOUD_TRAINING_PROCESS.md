# Cloud Training Process - What Actually Works

## Overview
Complete process for training video classifiers on RunPod GPUs. This is what we actually do, not what the docs say.

## Phase 1: Local Preparation (30 mins)

### 1.1 Create Sample Dataset
```python
# create_sample.py - Run locally
import random
import shutil
from pathlib import Path

source = Path('E:/organised_dataset')
dest = Path('sample_dataset')

# Take 500 videos per class (1500 total = ~11GB)
for class_name in ['1_Safe', '2_Unsafe', '3_Explicit']:
    class_source = source / class_name
    class_dest = dest / class_name
    class_dest.mkdir(parents=True, exist_ok=True)
    
    videos = list(class_source.glob('*.mp4'))
    sample = random.sample(videos, min(500, len(videos)))
    
    for video in sample:
        shutil.copy2(video, class_dest / video.name)
    print(f'{class_name}: {len(sample)} videos')
```

### 1.2 Compress Dataset
```powershell
# PowerShell - Creates ~11GB zip from 15GB videos
Compress-Archive -Path sample_dataset -DestinationPath sample_dataset.zip -CompressionLevel Optimal
# Takes ~10 minutes
```

### 1.3 Upload to Google Drive
- Upload `sample_dataset.zip` to Google Drive (~30 mins at 100Mbit)
- Get shareable link
- Extract file ID from URL

## Phase 2: RunPod Setup (10 mins)

### 2.1 Rent GPU
- Go to [RunPod.io](https://runpod.io) → Pods
- **Recommended**: A40 48GB ($0.40/hr) or RTX 4090 24GB ($0.44/hr)
- Select **PyTorch 2.0** template
- **50GB disk minimum**
- Deploy as **On-Demand** (not spot!)

### 2.2 Connect via SSH
```powershell
# They give you connection details
ssh root@[IP_ADDRESS] -p [PORT]

# Example that worked:
ssh root@69.30.85.213 -p 22150
```

**Note**: Web terminal works but copy/paste is broken in Firefox. Use SSH!

### 2.3 Initial Setup
```bash
# Clone our repo with all scripts
cd /workspace
git clone https://github.com/TyPoGamesTTV/runpod.git
cd runpod

# Run setup (installs packages, verifies GPU)
bash setup.sh

# Test model architecture
python3 quick_test.py
# Should show: "✓ Model created: 89.4M parameters"
```

## Phase 3: Training Pipeline (2-3 hours)

### 3.1 Start Training
```bash
# Set Google Drive file ID
export GOOGLE_DRIVE_ID='your_file_id_here'

# Test download first
bash test_download.sh

# Run full pipeline
bash run_training.sh
```

**What happens:**
1. Downloads dataset from Google Drive (~10 mins for 11GB)
2. Extracts zip file (~2 mins)
3. Processes videos to 768x768 grayscale frames (~30 mins)
4. Trains ViT model for 30 epochs (~1.5 hours on A40)

### 3.2 Monitor Progress
```bash
# In another SSH session
cd /workspace/runpod

# Live monitoring
python3 monitor_training.py

# Or check logs directly
tail -f /workspace/logs/training.log

# Watch GPU usage
watch -n 1 nvidia-smi
```

### 3.3 Training Performance

| GPU | VRAM | Batch Size | Time/Epoch | Total Time |
|-----|------|------------|------------|------------|
| 3080 Ti (local) | 12GB | 4 | ~3 min | ~90 min |
| RTX 4090 | 24GB | 8 | ~2 min | ~60 min |
| A40 | 48GB | 16 | ~1.5 min | ~45 min |

## Phase 4: Download Results (10 mins)

### 4.1 Package Results
```bash
# After training completes
bash download_results.sh
# Creates /workspace/training_results.zip
```

### 4.2 Download to Local
```powershell
# From Windows PowerShell
scp -P [PORT] root@[IP]:/workspace/training_results.zip ./

# Example:
scp -P 22150 root@69.30.85.213:/workspace/training_results.zip ./
```

### 4.3 Stop RunPod
**IMPORTANT**: Stop the pod in RunPod dashboard to avoid continued charges!

## What We Learned

### What Works
- ✅ SSH with key auth (not password)
- ✅ Google Drive with gdown
- ✅ GitHub repo for scripts
- ✅ A40 48GB for best price/performance
- ✅ 768x768 grayscale (good detail, manageable size)
- ✅ Sample dataset for testing (1500 videos)

### What Doesn't Work
- ❌ Web terminal copy/paste in Firefox
- ❌ Expecting pre-installed ML packages (need setup.sh)
- ❌ tmux/screen (not installed by default)
- ❌ 32GB full dataset (too big to transfer quickly)
- ❌ WeTransfer for wget (gives HTML page)
- ❌ gdown without --fuzzy flag (fails on large files)

### Time Breakdown (Actual)
- Local prep: 30 mins
- Upload to Google Drive: 30 mins
- RunPod setup: 10 mins
- Download to RunPod: 10 mins
- Frame extraction: 30 mins
- Training: 90 mins
- Download results: 10 mins
- **Total: ~3.5 hours**

### Cost
- A40 rental: 3.5 hours × $0.40 = **$1.40**
- Expected accuracy: **75-80%** (vs 64% on local 384x384)

## Quick Checklist for Next Time

Before renting:
- [ ] Sample dataset created and zipped
- [ ] Uploaded to Google Drive
- [ ] Google Drive file ID ready
- [ ] SSH key generated locally
- [ ] GitHub repo accessible

On RunPod:
- [ ] Clone repo
- [ ] Run setup.sh
- [ ] Test with quick_test.py
- [ ] Set GOOGLE_DRIVE_ID
- [ ] Run training pipeline
- [ ] Monitor progress
- [ ] Download results
- [ ] **STOP THE POD**

## Improvements for Next Time

1. **Pre-extract frames locally** (save 30 mins on RunPod)
2. **Use larger dataset** if time permits
3. **Try multiple experiments** (we have 10 hours!)
4. **Test RGB vs grayscale** comparison
5. **Save checkpoint models** during training

---

**Remember**: Everything is in https://github.com/TyPoGamesTTV/runpod - just clone and run!