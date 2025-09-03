# RunPod X3D Training - Issues & Fixes

## Critical Issues Solved

### 1. FFmpeg Frame Extraction Failure (66% fail rate)
**Problem:** FFmpeg filter using undefined `frames` variable
```bash
# BROKEN:
f"select='not(mod(n\\,int(max(1\\,floor(frames/{NUM_FRAMES})))))'"
```

**Fix:** Use simple modulo selection
```bash
# WORKING:
f"select='not(mod(n\\,4))'"  # Select every 4th frame
```

### 2. Model Mode Collapse (ViT)
**Problem:** ViT model predicting 100% single class, oscillating between classes each epoch

**Fix:** Switch to X3D-M architecture with:
- 3D convolutions for temporal understanding
- Smaller model (2.72M params vs 86M)
- Heavy regularization

### 3. Overfitting at Epoch 12
**Problem:** Training accuracy 95.9%, validation stuck at 68%

**Fix:** Aggressive regularization config:
```python
BASE_LR = 3e-5          # 3x lower
WEIGHT_DECAY = 5e-4     # 50x stronger
DROPOUT_RATE = 0.3      # Throughout network
LABEL_SMOOTHING = 0.1   # Prevent overconfidence
GRADIENT_CLIP = 0.5     # Aggressive clipping
```

### 4. Class Imbalance Issues
**Problem:** Model biased to Unsafe class (86-94% predictions)

**Fix:** Remove class weights - dataset already balanced:
```python
# Before: CLASS_WEIGHTS = [0.8, 1.4, 0.8]
# After:  CLASS_WEIGHTS = [1.0, 1.0, 1.0]
```

### 5. Corrupted NPZ Files During Training
**Problem:** zlib decompression errors

**Fix:** Extraction was hitting partially uploaded files
- Wait for uploads to complete before extracting
- Add validation check after extraction

### 6. SSH Connection Issues
**Problem:** Can't SSH until manually enabled

**Fix:** Use web console initially:
1. Access RunPod web terminal
2. Run setup script
3. SSH works after initial setup

### 7. No Training Output Visible
**Problem:** Can't monitor training progress

**Fix:** Always use logging wrapper:
```bash
./train_with_logging.sh
# Creates timestamped log: training_20250903_105800.log
```

## Performance Tips

### GPU Memory Management
- Batch size 16 for A40 (40GB)
- Batch size 12 for consumer GPUs
- 32 workers for extraction on high-end CPUs

### Frame Extraction Speed
- Use 32 parallel workers
- Process ~370 videos/second with fixed extractor
- Skip already extracted videos automatically

### Training Time
- ~50 seconds per epoch on A40
- Peak accuracy around epoch 12-15
- Early stopping prevents overfitting

## Common Commands

```bash
# Check GPU
nvidia-smi

# Monitor training
tail -f training_*.log

# Check extraction progress
ls -la frames_x3d_clean/*/*.npz | wc -l

# Kill stuck processes
pkill -9 python
pkill -9 ffmpeg

# Free space
rm -rf frames_x3d_clean/*
```

## Expected Results

- Initial validation: 35-40% (good - not memorizing)
- Peak validation: 65-75% 
- Training time: 12-15 epochs optimal
- Unsafe class: Most challenging (middle ground)

## Directory Paths

Always use absolute paths on RunPod:
- `/workspace/organised_dataset/` - Videos
- `/workspace/frames_x3d_clean/` - Extracted frames
- `/workspace/best_x3d_tuned.pth` - Best model