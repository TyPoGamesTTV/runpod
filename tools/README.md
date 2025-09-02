# Video Classification Tools

## Quick Start

### Single Video Classification
```bash
# Basic usage
python tools/inference/single_video.py video.mp4 --model best_model_768.pth

# Show all class probabilities
python tools/inference/single_video.py video.mp4 --model best_model_768.pth --show-all

# Use CPU instead of GPU
python tools/inference/single_video.py video.mp4 --model best_model_768.pth --device cpu

# Set confidence threshold
python tools/inference/single_video.py video.mp4 --model best_model_768.pth --threshold 0.7
```

### Batch Processing
```bash
# Process entire folder
python tools/inference/single_video.py /path/to/videos/ --model best_model_768.pth

# Save results to CSV
python tools/inference/single_video.py /path/to/videos/ --model best_model_768.pth --output results.csv

# Process with larger batch size (if enough VRAM)
python tools/inference/single_video.py /path/to/videos/ --model best_model_768.pth --batch-size 4
```

## Tool Descriptions

### 📁 inference/
**single_video.py** - Main inference tool
- Classifies single videos or folders
- Auto-detects GPU/CPU
- Adjusts batch size for available memory
- Outputs confidence scores
- Saves results to CSV

### 📊 evaluation/
*Coming soon*
- evaluate_model.py - Calculate accuracy metrics
- confusion_matrix.py - Visualize model performance
- error_analysis.py - Identify problem cases

### 🎯 dataset/
*Coming soon*
- prepare_dataset.py - Organize training data
- augment_data.py - Data augmentation
- split_dataset.py - Create train/val/test splits

### 📈 visualization/
*Coming soon*
- plot_training.py - Visualize training curves
- visualize_frames.py - Display extracted frames
- heatmap_attention.py - Model attention maps

## Requirements

```bash
pip install torch torchvision opencv-python numpy tqdm
```

## Model Download

After training completes on RunPod:
```bash
# On RunPod
cd /workspace
zip -r model_package.zip models/ logs/ checkpoints/

# On local machine
scp -P [PORT] root@[IP]:/workspace/model_package.zip ./
unzip model_package.zip
```

## Performance Notes

### GPU Memory Requirements
| Resolution | Frames | 3080 Ti (12GB) | A40 (48GB) |
|------------|--------|----------------|------------|
| 768x768 | 16 | Batch 1-2 | Batch 4-8 |
| 512x512 | 16 | Batch 4-6 | Batch 16-24 |
| 224x224 | 16 | Batch 16-32 | Batch 64-128 |

### Processing Speed
- **3080 Ti**: ~0.5-1 videos/second
- **A40**: ~2-4 videos/second
- **CPU**: ~0.1-0.2 videos/second

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python tools/inference/single_video.py videos/ --model model.pth --batch-size 1

# Or use CPU
python tools/inference/single_video.py videos/ --model model.pth --device cpu
```

### Model Loading Errors
- Ensure model was trained with same resolution (768x768)
- Check model has correct number of classes (3)
- Verify PyTorch version compatibility

### Video Processing Errors
- Supported formats: MP4, AVI, MOV, MKV, WEBM
- Minimum frames: 16 (will pad if fewer)
- Corrupted videos are skipped with warning

## Example Output

```
Using GPU: NVIDIA GeForce RTX 3080 Ti (12.0GB)
Loading model from best_model_768.pth...

Processing: test_video.mp4
Video info: 450 frames, 30.0 FPS, 15.0 seconds
Frame extraction: 1.23 seconds
Inference: 0.456 seconds

Classification: Unsafe
Confidence: 87.34%
```

## Advanced Usage

### Custom Resolution
```bash
# If model was trained at different resolution
python tools/inference/single_video.py video.mp4 \
    --model model_512.pth \
    --resolution 512 \
    --num-frames 8
```

### Integration Example
```python
from tools.inference.single_video import VideoClassifier

# Initialize
classifier = VideoClassifier('best_model_768.pth')

# Classify
class_name, confidence = classifier.classify('video.mp4')
print(f"{class_name}: {confidence:.2%}")

# Get all probabilities
class_name, confidence, all_probs = classifier.classify('video.mp4', return_probs=True)
for name, prob in zip(classifier.class_names, all_probs):
    print(f"{name}: {prob:.2%}")
```

## Coming Next

1. **evaluate_model.py** - Test model on validation set
2. **export_onnx.py** - Convert for deployment
3. **api_server.py** - REST API for classification
4. **gradio_demo.py** - Web interface