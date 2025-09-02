# Tools Architecture Plan

## /tools Directory Structure

```
/tools/
├── inference/
│   ├── single_video.py      # Classify single video
│   ├── batch_inference.py   # Process folder of videos
│   └── live_inference.py    # Real-time classification
│
├── evaluation/
│   ├── evaluate_model.py    # Calculate metrics
│   ├── confusion_matrix.py  # Generate confusion matrix
│   └── error_analysis.py    # Analyze misclassifications
│
├── dataset/
│   ├── prepare_dataset.py   # Create training dataset
│   ├── augment_data.py      # Data augmentation
│   ├── split_dataset.py     # Train/val/test splits
│   └── verify_dataset.py    # Check dataset integrity
│
├── visualization/
│   ├── plot_training.py     # Training curves
│   ├── visualize_frames.py  # Show extracted frames
│   └── heatmap_attention.py # Visualize model attention
│
├── conversion/
│   ├── export_onnx.py       # Export to ONNX
│   ├── export_tflite.py     # Export to TensorFlow Lite
│   └── quantize_model.py    # Model quantization
│
├── deployment/
│   ├── api_server.py        # REST API server
│   ├── gradio_demo.py       # Interactive demo
│   └── docker_build.py      # Docker deployment
│
└── utilities/
    ├── download_model.py    # Download from RunPod
    ├── benchmark.py         # Speed benchmarks
    └── system_check.py      # Verify GPU/environment
```

## Priority Tools (Build First)

### 1. inference/single_video.py
```python
# Core functionality:
- Load trained model
- Process single video
- Extract frames (reuse extraction code)
- Run inference
- Return classification with confidence
- Work on both GPU and CPU
```

### 2. inference/batch_inference.py
```python
# Core functionality:
- Process entire folders
- Parallel processing
- Progress tracking
- CSV output with results
- Move/copy files based on classification
```

### 3. evaluation/evaluate_model.py
```python
# Core functionality:
- Load test dataset
- Calculate accuracy, precision, recall, F1
- Per-class metrics
- Confusion matrix
- Save evaluation report
```

### 4. dataset/prepare_dataset.py
```python
# Core functionality:
- Organize videos into class folders
- Create train/val/test splits
- Handle class imbalance
- Generate dataset statistics
- Create sample datasets
```

### 5. visualization/plot_training.py
```python
# Core functionality:
- Load training history JSON
- Plot loss curves
- Plot accuracy curves
- Save high-quality figures
- Compare multiple runs
```

### 6. conversion/export_onnx.py
```python
# Core functionality:
- Load PyTorch model
- Export to ONNX format
- Verify ONNX model
- Optimize for inference
- Support different batch sizes
```

## Tool Design Principles

### 1. Standalone Operation
Each tool should work independently:
```bash
python tools/inference/single_video.py --model best_model.pth --video test.mp4
```

### 2. Consistent Interface
All tools follow similar patterns:
```python
def main():
    args = parse_arguments()
    verify_requirements()
    load_resources()
    process()
    save_results()
```

### 3. Local-First Design
- Work on local machine (3080 Ti)
- Automatic batch size adjustment
- Memory-efficient processing
- CPU fallback support

### 4. Configuration Support
Each tool supports:
- Command-line arguments
- Configuration files
- Environment variables
- Sensible defaults

### 5. Error Handling
- Graceful degradation
- Clear error messages
- Recovery mechanisms
- Logging support

## Implementation Order

### Phase 1: Core Inference (Now)
1. single_video.py - Test model locally
2. batch_inference.py - Process test videos
3. evaluate_model.py - Measure performance

### Phase 2: Production Tools (After Training)
4. prepare_dataset.py - For next training
5. export_onnx.py - For deployment
6. api_server.py - For integration

### Phase 3: Analysis Tools (Later)
7. error_analysis.py - Improve model
8. heatmap_attention.py - Understand model
9. benchmark.py - Optimize performance

## Resource Requirements

### GPU Memory Usage
| Tool | 3080 Ti (12GB) | A40 (48GB) |
|------|----------------|------------|
| single_video.py | Batch 1 | Batch 4 |
| batch_inference.py | Batch 2 | Batch 8 |
| evaluate_model.py | Batch 2 | Batch 8 |
| export_onnx.py | Batch 1 | Batch 1 |

### Processing Speed
| Tool | Videos/Second |
|------|---------------|
| single_video.py | 0.5-1 |
| batch_inference.py | 2-4 |
| live_inference.py | 5-10 |

## Dependencies
```python
# Core
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Deployment
onnx>=1.14.0
onnxruntime>=1.15.0
gradio>=3.40.0
fastapi>=0.100.0

# Utilities
tqdm>=4.65.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

## Testing Strategy

### Unit Tests
- Test each tool independently
- Mock model and data
- Verify outputs

### Integration Tests
- Test tool chains
- End-to-end workflows
- Performance benchmarks

### Validation Tests
- Compare with training results
- Verify accuracy metrics
- Check reproducibility

## Next Steps

1. Wait for model training to complete
2. Download best_model_768.pth
3. Implement single_video.py first
4. Test on local machine with 3080 Ti
5. Iterate and improve

## Success Metrics

- [ ] Inference works on local GPU
- [ ] Batch processing handles 1000+ videos
- [ ] Evaluation matches training metrics
- [ ] ONNX export reduces size by 50%
- [ ] API serves 10+ requests/second
- [ ] All tools have documentation
- [ ] Error handling prevents crashes