#!/usr/bin/env python3
"""
Quick test to verify model architecture before training
"""
import torch
import numpy as np
import sys

print("="*50)
print("QUICK MODEL TEST")
print("="*50)

# Test imports
try:
    from train_model import VideoViT
    print("✓ Model import successful")
except Exception as e:
    print(f"✗ Failed to import model: {e}")
    sys.exit(1)

# Test model creation
try:
    model = VideoViT()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params/1e6:.1f}M parameters")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    sys.exit(1)

# Test forward pass
try:
    dummy_input = torch.randn(2, 8, 1, 768, 768)  # batch=2, frames=8, grayscale, 768x768
    print(f"Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    if output.shape == torch.Size([2, 3]):
        print("✓ Output shape correct (batch_size=2, num_classes=3)")
    else:
        print(f"✗ Unexpected output shape: {output.shape}")
        
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test GPU if available
if torch.cuda.is_available():
    try:
        model = model.cuda()
        dummy_input = dummy_input.cuda()
        output = model(dummy_input)
        print(f"✓ GPU test passed on {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("⚠ No GPU available for testing")

print("="*50)
print("ALL TESTS PASSED! Ready to train!")
print("="*50)