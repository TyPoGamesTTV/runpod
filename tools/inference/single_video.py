#!/usr/bin/env python3
"""
Single Video Inference Tool
Classify a single video using trained 768x768 model
Works on both local GPU (3080 Ti) and CPU
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import argparse
import time
import sys
import os

# Add parent directory to path for model import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class VideoViT(nn.Module):
    """Same architecture as training - must match exactly!"""
    def __init__(self, num_frames=16, num_classes=3, resolution=768):
        super().__init__()
        self.num_frames = num_frames
        self.resolution = resolution
        self.patch_size = 32
        self.num_patches = (resolution // self.patch_size) ** 2
        self.dim = 768
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(1, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames * self.num_patches, self.dim) * 0.02)
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.dim,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=12
        )
        
        # Classification head
        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, num_classes)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Embed patches for each frame
        x = x.reshape(B * T, C, H, W)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Reshape to (B, T*num_patches, dim)
        x = x.reshape(B, T * self.num_patches, self.dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling and classification
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.head(x)
        
        return x

class VideoClassifier:
    def __init__(self, model_path, device=None, resolution=768, num_frames=16):
        """
        Initialize the video classifier
        
        Args:
            model_path: Path to trained .pth file
            device: 'cuda', 'cpu', or None (auto-detect)
            resolution: Frame resolution (768 default)
            num_frames: Number of frames to extract (16 default)
        """
        self.resolution = resolution
        self.num_frames = num_frames
        self.class_names = ['1_Safe', '2_Unsafe', '3_Explicit']
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = VideoViT(num_frames=num_frames, num_classes=3, resolution=resolution)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Print device info
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Using GPU: {gpu_name} ({vram:.1f}GB)")
        else:
            print("Using CPU (this will be slower)")
    
    def extract_frames(self, video_path):
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f} seconds")
        
        if total_frames < self.num_frames:
            print(f"Warning: Video has only {total_frames} frames, padding to {self.num_frames}")
        
        # Sample frames evenly across video
        indices = np.linspace(0, max(0, total_frames-1), self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                # Convert to grayscale
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Resize to target resolution
                frame = cv2.resize(frame, (self.resolution, self.resolution))
                frames.append(frame)
            else:
                # Padding with zeros if frame read fails
                frames.append(np.zeros((self.resolution, self.resolution), dtype=np.uint8))
        
        cap.release()
        
        # Convert to numpy array
        frames = np.array(frames, dtype=np.float32) / 255.0
        return frames
    
    def classify(self, video_path, return_probs=False):
        """
        Classify a single video
        
        Args:
            video_path: Path to video file
            return_probs: If True, return probabilities for all classes
        
        Returns:
            If return_probs=False: (class_name, confidence)
            If return_probs=True: (class_name, confidence, all_probabilities)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"\nProcessing: {video_path.name}")
        
        # Extract frames
        start_time = time.time()
        frames = self.extract_frames(video_path)
        extract_time = time.time() - start_time
        print(f"Frame extraction: {extract_time:.2f} seconds")
        
        # Prepare for model
        frames = np.expand_dims(frames, axis=1)  # Add channel dimension
        frames = np.expand_dims(frames, axis=0)  # Add batch dimension
        frames_tensor = torch.from_numpy(frames).to(self.device)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(frames_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        inference_time = time.time() - start_time
        print(f"Inference: {inference_time:.3f} seconds")
        
        # Get prediction
        probs_np = probs.cpu().numpy()[0]
        pred_idx = np.argmax(probs_np)
        pred_class = self.class_names[pred_idx]
        pred_conf = probs_np[pred_idx]
        
        if return_probs:
            return pred_class, pred_conf, probs_np
        return pred_class, pred_conf
    
    def classify_batch(self, video_paths, batch_size=1):
        """
        Classify multiple videos
        
        Args:
            video_paths: List of video paths
            batch_size: Number of videos to process at once (memory dependent)
        
        Returns:
            List of (video_path, class_name, confidence)
        """
        results = []
        
        for i in range(0, len(video_paths), batch_size):
            batch_paths = video_paths[i:i+batch_size]
            
            # Extract frames for batch
            batch_frames = []
            valid_paths = []
            
            for video_path in batch_paths:
                try:
                    frames = self.extract_frames(video_path)
                    frames = np.expand_dims(frames, axis=1)  # Add channel
                    batch_frames.append(frames)
                    valid_paths.append(video_path)
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    results.append((video_path, "ERROR", 0.0))
            
            if not batch_frames:
                continue
            
            # Stack and process batch
            batch_tensor = torch.from_numpy(np.array(batch_frames)).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
            
            # Extract results
            probs_np = probs.cpu().numpy()
            for path, prob in zip(valid_paths, probs_np):
                pred_idx = np.argmax(prob)
                pred_class = self.class_names[pred_idx]
                pred_conf = prob[pred_idx]
                results.append((path, pred_class, pred_conf))
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Classify videos using trained model')
    parser.add_argument('video', help='Path to video file or directory')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='Device to use (auto-detect if not specified)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for multiple videos')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for classification')
    parser.add_argument('--show-all', action='store_true', help='Show probabilities for all classes')
    parser.add_argument('--resolution', type=int, default=768, help='Frame resolution (must match training)')
    parser.add_argument('--num-frames', type=int, default=16, help='Number of frames to extract')
    parser.add_argument('--output', help='Output CSV file for batch processing')
    
    args = parser.parse_args()
    
    # Initialize classifier
    try:
        classifier = VideoClassifier(
            model_path=args.model,
            device=args.device,
            resolution=args.resolution,
            num_frames=args.num_frames
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1
    
    # Process input
    input_path = Path(args.video)
    
    if input_path.is_file():
        # Single video
        try:
            if args.show_all:
                pred_class, confidence, all_probs = classifier.classify(input_path, return_probs=True)
                print(f"\nResults for {input_path.name}:")
                print(f"  Prediction: {pred_class} ({confidence:.2%})")
                print(f"  All probabilities:")
                for i, (class_name, prob) in enumerate(zip(classifier.class_names, all_probs)):
                    print(f"    {class_name}: {prob:.2%}")
            else:
                pred_class, confidence = classifier.classify(input_path)
                print(f"\nClassification: {pred_class}")
                print(f"Confidence: {confidence:.2%}")
                
                if confidence < args.threshold:
                    print(f"⚠️  Low confidence (below {args.threshold:.0%} threshold)")
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return 1
    
    elif input_path.is_dir():
        # Batch processing
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.glob(f'*{ext}'))
            video_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"No video files found in {input_path}")
            return 1
        
        print(f"Found {len(video_files)} videos to process")
        
        # Process videos
        results = classifier.classify_batch(video_files, batch_size=args.batch_size)
        
        # Display results
        print("\n" + "="*60)
        print("BATCH RESULTS")
        print("="*60)
        
        for video_path, pred_class, confidence in results:
            status = "✓" if confidence >= args.threshold else "⚠"
            print(f"{status} {video_path.name}: {pred_class} ({confidence:.2%})")
        
        # Save to CSV if requested
        if args.output:
            import csv
            with open(args.output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Video', 'Classification', 'Confidence'])
                for video_path, pred_class, confidence in results:
                    writer.writerow([video_path.name, pred_class, f"{confidence:.4f}"])
            print(f"\nResults saved to {args.output}")
    
    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())