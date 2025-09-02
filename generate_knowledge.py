#!/usr/bin/env python3
"""
Teacher Knowledge Generator
Generates soft labels from trained models for knowledge distillation
Runs after each model completes training
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json
import time
import sys
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Import model architectures based on type
def get_model_class(model_type):
    """Get the appropriate model class based on type"""
    if '512' in model_type:
        from train_model_512_rgb import VideoViT_512_RGB
        return VideoViT_512_RGB
    elif 'rgb' in model_type.lower():
        from train_model_rgb import VideoViT_RGB
        return VideoViT_RGB
    else:
        from train_model import VideoViT
        return VideoViT

class KnowledgeGenerator:
    def __init__(self, model_path, model_type='768_grayscale', batch_size=8):
        """
        Initialize knowledge generator
        
        Args:
            model_path: Path to trained model
            model_type: Type of model (768_grayscale, 768_rgb, 512_rgb)
            batch_size: Batch size for inference
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        logging.info(f"Loading model from {model_path}")
        self.model = self.load_model()
        
        # Get dataset info
        self.resolution = 512 if '512' in model_type else 768
        self.channels = 3 if 'rgb' in model_type.lower() else 1
        self.num_frames = 16
        
        logging.info(f"Model type: {model_type}")
        logging.info(f"Resolution: {self.resolution}, Channels: {self.channels}")
        
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"GPU: {gpu_name} ({vram:.1f}GB)")
            
            # Adjust batch size based on model type and VRAM
            if vram < 40:
                if '768' in model_type and 'rgb' in model_type:
                    self.batch_size = min(self.batch_size, 4)
                elif '768' in model_type:
                    self.batch_size = min(self.batch_size, 6)
                elif '512' in model_type:
                    self.batch_size = min(self.batch_size, 12)
                logging.info(f"Adjusted batch size to {self.batch_size} for {vram:.1f}GB VRAM")
    
    def load_model(self):
        """Load the appropriate model"""
        ModelClass = get_model_class(self.model_type)
        
        # Initialize model
        if '512' in self.model_type:
            model = ModelClass(num_frames=16, num_classes=3)
        else:
            model = ModelClass()
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        logging.info(f"Model loaded: {total_params:.1f}M parameters")
        
        return model
    
    def process_dataset(self, dataset_path, output_path):
        """Generate knowledge for entire dataset"""
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine frame folder based on model type
        if '512' in self.model_type and 'rgb' in self.model_type:
            frame_dirs = ['frames_512_rgb', 'frames_512_rgb_smart']
        elif '768' in self.model_type and 'rgb' in self.model_type:
            frame_dirs = ['frames_768_rgb', 'frames_768_rgb_smart']
        else:
            frame_dirs = ['frames_768', 'frames_768_gray_smart']
        
        # Find the frame directory
        frame_dir = None
        for dir_name in frame_dirs:
            test_path = dataset_path / dir_name
            if test_path.exists():
                frame_dir = test_path
                logging.info(f"Found frame directory: {frame_dir}")
                break
        
        if frame_dir is None:
            logging.error(f"No frame directory found. Looked for: {frame_dirs}")
            return
        
        # Collect all frame files
        all_frames = []
        class_mapping = {}
        for class_idx, class_name in enumerate(['1_Safe', '2_Unsafe', '3_Explicit']):
            class_dir = frame_dir / class_name
            if class_dir.exists():
                for npz_path in class_dir.glob('*.npz'):
                    all_frames.append((npz_path, class_idx, class_name))
                    class_mapping[npz_path.stem] = class_idx
        
        logging.info(f"Found {len(all_frames)} frame files to process")
        
        # Process in batches
        knowledge_data = {
            'soft_labels': {},
            'embeddings': {},
            'predictions': {},
            'confidences': {},
            'ground_truth': {}
        }
        
        start_time = time.time()
        
        with torch.no_grad():
            for i in tqdm(range(0, len(all_frames), self.batch_size), desc="Generating knowledge"):
                batch_files = all_frames[i:i+self.batch_size]
                batch_frames = []
                batch_names = []
                batch_labels = []
                
                # Load batch
                for npz_path, class_idx, class_name in batch_files:
                    try:
                        data = np.load(npz_path)
                        frames = data['frames']
                        
                        # Ensure correct shape
                        if self.channels == 3 and len(frames.shape) == 3:
                            # Need to add channel dimension
                            frames = np.stack([frames, frames, frames], axis=1)
                        elif self.channels == 1 and len(frames.shape) == 3:
                            # Add channel dimension
                            frames = np.expand_dims(frames, axis=1)
                        
                        # Normalize
                        frames = frames.astype(np.float32) / 255.0
                        
                        # Handle different formats
                        if frames.shape[1] != self.channels:
                            # Transpose if needed
                            if frames.shape[-1] == self.channels:
                                frames = np.transpose(frames, (0, 3, 1, 2))
                        
                        batch_frames.append(frames)
                        batch_names.append(npz_path.stem)
                        batch_labels.append(class_idx)
                        
                    except Exception as e:
                        logging.error(f"Error loading {npz_path}: {e}")
                        continue
                
                if not batch_frames:
                    continue
                
                # Convert to tensor
                batch_tensor = torch.from_numpy(np.array(batch_frames)).to(self.device)
                
                # Get model outputs
                outputs = self.model(batch_tensor)
                
                # Get probabilities and predictions
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1)
                confidences = torch.max(probs, dim=1)[0]
                
                # Store results
                for j, name in enumerate(batch_names):
                    knowledge_data['soft_labels'][name] = probs[j].cpu().numpy().tolist()
                    knowledge_data['predictions'][name] = int(predictions[j].cpu())
                    knowledge_data['confidences'][name] = float(confidences[j].cpu())
                    knowledge_data['ground_truth'][name] = batch_labels[j]
                    
                    # Optional: Store last layer embeddings (before classification head)
                    # This would require modifying the model to return embeddings
                    # knowledge_data['embeddings'][name] = embeddings[j].cpu().numpy().tolist()
                
                # Clear memory periodically
                if i % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # Calculate statistics
        correct = sum(1 for name in knowledge_data['predictions'] 
                     if knowledge_data['predictions'][name] == knowledge_data['ground_truth'][name])
        total = len(knowledge_data['predictions'])
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Save knowledge
        output_file = output_path / f"knowledge_{self.model_type}.pth"
        torch.save(knowledge_data, output_file)
        logging.info(f"Knowledge saved to {output_file}")
        
        # Save metadata
        metadata = {
            'model_path': str(self.model_path),
            'model_type': self.model_type,
            'resolution': self.resolution,
            'channels': self.channels,
            'num_frames': self.num_frames,
            'total_samples': total,
            'accuracy': accuracy,
            'generation_time': time.time() - start_time,
            'batch_size': self.batch_size,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = output_path / f"knowledge_{self.model_type}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Generation complete! Accuracy: {accuracy:.2f}%")
        logging.info(f"Time: {(time.time() - start_time)/60:.1f} minutes")
        logging.info(f"Knowledge size: {output_file.stat().st_size / 1024**2:.1f} MB")
        
        # Print summary
        print("\n" + "="*60)
        print(f"Knowledge Generation Complete: {self.model_type}")
        print("="*60)
        print(f"Samples processed: {total}")
        print(f"Teacher accuracy: {accuracy:.2f}%")
        print(f"Output file: {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024**2:.1f} MB")
        print(f"Time taken: {(time.time() - start_time)/60:.1f} minutes")
        print("="*60)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate teacher knowledge for distillation')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--type', required=True, 
                       choices=['768_grayscale', '768_rgb', '512_rgb'],
                       help='Model type')
    parser.add_argument('--dataset', default='/workspace/training', 
                       help='Dataset path with extracted frames')
    parser.add_argument('--output', default='/workspace/knowledge',
                       help='Output directory for knowledge files')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    generator = KnowledgeGenerator(
        model_path=args.model,
        model_type=args.type,
        batch_size=args.batch_size
    )
    
    generator.process_dataset(args.dataset, args.output)

if __name__ == "__main__":
    main()