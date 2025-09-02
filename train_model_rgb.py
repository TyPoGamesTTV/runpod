#!/usr/bin/env python3
"""
768x768 RGB Vision Transformer Training
Color version for better skin/content detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time
import json
import random
import os
import gc

# RGB CONFIGURATION - Color training
BATCH_SIZE = 6  # Reduced from 8 due to 3x memory for RGB
NUM_WORKERS = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_FRAMES = 16
NUM_CLASSES = 3
RESOLUTION = 768
CHANNELS = 3  # RGB instead of grayscale

# Create directories safely
os.makedirs('/workspace/logs', exist_ok=True)
os.makedirs('/workspace/models', exist_ok=True)
os.makedirs('/workspace/checkpoints', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/training_rgb.log'),
        logging.StreamHandler()
    ]
)

class VideoDataset(Dataset):
    def __init__(self, root_dir='frames_768_rgb', split='train', train_ratio=0.85):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.class_to_idx = {'1_Safe': 0, '2_Unsafe': 1, '3_Explicit': 2}
        
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for npz_path in class_dir.glob('*.npz'):
                    self.samples.append((npz_path, class_idx))
        
        random.seed(42)
        random.shuffle(self.samples)
        
        split_idx = int(len(self.samples) * train_ratio)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        logging.info(f"{split.upper()}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npz_path, label = self.samples[idx]
        try:
            # Load with error handling
            data = np.load(npz_path)
            frames = data['frames']  # Should be (16, 768, 768, 3) for RGB
            
            # Ensure we have the right number of frames
            if len(frames) >= NUM_FRAMES:
                frames = frames[:NUM_FRAMES]
            else:
                # Pad if somehow we have fewer frames
                pad_count = NUM_FRAMES - len(frames)
                frames = np.concatenate([frames, np.zeros((pad_count, RESOLUTION, RESOLUTION, CHANNELS), dtype=np.uint8)])
            
            # Normalize to [0, 1]
            frames = frames.astype(np.float32) / 255.0
            
            # Transpose from (T, H, W, C) to (T, C, H, W)
            frames = np.transpose(frames, (0, 3, 1, 2))
            
            return torch.from_numpy(frames), label
        except Exception as e:
            logging.error(f"Error loading {npz_path}: {e}")
            # Return zeros as fallback
            return torch.zeros(NUM_FRAMES, CHANNELS, RESOLUTION, RESOLUTION), label

class VideoViT_RGB(nn.Module):
    def __init__(self, num_frames=NUM_FRAMES, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = 32
        self.num_patches = (RESOLUTION // self.patch_size) ** 2
        self.dim = 768
        
        # Patch embedding - now for RGB (3 channels)
        self.patch_embed = nn.Conv2d(CHANNELS, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        
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
        B, T, C, H, W = x.shape  # (batch, frames, 3, 768, 768)
        
        # Embed patches for each frame
        x = x.reshape(B * T, C, H, W)
        x = self.patch_embed(x)  # (B*T, dim, H/32, W/32)
        x = x.flatten(2).transpose(1, 2)  # (B*T, num_patches, dim)
        
        # Reshape to (B, T*num_patches, dim)
        x = x.reshape(B, T * self.num_patches, self.dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.norm(x)
        x = self.head(x)
        
        return x

def train():
    # Check GPU and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"GPU: {gpu_name} ({vram:.1f}GB)")
        logging.info("Training RGB model (3 channels)")
        
        # Clear cache before starting
        torch.cuda.empty_cache()
        gc.collect()
    
    # Adjust batch size based on available VRAM
    if device.type == 'cuda' and vram < 40:
        actual_batch_size = 4  # Even smaller for RGB on lower VRAM
        logging.warning(f"Low VRAM detected ({vram:.1f}GB), reducing batch size to {actual_batch_size}")
    else:
        actual_batch_size = BATCH_SIZE
    
    logging.info(f"Training config: Batch={actual_batch_size}, Epochs={EPOCHS}, Frames={NUM_FRAMES}, RGB")
    
    # Create datasets
    try:
        train_dataset = VideoDataset('frames_768_rgb', 'train')
        val_dataset = VideoDataset('frames_768_rgb', 'val')
    except Exception as e:
        logging.error(f"Failed to load datasets: {e}")
        logging.info("Checking alternative paths...")
        train_dataset = VideoDataset('/workspace/training/frames_768_rgb', 'train')
        val_dataset = VideoDataset('/workspace/training/frames_768_rgb', 'val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=actual_batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=actual_batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create model
    model = VideoViT_RGB().to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f"Model created: {total_params:.1f}M parameters (RGB version)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Mixed precision training
    use_amp = device.type == 'cuda' and torch.cuda.get_device_capability()[0] >= 7
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        logging.info("Using mixed precision training (AMP)")
    
    best_val_acc = 0
    training_history = []
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (frames, labels) in enumerate(pbar):
            frames, labels = frames.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            try:
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(frames)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100.*train_correct/train_total:.2f}%"
                })
                
                # Memory management
                if batch_idx % 50 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logging.error("OOM! Clearing cache and skipping batch")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(frames)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logging.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'/workspace/checkpoints/checkpoint_rgb_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc
            }, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '/workspace/models/best_model_768_rgb.pth')
            logging.info(f"Best RGB model saved! Val Acc: {val_acc:.2f}%")
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        })
        
        scheduler.step()
    
    # Save final model and history
    torch.save(model.state_dict(), '/workspace/models/final_model_768_rgb.pth')
    with open('/workspace/models/training_history_rgb.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logging.info(f"RGB Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc

if __name__ == "__main__":
    try:
        accuracy = train()
        print(f"\n✅ RGB Training successful! Best accuracy: {accuracy:.2f}%")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")