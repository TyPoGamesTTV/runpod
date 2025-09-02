#!/usr/bin/env python3
"""
768x768 Grayscale Vision Transformer Training
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/training.log'),
        logging.StreamHandler()
    ]
)

class VideoDataset(Dataset):
    def __init__(self, root_dir='frames_768', split='train', train_ratio=0.85):
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
        data = np.load(npz_path)
        frames = data['frames'][:8]  # Use first 8 of 16 frames
        frames = frames.astype(np.float32) / 255.0
        frames = np.expand_dims(frames, axis=1)  # Add channel dim
        return torch.from_numpy(frames), label

class VideoViT(nn.Module):
    def __init__(self, num_classes=3, num_frames=8, resolution=768, patch_size=32):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.num_patches = (resolution // patch_size) ** 2  # 576 patches for 768/32
        self.embed_dim = 768
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(1, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        total_patches = self.num_frames * self.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, total_patches + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(12)
        ])
        
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Process all frames
        x = x.reshape(B * T, C, H, W)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(B, T * x.shape[1], x.shape[2])
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :x.shape[1]]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.head(x[:, 0])

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check GPU
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Adjust batch size based on GPU memory
        if gpu_memory >= 40:  # A40, A100
            BATCH_SIZE = 16
        elif gpu_memory >= 24:  # 4090, 3090
            BATCH_SIZE = 8
        else:
            BATCH_SIZE = 4
    else:
        logging.error("No GPU found!")
        return
    
    EPOCHS = 30
    BASE_LR = 1e-4
    
    logging.info(f"Training config: Batch={BATCH_SIZE}, Epochs={EPOCHS}")
    
    # Create datasets
    train_dataset = VideoDataset('frames_768', split='train')
    val_dataset = VideoDataset('frames_768', split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model
    model = VideoViT().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params/1e6:.1f}M")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for frames, labels in pbar:
            frames, labels = frames.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 
                              'Acc': f'{100.*train_correct/train_total:.2f}%'})
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, '/workspace/models/best_model_768.pth')
            logging.info(f'Saved best model (Val: {val_acc:.2f}%)')
        
        scheduler.step()
        
        elapsed = (time.time() - start_time) / 60
        logging.info(f'Epoch {epoch+1}: Train={train_acc:.2f}%, Val={val_acc:.2f}% (Best={best_acc:.2f}%) | Time={elapsed:.1f}m')
    
    # Save final model
    torch.save(model.state_dict(), '/workspace/models/final_model_768.pth')
    logging.info(f"Training complete! Best accuracy: {best_acc:.2f}%")
    
    return best_acc

if __name__ == "__main__":
    accuracy = train()
    print(f"\nTraining finished! Best validation accuracy: {accuracy:.2f}%")
    print(f"Models saved to /workspace/models/")