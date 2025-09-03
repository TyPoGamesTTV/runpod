#!/usr/bin/env python3
"""
X3D-M Tuned Training - Aggressive Regularization Edition
Based on analysis of overfitting patterns from previous run
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging
import json
from tqdm import tqdm
import random
from collections import Counter
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Optimized Configuration
BATCH_SIZE = 16         # Increased for gradient stability
NUM_EPOCHS = 25         # Reduced - we peak early
BASE_LR = 3e-5          # Much more conservative
WARMUP_EPOCHS = 5       # Extended warmup
WEIGHT_DECAY = 5e-4     # 50x stronger regularization
LABEL_SMOOTHING = 0.1   # Prevent overconfidence
GRADIENT_CLIP = 0.5     # Aggressive clipping
DROPOUT_RATE = 0.3      # Add dropout everywhere
PATIENCE = 4            # Early stopping patience

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_FRAMES = 16
FRAME_SIZE = 224
NUM_CLASSES = 3

# Class weights - equal for all (dataset already balanced)
CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0]).to(DEVICE)

class VideoDataset(Dataset):
    """Dataset with temporal augmentation"""
    def __init__(self, root_dir='/workspace/frames_x3d_clean', split='train', train_ratio=0.85):
        self.root_dir = Path(root_dir)
        self.samples = []
        self.classes = ['1_Safe', '2_Unsafe', '3_Explicit']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.split = split
        
        # Load all samples
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for npz_file in class_dir.glob('*.npz'):
                    self.samples.append((npz_file, self.class_to_idx[class_name]))
        
        # Split dataset
        random.seed(42)
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * train_ratio)
        
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        logging.info(f"{split} dataset: {len(self.samples)} videos")
        
        # Count class distribution
        class_counts = Counter([s[1] for s in self.samples])
        for cls, idx in self.class_to_idx.items():
            count = class_counts.get(idx, 0)
            logging.info(f"  {cls}: {count} videos ({count/len(self.samples)*100:.1f}%)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        npz_path, label = self.samples[idx]
        
        # Load preprocessed frames
        data = np.load(npz_path)
        frames = data['frames']  # (16, 224, 224, 3)
        
        # Temporal augmentation for training
        if self.split == 'train':
            # Random frame dropping (temporal dropout)
            if random.random() < 0.1:  # 10% chance
                drop_indices = random.sample(range(NUM_FRAMES), 2)  # Drop 2 random frames
                for idx in drop_indices:
                    if idx > 0:
                        frames[idx] = frames[idx-1]  # Replace with previous frame
            
            # Random temporal shift
            if random.random() < 0.3:  # 30% chance
                shift = random.randint(-2, 2)
                if shift != 0:
                    frames = np.roll(frames, shift, axis=0)
        
        # Convert to tensor and normalize
        frames = torch.FloatTensor(frames).permute(3, 0, 1, 2)  # (C, T, H, W)
        frames = frames / 255.0
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        frames = (frames - mean) / std
        
        return frames, label

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) for regularization"""
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class SqueezeExcitation3D(nn.Module):
    """3D Squeeze-and-Excitation block with dropout"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
    def forward(self, x):
        b, c, t, h, w = x.shape
        y = self.avgpool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.dropout(y)  # Add dropout
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1, 1)
        return x * y

class X3DBlock(nn.Module):
    """X3D building block with dropout and stochastic depth"""
    def __init__(self, in_channels, out_channels, stride=1, use_se=True, drop_path=0.0):
        super().__init__()
        
        # Depthwise 3D conv
        self.conv1 = nn.Conv3d(in_channels, in_channels, 
                               kernel_size=(3, 3, 3),
                               stride=stride,
                               padding=1,
                               groups=in_channels,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(in_channels)
        
        # Pointwise conv
        self.conv2 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # SE block
        self.se = SqueezeExcitation3D(out_channels) if use_se else None
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        # Residual connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.se is not None:
            out = self.se(out)
        
        # Apply stochastic depth
        out = self.drop_path(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out

class X3D_M_Regularized(nn.Module):
    """X3D-M model with heavy regularization"""
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(3, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(24),
            nn.ReLU(),
            nn.Dropout3d(DROPOUT_RATE * 0.5)  # Light dropout early
        )
        
        # Stages with progressive stochastic depth
        self.stage1 = self._make_stage(24, 54, 3, stride=2, drop_path=0.05)
        self.stage2 = self._make_stage(54, 108, 5, stride=2, drop_path=0.10)
        self.stage3 = self._make_stage(108, 216, 11, stride=2, drop_path=0.15)
        self.stage4 = self._make_stage(216, 432, 6, stride=2, drop_path=0.20)
        
        # Head with strong dropout
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(432, num_classes)
        
        self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride, drop_path):
        layers = [X3DBlock(in_channels, out_channels, stride=stride, drop_path=drop_path)]
        for i in range(num_blocks - 1):
            # Progressive drop path within stage
            dp = drop_path * (i + 1) / num_blocks
            layers.append(X3DBlock(out_channels, out_channels, drop_path=dp))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Strong dropout before final layer
        x = self.fc(x)
        
        return x

class SmartTrainer:
    """Trainer with early stopping and adaptive features"""
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        
        # Loss with label smoothing and class weights
        self.criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, label_smoothing=LABEL_SMOOTHING)
        
        # Tracking
        self.best_val_acc = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def get_lr(self, epoch):
        """Warmup + cosine annealing"""
        if epoch < WARMUP_EPOCHS:
            # Linear warmup
            return BASE_LR * (epoch + 1) / WARMUP_EPOCHS
        else:
            # Cosine annealing
            progress = (epoch - WARMUP_EPOCHS) / (NUM_EPOCHS - WARMUP_EPOCHS)
            return BASE_LR * (1 + np.cos(np.pi * progress)) / 2
    
    def update_lr(self, epoch):
        """Update learning rate"""
        lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def train_epoch(self, epoch):
        """Train one epoch with aggressive clipping"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * NUM_CLASSES
        class_total = [0] * NUM_CLASSES
        
        lr = self.update_lr(epoch)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [LR={lr:.1e}]')
        
        for frames, labels in pbar:
            frames, labels = frames.to(DEVICE), labels.to(DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                             'acc': f'{100.*correct/total:.1f}%'})
        
        # Calculate per-class accuracy
        class_acc = []
        for i in range(NUM_CLASSES):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                class_acc.append(acc)
        
        avg_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return avg_loss, train_acc, class_acc
    
    def validate(self, epoch):
        """Validate with early stopping check"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * NUM_CLASSES
        class_total = [0] * NUM_CLASSES
        
        with torch.no_grad():
            for frames, labels in tqdm(self.val_loader, desc='Validation'):
                frames, labels = frames.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        avg_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        # Calculate per-class accuracy
        class_acc = []
        for i in range(NUM_CLASSES):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                class_acc.append(acc)
        
        # Check for improvement
        improved = False
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_loss = avg_loss
            improved = True
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_loss,
                'class_acc': class_acc
            }, '/workspace/best_x3d_tuned.pth')
            logging.info(f"💾 Saved best model: {val_acc:.2f}%")
        
        # Early stopping logic
        if not improved:
            self.patience_counter += 1
            if self.patience_counter >= PATIENCE:
                logging.info(f"Early stopping triggered! Best val acc: {self.best_val_acc:.2f}%")
                return avg_loss, val_acc, class_acc, True  # Signal to stop
        else:
            self.patience_counter = 0
        
        return avg_loss, val_acc, class_acc, False

def main():
    logging.info(f"Starting X3D-M Tuned Training on {DEVICE}")
    logging.info(f"Hyperparameters: LR={BASE_LR}, WD={WEIGHT_DECAY}, Dropout={DROPOUT_RATE}")
    logging.info(f"Label Smoothing={LABEL_SMOOTHING}, Batch={BATCH_SIZE}")
    
    # Create datasets
    train_dataset = VideoDataset(split='train')
    val_dataset = VideoDataset(split='val')
    
    if len(train_dataset) == 0:
        logging.error("No training data found! Check extraction status.")
        return
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model and trainer
    model = X3D_M_Regularized(num_classes=NUM_CLASSES).to(DEVICE)
    trainer = SmartTrainer(model, train_loader, val_loader)
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    logging.info(f"Training samples: {len(train_dataset)}")
    logging.info(f"Validation samples: {len(val_dataset)}")
    
    # Training loop with early stopping
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        
        # Train
        train_loss, train_acc, train_class_acc = trainer.train_epoch(epoch)
        
        # Validate
        val_loss, val_acc, val_class_acc, should_stop = trainer.validate(epoch)
        
        # Log results
        elapsed = time.time() - start
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} ({elapsed:.1f}s)")
        logging.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logging.info(f"    Per-class: {[f'{a:.1f}%' for a in train_class_acc]}")
        logging.info(f"  Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        logging.info(f"    Per-class: {[f'{a:.1f}%' for a in val_class_acc]}")
        logging.info(f"  Patience: {trainer.patience_counter}/{PATIENCE}")
        
        if should_stop:
            break
    
    logging.info(f"Training complete! Best validation accuracy: {trainer.best_val_acc:.2f}%")

if __name__ == '__main__':
    main()