#!/usr/bin/env python3
"""
Live training monitor - shows progress in real-time
"""
import time
import subprocess
from pathlib import Path
import json

def get_gpu_usage():
    """Get current GPU usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            return {
                'gpu_util': f"{values[0]}%",
                'memory': f"{float(values[1])/1024:.1f}GB / {float(values[2])/1024:.1f}GB"
            }
    except:
        pass
    return None

def get_latest_log_lines(log_file, n=5):
    """Get last n lines from log file"""
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return lines[-n:]
    except:
        return []

def monitor():
    """Main monitoring loop"""
    log_file = Path('/workspace/logs/training.log')
    model_dir = Path('/workspace/models')
    
    print("="*60)
    print("TRAINING MONITOR")
    print("Press Ctrl+C to exit")
    print("="*60)
    
    while True:
        # Clear screen
        print("\033[2J\033[H")  # ANSI escape codes to clear screen
        
        print("="*60)
        print("RUNPOD TRAINING MONITOR")
        print("="*60)
        
        # GPU Status
        gpu = get_gpu_usage()
        if gpu:
            print(f"\n📊 GPU Status:")
            print(f"   Utilization: {gpu['gpu_util']}")
            print(f"   Memory: {gpu['memory']}")
        
        # Model files
        print(f"\n💾 Model Files:")
        if model_dir.exists():
            models = list(model_dir.glob('*.pth'))
            if models:
                for model in models:
                    size = model.stat().st_size / (1024**3)  # GB
                    print(f"   {model.name}: {size:.2f}GB")
            else:
                print("   No models saved yet")
        
        # Latest log lines
        print(f"\n📝 Latest Training Log:")
        log_lines = get_latest_log_lines(log_file, 5)
        if log_lines:
            for line in log_lines:
                print(f"   {line.strip()}")
        else:
            print("   Waiting for training to start...")
        
        # Check for completion
        if log_lines and any("Training complete!" in line for line in log_lines):
            print("\n✅ TRAINING COMPLETE!")
            print("Download your model from /workspace/models/")
            break
        
        time.sleep(2)  # Update every 2 seconds

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")