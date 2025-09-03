#!/bin/bash
# Training wrapper with automatic logging
LOG_FILE="/workspace/training_$(date +%Y%m%d_%H%M%S).log"
echo "Starting X3D training with logging to: $LOG_FILE"
echo "Monitor with: tail -f $LOG_FILE"
echo ""
python /workspace/train_x3d_tuned.py 2>&1 | tee $LOG_FILE
echo ""
echo "Training complete. Log saved to: $LOG_FILE"
echo "Best model saved to: /workspace/best_x3d_tuned.pth"