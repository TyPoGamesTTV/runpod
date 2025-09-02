#!/bin/bash
# Master training script with automatic knowledge generation
# Trains models and generates knowledge after each completes

echo "=========================================="
echo "Automated Training + Knowledge Pipeline"
echo "=========================================="

# Function to generate knowledge after training
generate_knowledge() {
    local model_path=$1
    local model_type=$2
    local batch_size=$3
    
    echo ""
    echo "Generating knowledge from $model_type model..."
    python3 generate_knowledge.py \
        --model "$model_path" \
        --type "$model_type" \
        --dataset /workspace/training \
        --output /workspace/knowledge \
        --batch-size "$batch_size"
}

# Track what we're training
TRAIN_GRAYSCALE=false
TRAIN_512_RGB=false
TRAIN_768_RGB=false

# Check what needs training
echo "Checking what models to train..."

if [ ! -f "/workspace/models/best_model_768.pth" ]; then
    TRAIN_GRAYSCALE=true
    echo "✓ Will train: Grayscale 768"
fi

if [ ! -f "/workspace/models/best_model_512_rgb.pth" ]; then
    TRAIN_512_RGB=true
    echo "✓ Will train: RGB 512"
fi

if [ ! -f "/workspace/models/best_model_768_rgb.pth" ]; then
    TRAIN_768_RGB=true
    echo "✓ Will train: RGB 768 (MAIN)"
fi

# Train Grayscale 768
if [ "$TRAIN_GRAYSCALE" = true ]; then
    echo ""
    echo "=========================================="
    echo "[1/3] Training Grayscale 768 Model"
    echo "=========================================="
    
    # Check if frames exist
    if [ ! -d "/workspace/training/frames_768" ]; then
        echo "Extracting grayscale frames..."
        cd /workspace/training
        python3 /workspace/runpod/extract_frames.py
    fi
    
    # Train
    cd /workspace/training
    python3 /workspace/runpod/train_model.py
    
    # Generate knowledge immediately
    if [ -f "/workspace/models/best_model_768.pth" ]; then
        generate_knowledge "/workspace/models/best_model_768.pth" "768_grayscale" 8
    fi
fi

# Train RGB 512
if [ "$TRAIN_512_RGB" = true ]; then
    echo ""
    echo "=========================================="
    echo "[2/3] Training RGB 512 Model (Fast)"
    echo "=========================================="
    
    # Check if intelligent frames exist
    if [ ! -d "/workspace/training/frames_512_rgb_smart" ]; then
        echo "Extracting RGB 512 frames (intelligent selection)..."
        cd /workspace/training
        python3 /workspace/runpod/extract_frames_intelligent.py \
            sample_dataset frames_512_rgb_smart \
            --resolution 512 --color rgb --num-frames 16
    fi
    
    # Update training script to use smart frames
    cd /workspace/training
    sed -i "s/frames_512_rgb/frames_512_rgb_smart/g" /workspace/runpod/train_model_512_rgb.py
    
    # Train
    python3 /workspace/runpod/train_model_512_rgb.py
    
    # Generate knowledge immediately
    if [ -f "/workspace/models/best_model_512_rgb.pth" ]; then
        generate_knowledge "/workspace/models/best_model_512_rgb.pth" "512_rgb" 16
    fi
fi

# Train RGB 768 (MAIN MODEL)
if [ "$TRAIN_768_RGB" = true ]; then
    echo ""
    echo "=========================================="
    echo "[3/3] Training RGB 768 Model (MAIN)"
    echo "=========================================="
    
    # Check if intelligent frames exist
    if [ ! -d "/workspace/training/frames_768_rgb_smart" ]; then
        echo "Extracting RGB 768 frames (intelligent selection)..."
        cd /workspace/training
        python3 /workspace/runpod/extract_frames_intelligent.py \
            sample_dataset frames_768_rgb_smart \
            --resolution 768 --color rgb --num-frames 16
    fi
    
    # Update training script to use smart frames
    cd /workspace/training
    sed -i "s/frames_768_rgb/frames_768_rgb_smart/g" /workspace/runpod/train_model_rgb.py
    
    # Train
    python3 /workspace/runpod/train_model_rgb.py
    
    # Generate knowledge immediately
    if [ -f "/workspace/models/best_model_768_rgb.pth" ]; then
        generate_knowledge "/workspace/models/best_model_768_rgb.pth" "768_rgb" 6
    fi
fi

echo ""
echo "=========================================="
echo "ALL TRAINING AND KNOWLEDGE GENERATION COMPLETE!"
echo "=========================================="

# Package everything for download
echo "Creating download package..."
cd /workspace
zip -r training_results.zip models/ knowledge/ logs/ checkpoints/

echo ""
echo "Package ready: /workspace/training_results.zip"
echo "Download with:"
echo "scp -P [PORT] root@[IP]:/workspace/training_results.zip ./"

# Show final stats
echo ""
echo "Models trained:"
ls -lh /workspace/models/

echo ""
echo "Knowledge generated:"
ls -lh /workspace/knowledge/

echo ""
echo "Total time: $SECONDS seconds"