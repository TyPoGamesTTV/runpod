#!/bin/bash
# Generate knowledge from all trained models
# Run this after all models are trained

echo "=========================================="
echo "Generating Knowledge from All Models"
echo "=========================================="

# Create knowledge directory
mkdir -p /workspace/knowledge

# Check which models exist and generate knowledge
echo ""
echo "[1/4] Checking for Grayscale 768 model..."
if [ -f "/workspace/models/best_model_768.pth" ]; then
    echo "Found! Generating knowledge..."
    python3 generate_knowledge.py \
        --model /workspace/models/best_model_768.pth \
        --type 768_grayscale \
        --dataset /workspace/training \
        --output /workspace/knowledge \
        --batch-size 8
else
    echo "Grayscale 768 model not found, skipping..."
fi

echo ""
echo "[2/4] Checking for RGB 512 model..."
if [ -f "/workspace/models/best_model_512_rgb.pth" ]; then
    echo "Found! Generating knowledge..."
    python3 generate_knowledge.py \
        --model /workspace/models/best_model_512_rgb.pth \
        --type 512_rgb \
        --dataset /workspace/training \
        --output /workspace/knowledge \
        --batch-size 16
else
    echo "RGB 512 model not found, skipping..."
fi

echo ""
echo "[3/4] Checking for RGB 768 model (MAIN MODEL)..."
if [ -f "/workspace/models/best_model_768_rgb.pth" ]; then
    echo "Found! Generating knowledge from MAIN MODEL..."
    python3 generate_knowledge.py \
        --model /workspace/models/best_model_768_rgb.pth \
        --type 768_rgb \
        --dataset /workspace/training \
        --output /workspace/knowledge \
        --batch-size 6
else
    echo "RGB 768 model not found, skipping..."
fi

echo ""
echo "[4/4] Checking for any checkpoint models..."
for checkpoint in /workspace/checkpoints/checkpoint*.pth; do
    if [ -f "$checkpoint" ]; then
        echo "Found checkpoint: $checkpoint"
        # Extract type from filename
        if [[ "$checkpoint" == *"512_rgb"* ]]; then
            model_type="512_rgb"
            batch_size=16
        elif [[ "$checkpoint" == *"rgb"* ]]; then
            model_type="768_rgb"
            batch_size=6
        else
            model_type="768_grayscale"
            batch_size=8
        fi
        
        echo "Generating knowledge from checkpoint..."
        python3 generate_knowledge.py \
            --model "$checkpoint" \
            --type "$model_type" \
            --dataset /workspace/training \
            --output /workspace/knowledge \
            --batch-size "$batch_size"
    fi
done

echo ""
echo "=========================================="
echo "Knowledge Generation Complete!"
echo "=========================================="
echo "Knowledge files saved in: /workspace/knowledge/"
ls -lh /workspace/knowledge/

echo ""
echo "Package for download:"
echo "cd /workspace && zip -r knowledge_package.zip knowledge/"