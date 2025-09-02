#!/bin/bash
# Reliable dataset download script with verification

echo "=========================================="
echo "Dataset Download with Verification"
echo "=========================================="

# Check if GOOGLE_DRIVE_ID is set
if [ -z "$GOOGLE_DRIVE_ID" ]; then
    echo "ERROR: GOOGLE_DRIVE_ID not set!"
    echo "Usage: export GOOGLE_DRIVE_ID='your_file_id'"
    exit 1
fi

# Change to training directory
cd /workspace/training
echo "Working directory: $(pwd)"

# Remove any existing partial downloads
rm -f sample_dataset.zip sample_dataset.zip.tmp

# Download with gdown --fuzzy (REQUIRED for large files)
echo "Downloading from Google Drive (this may take 5-10 minutes)..."
echo "File ID: $GOOGLE_DRIVE_ID"

gdown --fuzzy "https://drive.google.com/uc?id=${GOOGLE_DRIVE_ID}" -O sample_dataset.zip

# Verify download succeeded
if [ ! -f sample_dataset.zip ]; then
    echo "ERROR: Download failed - file not found"
    exit 1
fi

# Check file size
SIZE=$(stat -c%s "sample_dataset.zip" 2>/dev/null || echo "0")
SIZE_GB=$(echo "scale=2; $SIZE / 1073741824" | bc)

echo "Downloaded file size: ${SIZE_GB}GB"

if (( $(echo "$SIZE_GB < 1" | bc -l) )); then
    echo "ERROR: File too small (${SIZE_GB}GB). Likely got Google's warning page."
    echo "Trying alternative download method..."
    
    # Try with Python gdown
    python3 << EOF
import gdown
url = 'https://drive.google.com/uc?id=${GOOGLE_DRIVE_ID}'
output = 'sample_dataset.zip'
print("Downloading with Python gdown...")
gdown.download(url, output, quiet=False, fuzzy=True)
EOF
    
    # Check size again
    SIZE=$(stat -c%s "sample_dataset.zip" 2>/dev/null || echo "0")
    SIZE_GB=$(echo "scale=2; $SIZE / 1073741824" | bc)
    
    if (( $(echo "$SIZE_GB < 1" | bc -l) )); then
        echo "ERROR: Still failed. File is only ${SIZE_GB}GB"
        exit 1
    fi
fi

echo "✓ Download successful: ${SIZE_GB}GB"

# Extract dataset
echo "Extracting dataset..."
unzip -q sample_dataset.zip

# Verify extraction
if [ ! -d "sample_dataset" ]; then
    echo "ERROR: Extraction failed - directory not found"
    exit 1
fi

# Count videos
TOTAL_VIDEOS=$(find sample_dataset -name "*.mp4" -type f | wc -l)
echo "✓ Extraction successful: Found $TOTAL_VIDEOS videos"

# Show structure
echo "Dataset structure:"
ls -la sample_dataset/

echo "=========================================="
echo "Dataset ready for processing!"
echo "=========================================="