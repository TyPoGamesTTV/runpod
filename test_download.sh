#!/bin/bash
# Test Google Drive download before running full pipeline

echo "=========================================="
echo "Google Drive Download Test"
echo "=========================================="

# Check if gdown is installed
python3 -c "import gdown" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ gdown not installed! Installing..."
    pip install gdown
else
    echo "✅ gdown is installed"
fi

# Check if GOOGLE_DRIVE_ID is set
if [ -z "$GOOGLE_DRIVE_ID" ]; then
    echo "❌ GOOGLE_DRIVE_ID not set!"
    echo ""
    echo "To set it:"
    echo "  export GOOGLE_DRIVE_ID='your_file_id_here'"
    echo ""
    echo "Get the ID from your Google Drive share link:"
    echo "  https://drive.google.com/file/d/FILE_ID_HERE/view"
    exit 1
else
    echo "✅ GOOGLE_DRIVE_ID is set: $GOOGLE_DRIVE_ID"
fi

# Show the download URL
echo ""
echo "Download URL that will be used:"
echo "https://drive.google.com/uc?id=${GOOGLE_DRIVE_ID}"

# Test if we can reach Google Drive
echo ""
echo "Testing connection to Google Drive..."
curl -s -o /dev/null -w "%{http_code}" "https://drive.google.com" > /tmp/status_code
STATUS=$(cat /tmp/status_code)

if [ "$STATUS" = "200" ]; then
    echo "✅ Google Drive is reachable"
else
    echo "⚠️  Unexpected status code: $STATUS"
fi

echo ""
echo "=========================================="
echo "Test complete! Ready to download."
echo "Run: bash run_training.sh"
echo "=========================================="