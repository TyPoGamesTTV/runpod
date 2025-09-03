# Git Repository Update Instructions

## Clean Slate Push to https://github.com/TyPoGamesTTV/runpod

### Step 1: Clone existing repo (if not already)
```bash
git clone https://github.com/TyPoGamesTTV/runpod.git runpod_old
```

### Step 2: Create new clean repo
```bash
# Go to the runpod_setup folder
cd D:/ContentClassifierPro/ML_PLATFORM/VideoMLPlatform/runpod_setup/

# Initialize new git repo
git init

# Add remote
git remote add origin https://github.com/TyPoGamesTTV/runpod.git
```

### Step 3: Clear remote and force push
```bash
# Add all our clean files
git add .
git commit -m "X3D Video Classifier - Production Ready

- Battle-tested frame extractor (100% success rate)
- Tuned X3D-M model with aggressive regularization
- Fixes for all major issues (mode collapse, overfitting, class imbalance)
- One-click setup from bare RunPod
- Automatic logging and monitoring"

# Force push to overwrite everything
git push --force origin main
```

### Alternative: Keep history but clean files
```bash
# Clone existing
git clone https://github.com/TyPoGamesTTV/runpod.git
cd runpod

# Remove all old files
git rm -rf .
git commit -m "Clean slate for X3D classifier"

# Copy our new files
cp -r ../runpod_setup/* .
git add .
git commit -m "Add production-ready X3D video classifier"

git push origin main
```

## What's in the new repo:

- `README.md` - Quick start guide
- `setup.sh` - One-click setup script
- `x3d_extractor.py` - Robust frame extraction (fixed)
- `train_x3d_tuned.py` - Training with all fixes
- `train_with_logging.sh` - Auto-logging wrapper
- `TROUBLESHOOTING.md` - All issues and solutions
- `requirements.txt` - Python dependencies
- `.gitignore` - Proper ignores

## Usage on new RunPod:

```bash
git clone https://github.com/TyPoGamesTTV/runpod.git
cd runpod
chmod +x setup.sh
./setup.sh
```