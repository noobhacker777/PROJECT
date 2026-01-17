# 🎉 Setup Instructions

## Quick Start (Choose Your Method)

### 🪟 **Windows Users**
Simply double-click:
```
start.bat
```
This will automatically setup and start the server.

### 🐧 **Linux/WSL Users**
Run in terminal:
```bash
bash start.sh
```
Or manually:
```bash
python setup.py install
python run_app.py
```

---

## 📋 What Gets Installed

The `setup.py` script installs:

**Core Packages:**
- Flask (web framework)
- PyTorch 2.5.1 (GPU-enabled with CUDA 12.1)
- OpenCV (image processing)
- NumPy, Pillow, SciPy
- PyYAML, Pydantic

**ML Packages:**
- HOLO11 (object detection)
- OpenCLIP (image embeddings)
- FAISS (vector similarity search)
- Additional utilities

**Total:** ~35 packages, ~3-5GB disk space

---

## ✅ What Happens During Setup

1. **Python Version Check** - Ensures Python 3.9+
2. **Dependency Installation** - Installs all packages
3. **Directory Creation** - Creates folders for data/models
4. **Import Verification** - Tests critical packages
5. **Model Check** - Verifies HOLO model exists

---

## 🚀 Start the Server

After setup completes, the server starts automatically at:
```
http://localhost:5002
```

**Test the API:**
```
http://localhost:5002/scan?image=IMG_1445.jpeg
```

---

## 🔍 Available Images

Pre-included test images in `dataset/images/`:
- IMG_1421.jpeg through IMG_1445.jpeg

Try scanning any of these:
```
http://localhost:5002/scan?image=IMG_1421.jpeg
http://localhost:5002/scan?image=IMG_1422.jpeg
```

---

## 📖 Full Documentation

- **01_QUICK_START.md** - Quick start guide
- **03_SCAN_API_GUIDE.md** - Complete API documentation
- **setup.py** - Setup script with all options

---

## ❓ Help

**Check environment without installing:**
```bash
python setup.py check
```

**Get setup help:**
```bash
python setup.py help
```

---

## 🔧 Advanced Setup Options

### Custom Python Environment

**Create virtual environment (optional):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

**Then run setup:**
```bash
python setup.py install
```

### Offline Installation

**Download packages:**
```bash
pip download -r requirements.txt -d ./packages
```

**Install from local:**
```bash
pip install --no-index --find-links ./packages -r requirements.txt
```

---

## 🐛 Troubleshooting

### Python version error
```bash
# Check version
python --version

# Should be 3.9 or higher
# If not, install Python 3.11 from python.org
```

### Permission denied errors
```bash
# Windows: Run Command Prompt as Administrator

# Linux/Mac: Use sudo
sudo python setup.py install
```

### PyTorch installation fails
```bash
# Manual GPU installation (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or CPU version
pip install torch torchvision torchaudio
```

### Import errors after install
```bash
# Verify all critical imports
python setup.py check

# Should show all imports as ✓
```

---

## 📊 System Verification

After setup, verify everything:

```bash
# Check critical packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import flask; print(f'Flask: {flask.__version__}')"
python -c "import faiss; print('FAISS: OK')"
```

All should print without errors.

---

## 🎯 Next Steps

1. Verify setup: `python setup.py check`
2. Start server: `python run_app.py`
3. Test API: `http://localhost:5002`
4. Read API guide: See [03_SCAN_API_GUIDE.md](03_SCAN_API_GUIDE.md)

---

**Ready?** Run: `python setup.py install`

✅ Then: `python run_app.py`

🚀 Your API is ready at `http://localhost:5002`
