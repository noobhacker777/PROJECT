# 🚀 HOLO SKU Detection API - Quick Start Guide

**HOLO11x-based product detection and SKU matching system**

---

## 📋 Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- **pip** (Python package manager)
- **Git** (optional, for cloning)

**For GPU Support:**
- NVIDIA GPU (CUDA compatible)
- CUDA 12.1 or compatible version

---

## ⚡ Quick Start (3 Steps)

### Step 1: Setup Environment
```bash
# Run setup script to install all dependencies
python setup.py install
```

This will:
- ✓ Check Python version
- ✓ Install all packages from requirements.txt
- ✓ Create necessary directories
- ✓ Verify critical imports

### Step 2: Start the API Server
```bash
python run_app.py
```

You should see:
```
================================================================================
HOLO SKU Detection API Server
================================================================================
Starting Flask server on http://localhost:5002
...
```

### Step 3: Test the API
Open your browser or use curl:

**HTML Results (Browser):**
```
http://localhost:5002/scan?image=IMG_1445.jpeg
```

**JSON Results (API):**
```
http://localhost:5002/scan?image=IMG_1445.jpeg&format=json
```

---

## 📚 Full API Documentation

See [03_SCAN_API_GUIDE.md](03_SCAN_API_GUIDE.md) for:
- Detailed endpoint documentation
- API response formats
- Usage examples (JavaScript, Python, cURL)
- Error handling

---

## 🔧 Setup Script Options

```bash
# Full setup (recommended)
python setup.py install

# Just check if environment is ready
python setup.py check

# Show help
python setup.py help
```

---

## 📁 Project Structure

```
Project/
├── setup.py                 # Environment setup script
├── run_app.py              # Start the Flask server
├── requirements.txt        # Python dependencies
├── doc/                    # Documentation (this folder)
├── image_model_app.py     # Flask application
├── dataset/
│   ├── images/            # Input images
│   └── labels/            # HOLO labels
├── scan_models/
│   └── scan_model.pt      # HOLO model
├── tmp/                   # Temporary files
└── static/                # Web UI files
```

---

## 🌐 Web Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Dashboard |
| `/scan?image=...` | GET | Detect objects & match SKUs |
| `/api/detect` | POST | Upload image for detection |
| `/uploads/image/...` | GET | Serve uploaded images |
| `/tmp/...` | GET | Serve temporary files |

---

## 🆘 Troubleshooting

### Setup fails with "permission denied"
```bash
# Run with sudo (Linux/Mac)
sudo python setup.py install

# Or use --user flag
python setup.py install --user
```

### "No module named torch"
```bash
# Reinstall PyTorch manually
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### API returns 400 error
```
✓ Use simple filename:    /scan?image=IMG_1445.jpeg
✗ Don't use full paths:   /scan?image=D:/path/to/image.jpeg
✗ Don't use quotes:       /scan?image="IMG_1445.jpeg"
```

### Port 5002 already in use
```bash
# The server will try the next available port
# Or change port in run_app.py
```

---

## 📊 System Requirements

**Minimum:**
- Python 3.9+
- 4GB RAM
- 2GB disk space

**Recommended (with GPU):**
- Python 3.10 or 3.11
- 8GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- CUDA 12.1

---

## 🎯 Next Steps

1. **Complete Setup:**
   ```bash
   python setup.py install
   ```

2. **Start Server:**
   ```bash
   python run_app.py
   ```

3. **Test API:**
   ```bash
   http://localhost:5002/scan?image=IMG_1445.jpeg
   ```

4. **Read Documentation:**
   - See [03_SCAN_API_GUIDE.md](03_SCAN_API_GUIDE.md)
   - For detailed API examples

---

## 📝 License

HOLO SKU Detection API - 2026

---

## 💬 Support

For issues or questions:
1. Check [03_SCAN_API_GUIDE.md](03_SCAN_API_GUIDE.md)
2. Review error messages in server logs
3. Ensure all dependencies installed: `python setup.py check`

---

**Ready to get started?**
```bash
python setup.py install && python run_app.py
```

🚀 Your API will be ready at `http://localhost:5002`
