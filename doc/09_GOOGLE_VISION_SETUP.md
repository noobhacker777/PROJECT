# Google Vision API Integration

## Overview

The OCR system now supports **Google Vision API** with automatic fallback to **EasyOCR**:

### Priority Order:
1. **Google Vision API** (if configured) → Superior accuracy, enterprise-grade
2. **EasyOCR** (fallback) → Local processing, no API required, handles rotated text

---

## Setup Instructions

### 1. Create Google Cloud Project

```bash
# 1. Go to Google Cloud Console
# https://console.cloud.google.com/

# 2. Create a new project (or select existing)
# 3. Enable Vision API for the project
#    - Search for "Cloud Vision API"
#    - Click "Enable"

# 4. Create a Service Account
#    - Go to "Service Accounts" page
#    - Click "Create Service Account"
#    - Name: "sku-detection" (or similar)
#    - Click "Create and Continue"

# 5. Grant permissions to service account
#    - Role: "Basic" > "Editor" (or "Viewer" for read-only)
#    - Click "Done"

# 6. Create and download JSON key
#    - In Service Account list, click on created account
#    - Go to "Keys" tab
#    - Click "Add Key" > "Create new key"
#    - Choose "JSON" format
#    - Download the JSON file
```

### 2. Install Google Vision Package

```bash
# Install the Google Cloud Vision library
pip install google-cloud-vision

# Or uncomment in requirements.txt:
# google-cloud-vision>=3.0.0
```

### 3. Configure Credentials

**Option A: Environment Variable (Recommended)**

```bash
# Windows PowerShell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account-key.json"

# Windows CMD
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account-key.json

# Linux/Mac
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"

# Then run the Flask app
python image_model_app.py
```

**Option B: Batch/Shell Script**

**Windows (run_with_google.bat)**
```batch
@echo off
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account-key.json
python image_model_app.py
```

**Linux/Mac (run_with_google.sh)**
```bash
#!/bin/bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
python image_model_app.py
```

### 4. Verify Setup

Start the Flask app and check console logs:

```
[INFO] Google Vision API not installed. Will use EasyOCR fallback.
```
↓ Google not configured yet

```
[OCR] 🔍 Attempting Google Vision API (priority)...
[GOOGLE-OCR] Calling Google Vision API for text detection...
[GOOGLE-OCR] ✓ Google Vision extracted 15 keywords
[GOOGLE-OCR] ✓ High confidence keywords (>30%): 12
```
↓ Google Vision is working!

---

## API Costs

### Google Vision API Pricing

- **Free Tier**: 1,000 Document Text Detection requests/month
- **Paid Tier**: $1.50 per 1,000 requests

**Cost Estimate:**
- 100 images/day = ~3,000 requests/month ≈ $4.50/month
- 1,000 images/day = ~30,000 requests/month ≈ $45/month

### When to Use Google OCR

✅ **Use Google Vision when:**
- High accuracy is critical
- Budget allows for API costs
- Images have complex layouts
- Text in multiple languages needed

✅ **Use EasyOCR when:**
- Running locally
- No API budget
- Privacy concerns
- Offline operation required

---

## OCR Response Format

Both Google and EasyOCR return identical JSON structure:

```json
{
  "success": true,
  "ocr_engine": "google",  // or "easyocr"
  "keywords": [
    {
      "text": "Coca-Cola",
      "confidence": 0.95
    },
    {
      "text": "500ml",
      "confidence": 0.92
    }
  ],
  "high_confidence_keywords": [  // confidence >= 0.30
    {
      "text": "Coca-Cola",
      "confidence": 0.95
    }
  ],
  "full_text": "Coca-Cola 500ml",
  "keyword_count": 2,
  "high_confidence_count": 2,
  "average_confidence": 0.935,
  "text_orientation": "horizontal",
  "rotation_applied": 0,
  "timestamp": "2026-02-27T10:30:45.123456"
}
```

---

## Troubleshooting

### Issue: "Google Vision API not installed"

```bash
# Solution: Install the library
pip install google-cloud-vision
```

### Issue: "GOOGLE_APPLICATION_CREDENTIALS environment variable not set"

```bash
# Solution: Set the environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"

# Verify it's set
echo $GOOGLE_APPLICATION_CREDENTIALS  # Linux/Mac
echo %GOOGLE_APPLICATION_CREDENTIALS% # Windows
```

### Issue: "401 Unauthorized" when calling API

✓ Check that JSON key file is correct
✓ Verify Cloud Vision API is enabled in Google Cloud Console
✓ Ensure service account has proper permissions
✓ Check that project ID in key matches project in Console

### Issue: "Quota exceeded"

- Free tier limited to 1,000 requests/month
- Upgrade to paid tier or wait until next month
- Check billing settings in Google Cloud Console

### Issue: Google fails, trying EasyOCR

```
[OCR] Google Vision failed: [error message]
[OCR] Falling back to EasyOCR...
```

This is expected behavior. The system automatically switches to EasyOCR. Check error message for details.

---

## Code Changes

The `extract_ocr_keywords()` function now:

1. **Checks for Google credentials** at startup
2. **Tries Google Vision API first** if configured
3. **Automatically falls back to EasyOCR** if:
   - Google API not installed
   - Credentials not configured
   - API call fails
4. **Returns identical response format** regardless of engine
5. **Adds `ocr_engine` field** to response ("google" or "easyocr")

### Usage Example

```python
# No code changes needed! Same API:
result = extract_ocr_keywords(image)

# Response includes:
if result['ocr_engine'] == 'google':
    print("Using Google Vision API")
else:
    print("Using EasyOCR")
```

---

## Performance Comparison

| Feature | Google Vision | EasyOCR |
|---------|---------------|---------|
| Accuracy | ★★★★★ | ★★★★ |
| Speed | 0.5-2s (API) | 1-5s (GPU) |
| Setup | Requires API key | Install package |
| Cost | $1.50/1K requests | Free |
| Offline | No | Yes |
| Languages | 100+ | Limited |
| Rotation | Automatic | Manual trials |

---

## Reference

- [Google Cloud Vision API Docs](https://cloud.google.com/vision/docs)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [Service Account Setup](https://cloud.google.com/docs/authentication/getting-started)
