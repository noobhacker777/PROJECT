# 🔍 Image Scan & Detection API Guide

**Base URL:** `http://localhost:5002`

---

## 🚀 SIMPLEST WAY - Just Paste URL in Browser

**No GET/POST selection needed - it just works!**

### 📝 Usage Format
Pass image as **filename** or **server URL**, not full file paths:

```
/scan?image=FILENAME                    (searches dataset/images, uploads, tmp)
/scan?image=/uploads/image/FILENAME    (from uploads folder)
/scan?image=/tmp/FILENAME              (from tmp folder)
/scan?image=http://...                 (remote URLs)
```

### ✅ SIMPLE Examples (These work!)

**Just use the filename:**
```
http://localhost:5002/scan?image=IMG_1445.jpeg
```
→ Automatically finds it in dataset/images, uploads, or tmp

**Or use server paths:**
```
http://localhost:5002/scan?image=/uploads/image/IMG_1445.jpeg
http://localhost:5002/scan?image=/tmp/scan_file.jpg
http://localhost:5002/scan?image=http://example.com/product.jpg
```

### ❌ Don't Do This (Will fail)
```
# ❌ Full file paths - WRONG!
http://localhost:5002/scan?image=D:/Client/Python/HOLO_unified_C/dataset/images/IMG_1445.jpeg
http://localhost:5002/scan?image=/mnt/d/Client/Python/HOLO_unified_C/dataset/images/IMG_1445.jpeg

# ❌ Extra quotes - WRONG!
http://localhost:5002/scan?image="IMG_1445.jpeg"
```

---

## ✅ How It Works - User Perspective

**Step 1:** Just use the image filename or URL
```
http://localhost:5002/scan?image=IMG_1445.jpeg
```

**Step 2:** Paste into browser address bar (no full paths!)

**Step 3:** Press Enter

**Result:** Backend automatically finds the image and processes it

**That's it! No file paths, no coding, no buttons needed.**

---

## 📊 HTML Response (Browser View)

When you visit the URL, you see:

```
✅ Detection Complete - Found 3 Product(s)

[Original Image]                    [Detected Image with Boxes]

📦 Detected Products (3 total)

Product #1
🎯 SKU: BIRDS
📊 Confidence: 96.2%
✔️ Match Score: 89.5%
📍 Box: [120, 85, 200, 250]

Product #2
🎯 SKU: CABLE
📊 Confidence: 92.1%
✔️ Match Score: 81.3%
📍 Box: [350, 100, 180, 220]

Product #3
🎯 SKU: SCREWDRIVER
📊 Confidence: 87.4%
✔️ Match Score: 78.6%
📍 Box: [500, 150, 220, 200]

🎯 All SKU Matches (Confidence Scores)

BIRDS: 89.5% (🟢 strong)
CABLE: 81.3% (🟡 good)
SCREWDRIVER: 78.6% (🟡 good)
OTHER: 42.1% (🔴 weak)

📋 Summary
Total Products Detected: 3
Image Size: 800 x 600 pixels
Message: Detection completed successfully
```

---

## 📋 JSON Response Format (API View)

Add `&format=json` to get JSON response:

```
http://localhost:5002/scan?image=/path/to/image.jpg&format=json
```

**Response:**
```json
{
  "success": true,
  "product_count": 3,
  "detections": [
    {
      "id": 0,
      "confidence": 0.962,
      "box": [120, 85, 200, 250],
      "matched_sku": "BIRDS",
      "sku_similarity": 0.895
    },
    {
      "id": 1,
      "confidence": 0.921,
      "box": [350, 100, 180, 220],
      "matched_sku": "CABLE",
      "sku_similarity": 0.813
    },
    {
      "id": 2,
      "confidence": 0.874,
      "box": [500, 150, 220, 200],
      "matched_sku": "SCREWDRIVER",
      "sku_similarity": 0.786
    }
  ],
  "image_size": [800, 600],
  "image_url": "/tmp/scan_20231230_145030_a1b2c3d4.jpg",
  "crops_url": "/tmp/scan_20231230_145030_a1b2c3d4_detected_crops.jpg",
  "sku_matches": {
    "BIRDS": 0.895,
    "CABLE": 0.813,
    "SCREWDRIVER": 0.786,
    "OTHER": 0.421
  },
  "message": "Detection completed successfully"
}
```

---

## 🧪 API-Only POST (Same as GUI Detect)

Use this when you want the **same backend pipeline as the GUI** but **JSON only**:

```
POST /api/scanimg
```

### cURL
```bash
curl -X POST http://127.0.0.1:5002/api/scanimg \
  -F "image=@D:\\path\\to\\image.jpg"
```

### cURL (Multiple Images)
```bash
curl -X POST http://127.0.0.1:5002/api/scanimg \
  -F "image=@D:\\path\\to\\image_1.jpg" \
  -F "image=@D:\\path\\to\\image_2.jpg"
```

### JavaScript (Fetch)
```javascript
const formData = new FormData();
formData.append('image', file);

const response = await fetch('/api/scanimg', {
  method: 'POST',
  body: formData
});
const data = await response.json();
```

---

## 📝 Complete Usage Examples

### Browser - View Results as HTML
```
http://localhost:5002/scan?image=IMG_1445.jpeg
```
→ Opens page with images and detection boxes

### API - Get JSON Response
```
http://localhost:5002/scan?image=IMG_1445.jpeg&format=json
```
→ Returns JSON with all detection data

### cURL - Command Line
```bash
curl "http://localhost:5002/scan?image=IMG_1445.jpeg&format=json"
```

### JavaScript - Fetch API
```javascript
const scanUrl = 'http://localhost:5002/scan?image=IMG_1445.jpeg&format=json';

fetch(scanUrl)
  .then(r => r.json())
  .then(data => {
    console.log(`Found ${data.product_count} products`);
    console.log(data.detections);
  })
```

### Python - Requests
```python
import requests

response = requests.get('http://localhost:5002/scan', params={
    'image': 'IMG_1445.jpeg',
    'format': 'json'
})
result = response.json()
print(f"Found {result['product_count']} products")
print(result['detections'])
```

---

## 🎯 Response Fields Explained

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | True if detection worked |
| `product_count` | number | How many products detected |
| `detections` | array | List of detected items |
| `detections[].id` | number | Detection ID (0, 1, 2...) |
| `detections[].confidence` | number | 0-1 HOLO confidence score |
| `detections[].box` | array | `[x, y, width, height]` |
| `detections[].matched_sku` | string | Best matching SKU |
| `detections[].sku_similarity` | number | 0-1 similarity to SKU |
| `image_size` | array | `[width, height]` |
| `image_url` | string | URL to uploaded image |
| `crops_url` | string | URL to detection visualization |
| `sku_matches` | object | All SKU matches with scores |
| `message` | string | Status message |

---

## 🔗 Image Format Reference

### Supported Input Formats

**Filename (searches default folders):**
```
IMG_1445.jpeg
photo.jpg
```

**Server paths:**
```
/uploads/image/IMG_1445.jpeg
/tmp/scan_20231230_145030.jpg
```

**Remote URLs:**
```
http://example.com/product.jpg
https://storage.googleapis.com/bucket/image.jpg
```

---

## ❌ Error Responses

### File Not Found
```
http://localhost:5002/scan?image=nonexistent.jpg
```
Response: "❌ File Not Found: /nonexistent/path.jpg"

### Invalid Image Format
```
http://localhost:5002/scan?image=/path/to/text.txt
```
Response: "❌ Could not decode image"

### No Image Provided
```
http://localhost:5002/scan
```
Shows: Interactive form to enter image path

---

## 💡 Quick Tips

✅ Use either backslashes or forward slashes in paths
✅ Works with local files and remote URLs
✅ Automatically creates images in `/tmp/` folder
✅ Detects up to hundreds of products per image
✅ Matches to SKU embeddings in seconds
✅ Returns both HTML (browser) and JSON (API) formats

**Remember:** Just enter the URL - the backend does the rest! 🚀
