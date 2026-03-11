# API Scan Image (JSON-Only)

This document describes the API-only equivalent of the GUI "Detect Products"
workflow (the `scanAndDetect()` function in the dashboard).

## Endpoint Summary

- GUI page: `GET /scan` (serves the dashboard)
- GUI detect pipeline: `POST /api/detect` (includes preview URLs)
- API-only JSON: `POST /api/scanimg` (same backend, no preview files)
- JSON-only alias: `POST /detect` (same as `/api/scanimg`)
- Legacy GET scan: `GET /api/scan?image=...` (older flow)

## API-Only Endpoint

### POST /api/scanimg

Runs the same detection + FAISS + OCR backend as the GUI, but returns JSON only
and skips preview file outputs (`image_url`, `crops_url`).

Request:
- Content-Type: `multipart/form-data`
- Field name: `image` (file)
- Optional query param: `accuracy_mode=true`

Response (JSON):
```
{
  "success": true,
  "product_count": 2,
  "detections": [
    {
      "id": 0,
      "confidence": 0.95,
      "box": [x, y, w, h],
      "matched_sku": "SKU_NAME",
      "sku_similarity": 0.87,
      "ocr_data": {...}
    }
  ],
  "image_size": [width, height],
  "matching_mode": "cached_index",
  "accuracy_mode_enabled": false,
  "ocr_keywords": {...},
  "ocr_sku_match": {...},
  "message": "Detected N product(s)"
}
```

Notes:
- `image_url` and `crops_url` are omitted in `/api/scanimg`.
- Use `/api/detect` if you want preview URLs for the GUI.

## Examples

### cURL
```
curl -X POST http://127.0.0.1:5002/api/scanimg \
  -F "image=@D:\\path\\to\\image.jpg"
```

### cURL (Multiple Images)
Send multiple files by repeating the `image` field:
```
curl -X POST http://127.0.0.1:5002/api/scanimg \
  -F "image=@D:\\path\\to\\image_1.jpg" \
  -F "image=@D:\\path\\to\\image_2.jpg"
```

Response:
```
{
  "success": true,
  "count": 2,
  "results": [
    {"filename": "image_1.jpg", "status": 200, "ok": true, "response": {...}},
    {"filename": "image_2.jpg", "status": 200, "ok": true, "response": {...}}
  ]
}
```

### JavaScript (Fetch)
```
const formData = new FormData();
formData.append('image', fileInput.files[0]);

const response = await fetch('/api/scanimg', {
  method: 'POST',
  body: formData
});
const data = await response.json();
console.log(data);
```

### Python (requests)
```
import requests

with open(r"D:\\path\\to\\image.jpg", "rb") as f:
    r = requests.post("http://127.0.0.1:5002/api/scanimg", files={"image": f})
print(r.json())
```

## CLI Test Script

You can also use the built-in test script:
```
python test.py --mode detect --endpoint /api/scanimg D:\\path\\to\\image.jpg
```
