"""
HOLO11 SKU Detection API - Flask App
Licensed under AGPL-3.0. See LICENSE file for details.

Portions derived from Ultralytics YOLOv11 (AGPL-3.0).

Flask app that serves inference and training endpoints.
Run: python image_model_app.py
"""
from flask import Flask, request, jsonify, send_from_directory, send_file, redirect
from pathlib import Path
import threading
import time
import json
import sys
import subprocess
import shutil
from datetime import datetime
from functools import wraps
import os
import importlib.util
import cv2
import numpy as np
import threading



PROJECT = Path(__file__).resolve().parent
APP = Flask(__name__, static_folder=str(PROJECT / 'static'), static_url_path='')

# Global cache for SKU embeddings (loaded once on startup)
SKU_EMBEDDINGS_CACHE = {}
SKU_EMBEDDINGS_INITIALIZED = False
MATCHER_INSTANCE = None
_EMBEDDINGS_LOCK = threading.Lock()  # Thread-safe lock for SKU embeddings initialization

def initialize_sku_embeddings():
    """Initialize SKU embeddings cache on first run (thread-safe)."""
    global SKU_EMBEDDINGS_CACHE, SKU_EMBEDDINGS_INITIALIZED, MATCHER_INSTANCE
    
    with _EMBEDDINGS_LOCK:
        # Double-check after acquiring lock
        if SKU_EMBEDDINGS_INITIALIZED:
            return
        
        try:
            print("[INIT] Loading SKU embeddings cache...")
            from sku_embeddings import SKUEmbeddingMatcher
            MATCHER_INSTANCE = SKUEmbeddingMatcher()
            
            # Build embeddings dict from SKU folders
            openclip_dir = PROJECT / 'openclip_dataset'
            if openclip_dir.exists():
                for sku_folder in openclip_dir.iterdir():
                    if sku_folder.is_dir():
                        sku_name = sku_folder.name
                        images = list(sku_folder.glob('*.jpg')) + list(sku_folder.glob('*.jpeg')) + list(sku_folder.glob('*.png')) + list(sku_folder.glob('*.JPG')) + list(sku_folder.glob('*.JPEG')) + list(sku_folder.glob('*.PNG'))
                        
                        if images:
                            print(f"[INIT] Processing SKU: {sku_name} ({len(images)} images)")
                            embeddings_list = []
                            
                            for img_path in images:
                                try:
                                    img_embedding = MATCHER_INSTANCE.get_image_embedding(str(img_path))
                                    if img_embedding is not None:
                                        embeddings_list.append(img_embedding)
                                except Exception as e:
                                    print(f"[INIT] Error with {img_path}: {e}")
                            
                            if embeddings_list:
                                SKU_EMBEDDINGS_CACHE[sku_name] = np.mean(np.array(embeddings_list), axis=0)
                                print(f"[INIT] ✓ {sku_name} ready")
            
            SKU_EMBEDDINGS_INITIALIZED = True
            print("[INIT] SKU embeddings loaded!")
        except Exception as e:
            print(f"[INIT] Error loading embeddings: {e}")
            import traceback
            traceback.print_exc()

def cleanup_old_tmp_files(max_age_hours=1):
    tmp_dir = PROJECT / 'tmp'
    if tmp_dir.exists():
        current_time = time.time()
        for file in tmp_dir.glob('scan_*.jpg'):
            try:
                age = (current_time - file.stat().st_mtime) / 3600
                if age > max_age_hours:
                    file.unlink()
            except:
                pass

cleanup_old_tmp_files()



# Serve dashboard at root to bypass index.html
@APP.route('/')
def root_dashboard():
    return send_from_directory(str(PROJECT / 'static'), 'dashboard.html')


@APP.route('/api-demo')
def api_demo():
    """Serve API documentation and testing page."""
    return send_from_directory(str(PROJECT / 'static'), 'api_demo.html')


# Use only internal dataset folder inside this project
DATASET = PROJECT / 'dataset'
UPLOAD_IMAGES = DATASET / 'images'
UPLOAD_LABELS = DATASET / 'labels'


@APP.route('/uploads/image/<path:name>')
def serve_upload_image(name):
    return send_from_directory(str(UPLOAD_IMAGES), name)


def safe_json(f):
    @wraps(f)
    def _wrap(*a, **k):
        try:
            return f(*a, **k)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return _wrap




@APP.route('/backups', methods=['GET', 'DELETE', 'POST'])
def backups():
    """List, restore, download, or delete backups."""
    scan_backup_dir = PROJECT / 'scan' / 'backups'
    scan_model = PROJECT / 'scan' / 'scan.pt'
    
    if request.method == 'GET':
        # List all scan.pt backups
        backups_list = []
        if scan_backup_dir.exists():
            for f in sorted(scan_backup_dir.glob('*.pt'), reverse=True):
                try:
                    size = f.stat().st_size / (1024 * 1024)  # Size in MB
                    mtime = f.stat().st_mtime
                    from datetime import datetime as dt
                    mod_time = dt.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    backups_list.append({
                        'name': f.name,
                        'size_mb': round(size, 2),
                        'modified': mod_time,
                        'path': str(f)
                    })
                except Exception:
                    pass
        
        # Get current scan model info
        current_model = None
        if scan_model.exists():
            try:
                size = scan_model.stat().st_size / (1024 * 1024)
                mtime = scan_model.stat().st_mtime
                from datetime import datetime as dt
                mod_time = dt.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                current_model = {
                    'name': 'scan.pt (Current)',
                    'size_mb': round(size, 2),
                    'modified': mod_time
                }
            except Exception:
                pass
        
        return jsonify({
            'backups': backups_list,
            'current_model': current_model,
            'total': len(backups_list),
            'backup_dir': str(scan_backup_dir)
        })
    
    elif request.method == 'POST':
        # Restore a backup
        backup_name = request.json.get('backup_name') if request.is_json else request.form.get('backup_name')
        if not backup_name:
            return jsonify({'error': 'no backup_name specified'}), 400
        
        # Security: prevent directory traversal
        if '..' in backup_name or '/' in backup_name or '\\' in backup_name:
            return jsonify({'error': 'invalid backup_name'}), 400
        
        backup_file = scan_backup_dir / backup_name
        if not backup_file.exists():
            return jsonify({'error': 'backup file not found'}), 404
        
        try:
            import shutil
            # Backup current scan.pt before restoring
            if scan_model.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pre_restore = scan_backup_dir / f"pre_restore_{timestamp}.pt"
                shutil.copy2(str(scan_model), str(pre_restore))
            
            # Restore backup
            shutil.copy2(str(backup_file), str(scan_model))
            
            # Reload model
            global model, model_mtime
            model = None
            model_mtime = None
            load_model()
            
            return jsonify({
                'success': True,
                'restored': backup_name,
                'message': f'Restored {backup_name} successfully'
            })
        except Exception as e:
            return jsonify({'error': f'Could not restore backup: {str(e)}'}), 500
    
    elif request.method == 'DELETE':
        # Delete specific backup
        filename = request.args.get('file')
        if not filename:
            return jsonify({'error': 'no file specified'}), 400
        
        # Security: prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'error': 'invalid filename'}), 400
        
        backup_file = scan_backup_dir / filename
        
        if not backup_file.exists():
            return jsonify({'error': 'backup file not found'}), 404
        
        try:
            backup_file.unlink()
            return jsonify({
                'deleted': filename,
                'message': f'Backup {filename} deleted successfully'
            })
        except Exception as e:
            return jsonify({'error': f'Could not delete backup: {str(e)}'}), 500





@APP.route('/backups/download/<backup_name>')
def download_backup(backup_name):
    """Download a backup file to local machine."""
    scan_backup_dir = PROJECT / 'scan' / 'backups'
    
    # Security: prevent directory traversal
    if '..' in backup_name or '/' in backup_name or '\\' in backup_name:
        return jsonify({'error': 'invalid backup_name'}), 400
    
    backup_file = scan_backup_dir / backup_name
    if not backup_file.exists():
        return jsonify({'error': 'backup file not found'}), 404
    
    try:
        from flask import send_file
        return send_file(
            str(backup_file),
            as_attachment=True,
            download_name=backup_name,
            mimetype='application/octet-stream'
        )
    except Exception as e:
        return jsonify({'error': f'Could not download backup: {str(e)}'}), 500


@APP.after_request
def add_header_no_cache(response):
    # Prevent caching of static assets so edits appear immediately in browser
    try:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    except Exception:
        pass
    return response

@APP.route('/images')
def images_list():
    """Return JSON list of image filenames from dataset/images directory."""
    ds = PROJECT / 'dataset'
    images_dir = ds / 'images'
    result = {'images': []}
    
    if images_dir.exists():
        for f in images_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                result['images'].append(f.name)
    
    # sort alphabetically
    result['images'].sort()
    return jsonify(result)


@APP.route('/tmp/<path:filename>')
def serve_tmp_file(filename):
    """Serve temporary scan images from tmp/ folder."""
    # Security: ensure no path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'error': 'invalid filename'}), 400
    
    tmp_dir = PROJECT / 'tmp'
    file_path = tmp_dir / filename
    
    if not file_path.exists():
        return jsonify({'error': 'file not found'}), 404
    
    if not str(file_path).startswith(str(tmp_dir)):
        return jsonify({'error': 'access denied'}), 403
    
    return send_file(str(file_path), mimetype='image/jpeg')


@APP.route('/api/tmp/cleanup', methods=['POST'])
def cleanup_tmp_file():
    """Delete a temporary scan image file."""
    filename = request.json.get('filename') if request.is_json else request.form.get('filename')
    
    if not filename:
        return jsonify({'success': False, 'error': 'No filename specified'}), 400
    
    # Security: ensure no path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        return jsonify({'success': False, 'error': 'invalid filename'}), 400
    
    tmp_dir = PROJECT / 'tmp'
    file_path = tmp_dir / filename
    
    # Verify file is in tmp directory
    if not str(file_path).startswith(str(tmp_dir)):
        return jsonify({'success': False, 'error': 'access denied'}), 403
    
    try:
        if file_path.exists():
            file_path.unlink()  # Delete the file
            return jsonify({'success': True, 'message': f'Deleted {filename}'}), 200
        else:
            return jsonify({'success': False, 'error': 'file not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': f'Could not delete file: {str(e)}'}), 500


@APP.route('/upload_image', methods=['POST'])
@safe_json
def upload_image():
    """Upload an image into dataset/images. Returns saved filename."""
    ds = PROJECT / 'dataset'
    images_dir = ds / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    img = request.files.get('image')
    if not img:
        return jsonify({'error': 'no image uploaded'}), 400
    # sanitize filename
    fname = Path(img.filename).name
    dest = images_dir / fname
    # if exists, append a counter
    if dest.exists():
        stem = dest.stem
        ext = dest.suffix
        i = 1
        while (images_dir / f"{stem}_{i}{ext}").exists():
            i += 1
        dest = images_dir / f"{stem}_{i}{ext}"
    img.save(dest)
    return jsonify({'saved': str(dest), 'name': dest.name})

@APP.route('/api/save_crop', methods=['POST'])
def save_crop():
    """Save cropped product image for OpenCLIP dataset.
    
    Expected form data:
    - image: file (JPEG crop)
    - sku_id: string (e.g., "SKU_0001")
    - image_name: string (original image name for reference)
    - box_index: optional int (box index for base filename)
    
    Saves to: openclip_dataset/{sku_id}/{image_stem}_box{box_index}.jpg
    If file exists: openclip_dataset/{sku_id}/{image_stem}_box{box_index}_{random}.jpg
    """
    try:
        from PIL import Image
        from io import BytesIO
        import random
        import string
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file', 'ok': False}), 400
        
        sku_id = request.form.get('sku_id', 'unknown').strip()
        if not sku_id:
            return jsonify({'error': 'SKU ID required', 'ok': False}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'ok': False}), 400
        
        # Create directory structure
        crop_dir = PROJECT / 'openclip_dataset' / sku_id
        crop_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename using image_stem and box_index
        image_stem = request.form.get('image_stem', '').strip()
        box_index = request.form.get('box_index', '0').strip()
        
        # Fallback if image_stem is missing
        if not image_stem:
            image_stem = Path(file.filename).stem if file.filename else 'crop'
        
        # Format: {image_stem}_box{box_index}.jpg
        # Example: IMG_1421_box0.jpg, IMG_1422_box1.jpg, etc.
        base_filename = f"{image_stem}_box{box_index}.jpg"
        filepath = crop_dir / base_filename
        
        # If file already exists, add random suffix
        if filepath.exists():
            # Generate random 6-char suffix
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            filename = f"{image_stem}_box{box_index}_{random_suffix}.jpg"
            filepath = crop_dir / filename
            print(f"📌 File exists: {base_filename} → Using: {filename}")
        else:
            filename = base_filename
        
        # Load image, validate, and save
        img = Image.open(BytesIO(file.read()))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save with quality
        img.save(filepath, quality=95, optimize=True)
        
        # Count images in SKU folder
        image_count = len(list(crop_dir.glob('*.jpg')))
        
        return jsonify({
            'ok': True,
            'sku_id': sku_id,
            'filename': filename,
            'path': str(filepath.relative_to(PROJECT)),
            'total_images': image_count,
            'message': f'Saved crop for {sku_id}: {filename}'
        })
    
    except Exception as e:
        return jsonify({'error': f'Save failed: {str(e)}', 'ok': False}), 500


@APP.route('/api/save_sku_list', methods=['POST'])
def save_sku_list():
    """Save SKU list to dataset/SKU.json"""
    try:
        data = request.get_json()
        sku_list = data.get('skuList', [])
        
        # Save to dataset/SKU.json
        sku_file = PROJECT / 'dataset' / 'SKU.json'
        sku_file.write_text(json.dumps({
            'skus': sku_list,
            'timestamp': datetime.now().isoformat(),
            'total_skus': len(sku_list)
        }, indent=2))
        
        return jsonify({
            'ok': True,
            'message': f'Saved {len(sku_list)} SKUs to dataset/SKU.json',
            'count': len(sku_list)
        })
    except Exception as e:
        return jsonify({'error': f'Save failed: {str(e)}', 'ok': False}), 500


@APP.route('/api/load_sku_list', methods=['GET'])
def load_sku_list():
    """Load SKU list from dataset/SKU.json"""
    try:
        sku_file = PROJECT / 'dataset' / 'SKU.json'
        
        if sku_file.exists():
            data = json.loads(sku_file.read_text())
            return jsonify({
                'ok': True,
                'skus': data.get('skus', []),
                'count': len(data.get('skus', []))
            })
        else:
            return jsonify({
                'ok': True,
                'skus': [],
                'count': 0
            })
    except Exception as e:
        return jsonify({'error': f'Load failed: {str(e)}', 'ok': False, 'skus': []}), 500


@APP.route('/api/rename_sku', methods=['POST'])
def rename_sku():
    """Rename SKU ID and move crop files from old to new folder"""
    try:
        data = request.get_json()
        old_sku = data.get('oldSKU', '').strip()
        new_sku = data.get('newSKU', '').strip()
        
        if not old_sku or not new_sku:
            return jsonify({'error': 'Missing oldSKU or newSKU', 'ok': False}), 400
        
        old_dir = PROJECT / 'openclip_dataset' / old_sku
        new_dir = PROJECT / 'openclip_dataset' / new_sku
        
        moved_count = 0
        
        # If old folder exists, move/rename it
        if old_dir.exists():
            # Create new directory if needed
            new_dir.mkdir(parents=True, exist_ok=True)
            
            # Move all files from old to new folder
            for file_path in old_dir.iterdir():
                if file_path.is_file():
                    dest_path = new_dir / file_path.name
                    # Copy file to new location
                    dest_path.write_bytes(file_path.read_bytes())
                    # Remove old file
                    file_path.unlink()
                    moved_count += 1
            
            # Remove old directory if empty
            if not any(old_dir.iterdir()):
                old_dir.rmdir()
        
        return jsonify({
            'ok': True,
            'message': f'Moved {moved_count} files from {old_sku} to {new_sku}',
            'movedCount': moved_count
        })
    except Exception as e:
        return jsonify({'error': f'Rename failed: {str(e)}', 'ok': False}), 500


@APP.route('/api/delete_crop', methods=['POST'])
def delete_crop():
    """Delete crop file from SKU folder when box is deleted"""
    try:
        data = request.get_json()
        sku_id = data.get('skuId', '').strip()
        crop_filename = data.get('cropFilename', '').strip()
        
        if not sku_id or not crop_filename:
            return jsonify({'error': 'Missing skuId or cropFilename', 'ok': False}), 400
        
        crop_path = PROJECT / 'openclip_dataset' / sku_id / crop_filename
        
        if crop_path.exists():
            file_size = crop_path.stat().st_size
            crop_path.unlink()
            print(f"🗑️  Deleted crop file: openclip_dataset/{sku_id}/{crop_filename} ({file_size} bytes)")
            
            return jsonify({
                'ok': True,
                'message': f'Deleted {crop_filename} from {sku_id}',
                'deleted': True,
                'file_size': file_size,
                'path': f'openclip_dataset/{sku_id}/{crop_filename}'
            })
        else:
            print(f"ℹ️  Crop file not found for deletion: openclip_dataset/{sku_id}/{crop_filename}")
            return jsonify({
                'ok': True,
                'message': f'File not found: {crop_filename}',
                'deleted': False,
                'path': f'openclip_dataset/{sku_id}/{crop_filename}'
            })
    except PermissionError as e:
        error_msg = f'Permission denied deleting crop file: {str(e)}'
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg, 'ok': False}), 403
    except Exception as e:
        error_msg = f'Delete failed: {str(e)}'
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg, 'ok': False}), 500


@APP.route('/api/move_crop', methods=['POST'])
def move_crop():
    """Move crop file from old SKU folder to new SKU folder in real-time
    
    When a box's SKU is changed, this moves the crop image from:
    openclip_dataset/{oldSKU}/{cropFilename} → openclip_dataset/{newSKU}/{cropFilename}
    """
    import os
    import shutil
    import time
    
    try:
        data = request.get_json()
        crop_filename = data.get('cropFilename', '').strip()
        old_sku = data.get('oldSKU', '').strip()
        new_sku = data.get('newSKU', '').strip()
        
        if not crop_filename or not old_sku or not new_sku:
            return jsonify({'error': 'Missing cropFilename, oldSKU, or newSKU', 'ok': False}), 400
        
        old_path = PROJECT / 'openclip_dataset' / old_sku / crop_filename
        new_dir = PROJECT / 'openclip_dataset' / new_sku
        new_path = new_dir / crop_filename
        
        # If old file doesn't exist, no need to move
        if not old_path.exists():
            print(f"ℹ️  Crop file not found at: {old_path}")
            print(f"    No crop image to move yet (will be created on next Ctrl+S)")
            return jsonify({
                'ok': True,
                'message': f'Crop file not found - no move needed',
                'moved': False,
                'from': f'openclip_dataset/{old_sku}/{crop_filename}',
                'to': f'openclip_dataset/{new_sku}/{crop_filename}',
                'note': 'File does not exist yet - will be saved to new SKU folder on next Ctrl+S'
            })
        
        # Create new directory if needed
        new_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created/verified directory: {new_dir}")
        
        # Copy file to new location using shutil (more robust)
        file_size = old_path.stat().st_size
        shutil.copy2(str(old_path), str(new_path))
        print(f"✅ Copied crop image: {crop_filename} ({file_size} bytes)")
        
        # Verify copy was successful
        if not new_path.exists():
            raise Exception(f"Failed to verify copied file at {new_path}")
        
        # Give system a moment for file locks to release
        time.sleep(0.1)
        
        # Delete old file - try multiple methods if first fails
        deleted = False
        delete_error = None
        
        # Method 1: pathlib unlink
        try:
            old_path.unlink()
            deleted = True
            print(f"🗑️  Removed old crop file using unlink: {old_path}")
        except Exception as e1:
            delete_error = str(e1)
            print(f"⚠️  unlink() failed: {e1}")
            
            # Method 2: os.remove
            try:
                os.remove(str(old_path))
                deleted = True
                print(f"🗑️  Removed old crop file using os.remove: {old_path}")
            except Exception as e2:
                print(f"⚠️  os.remove() failed: {e2}")
                delete_error = f"unlink: {e1} | os.remove: {e2}"
                
                # Method 3: Try with longer delay
                try:
                    time.sleep(0.2)
                    os.remove(str(old_path))
                    deleted = True
                    print(f"🗑️  Removed old crop file using os.remove (retry): {old_path}")
                except Exception as e3:
                    print(f"❌ All deletion methods failed: {e3}")
        
        # Final verification
        if new_path.exists():
            print(f"✅ Crop image move completed:")
            print(f"   FROM: openclip_dataset/{old_sku}/{crop_filename}")
            print(f"   TO:   openclip_dataset/{new_sku}/{crop_filename}")
            if deleted:
                print(f"   OLD FILE: DELETED ✓")
            else:
                print(f"   OLD FILE: STILL EXISTS (but copy successful) ⚠️")
            
            return jsonify({
                'ok': True,
                'message': f'Successfully moved {crop_filename} from {old_sku} to {new_sku}',
                'moved': True,
                'deleted': deleted,
                'from': f'openclip_dataset/{old_sku}/{crop_filename}',
                'to': f'openclip_dataset/{new_sku}/{crop_filename}',
                'file_size': file_size,
                'warning': None if deleted else 'Old file could not be deleted (may be locked)'
            })
        else:
            raise Exception('File verification failed after move operation')
            
    except FileNotFoundError as e:
        error_msg = f'File not found during move: {str(e)}'
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg, 'ok': False}), 404
    except PermissionError as e:
        error_msg = f'Permission denied: {str(e)}'
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg, 'ok': False}), 403
    except Exception as e:
        error_msg = f'Move failed: {str(e)}'
        print(f"❌ {error_msg}")
        return jsonify({'error': error_msg, 'ok': False}), 500




# ============================================================================
# SKU & Label Management Endpoints
# ============================================================================

@APP.route('/api/skus', methods=['GET'])
def get_skus():
    """Get all available SKUs."""
    try:
        from sku_manager import read_skus
        skus = read_skus()
        return jsonify(skus), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@APP.route('/api/skus', methods=['POST'])
def create_sku():
    """Create a new SKU."""
    try:
        from sku_manager import create_sku
        data = request.json
        sku_id = data.get('sku_id', '').strip()
        
        if not sku_id:
            return jsonify({'error': 'sku_id required'}), 400
        
        result = create_sku(sku_id)
        status_code = 201 if result['status'] == 'created' else 400
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@APP.route('/api/skus/<sku_id>', methods=['PUT'])
def update_sku(sku_id):
    """Update an existing SKU."""
    try:
        from sku_manager import update_sku
        data = request.json
        new_sku_id = data.get('new_sku_id', '').strip()
        
        if not new_sku_id:
            return jsonify({'error': 'new_sku_id required'}), 400
        
        result = update_sku(sku_id, new_sku_id)
        status_code = 200 if result['status'] == 'updated' else 400
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@APP.route('/api/skus/<sku_id>', methods=['DELETE'])
def delete_sku(sku_id):
    """Delete a SKU."""
    try:
        from sku_manager import delete_sku
        result = delete_sku(sku_id)
        status_code = 200 if result['status'] == 'deleted' else 400
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@APP.route('/api/load_json_labels', methods=['GET'])
def load_json_labels():
    """Load annotation labels from JSON file."""
    try:
        image_stem = request.args.get('image')
        if not image_stem:
            return jsonify({'error': 'image parameter required'}), 400
        
        ds = PROJECT / 'dataset'
        json_data_dir = ds / 'json_data'
        json_path = json_data_dir / f'{image_stem}.json'
        
        if not json_path.exists():
            return jsonify({'boxes': []}), 200
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return jsonify(data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@APP.route('/api/save_simple_labels', methods=['POST'])
def save_simple_labels():
    """Save annotation labels as individual JSON files."""
    try:
        data = request.json
        annotation = data.get('boxes', {})  # Get the full annotation object
        
        if not annotation:
            return jsonify({'error': 'annotation required'}), 400
        
        image_stem = annotation.get('image_stem')
        if not image_stem:
            return jsonify({'error': 'image_stem required'}), 400
        
        ds = PROJECT / 'dataset'
        json_data_dir = ds / 'json_data'
        json_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save annotation to individual JSON file with image name
        json_path = json_data_dir / f'{image_stem}.json'
        json_path.write_text(json.dumps(annotation, indent=2), encoding='utf-8')
        
        return jsonify({
            'success': True,
            'saved': str(json_path),
            'filename': json_path.name,
            'boxes_count': len(annotation.get('boxes', []))
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@APP.route('/api/dataset_stats', methods=['GET'])
def get_dataset_stats():
    """Get current dataset statistics."""
    try:
        from label_storage import get_dataset_stats
        stats = get_dataset_stats()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@APP.route('/api/images', methods=['GET'])
def get_images():
    """Get list of images in dataset."""
    try:
        from pathlib import Path
        images_dir = PROJECT / 'dataset' / 'images'
        if not images_dir.exists():
            return jsonify([]), 200
        
        image_files = sorted([f.name for f in images_dir.glob('image_*.jpg')])
        return jsonify(image_files), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@APP.route('/train', methods=['GET', 'POST'])
def training_endpoint():
    """Handle training requests.
    
    GET: Serve training page (kept for backwards compatibility)
    POST: Start training with parameters from query string
    """
    if request.method == 'GET':
        return send_from_directory(str(PROJECT / 'static'), 'training.html')
    
    # POST request - start training with custom parameters
    global training_progress
    
    try:
        # Get parameters from query string or form data
        epochs = int(request.args.get('epochs', 500))
        imgsz = int(request.args.get('imgsz', 640))
        batch = int(request.args.get('batch', 8))
        patience = int(request.args.get('patience', 100))
        device = request.args.get('device', '0')
        train_mode = request.args.get('mode', 'start')  # 'start' or 'finetune'
        
        # Validate parameters
        if epochs < 10 or epochs > 5000:
            return jsonify({'error': 'Epochs must be between 10 and 5000'}), 400
        if imgsz < 320 or imgsz > 1024:
            return jsonify({'error': 'Image size must be between 320 and 1024'}), 400
        if batch < 1 or batch > 256:
            return jsonify({'error': 'Batch size must be between 1 and 256'}), 400
        if patience < 5 or patience > 500:
            return jsonify({'error': 'Patience must be between 5 and 500'}), 400
        if train_mode not in ['start', 'finetune']:
            return jsonify({'error': 'Training mode must be "start" or "finetune"'}), 400
        
        # Check if training is already running
        if training_progress['status'] == 'training':
            return jsonify({'error': 'Training already in progress', 'started': False}), 400
        
        # Reset progress
        training_progress = {'status': 'training', 'progress': 0, 'message': 'Initializing...'}
        
        # Store training config in progress dict
        training_progress['patience'] = patience
        training_progress['mode'] = train_mode
        training_progress['device'] = device
        
        # Start training in background thread
        thread = threading.Thread(
            target=run_holo_training,
            args=(epochs, imgsz, batch),
            daemon=True
        )
        thread.start()
        
        mode_label = '♻️ Fine-tune' if train_mode == 'finetune' else '🆕 Train from Start'
        description = f'{mode_label}: {epochs} epochs, {imgsz}px, batch {batch}, patience {patience}'
        return jsonify({
            'started': True,
            'status': 'training',
            'description': description,
            'message': description,
            'mode': train_mode,
            'config': {
                'epochs': epochs,
                'imgsz': imgsz,
                'batch': batch,
                'patience': patience,
                'device': device,
                'mode': train_mode
            }
        }), 202
    
    except Exception as e:
        training_progress['status'] = 'error'
        training_progress['message'] = str(e)
        return jsonify({'error': str(e), 'started': False}), 500


# Global variable to track training progress
training_progress = {'status': 'idle', 'progress': 0, 'message': ''}


@APP.route('/api/training/config', methods=['GET'])
def get_training_config():
    """Get training configuration options."""
    return jsonify({
        'epochs': [50, 100, 200, 500, 1000],
        'imgsz': [320, 416, 512, 640, 800, 1024],
        'batch': [4, 8, 16, 32, 64],
        'defaults': {
            'epochs': 500,
            'imgsz': 640,
            'batch': 8
        }
    })


@APP.route('/api/training/start', methods=['POST'])
def start_training():
    """Start HOLO model training with custom parameters."""
    global training_progress
    
    try:
        data = request.get_json() or {}
        epochs = int(data.get('epochs', 500))
        imgsz = int(data.get('imgsz', 640))
        batch = int(data.get('batch', 8))
        
        # Validate parameters
        if epochs < 10 or epochs > 5000:
            return jsonify({'error': 'Epochs must be between 10 and 5000'}), 400
        if imgsz < 320 or imgsz > 1024:
            return jsonify({'error': 'Image size must be between 320 and 1024'}), 400
        if batch < 1 or batch > 256:
            return jsonify({'error': 'Batch size must be between 1 and 256'}), 400
        
        # Check if training is already running
        if training_progress['status'] == 'training':
            return jsonify({'error': 'Training already in progress'}), 400
        
        # Reset progress
        training_progress = {'status': 'training', 'progress': 0, 'message': 'Initializing...'}
        
        # Start training in background thread
        thread = threading.Thread(
            target=run_holo_training,
            args=(epochs, imgsz, batch),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': f'Training started: {epochs} epochs, {imgsz}px, batch {batch}'
        }), 202
    
    except Exception as e:
        training_progress['status'] = 'error'
        training_progress['message'] = str(e)
        return jsonify({'error': str(e)}), 500


@APP.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get current training status and progress."""
    return jsonify(training_progress), 200


@APP.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop the current training process."""
    global training_progress
    training_progress['status'] = 'stopping'
    training_progress['message'] = 'Stopping training...'
    return jsonify({'status': 'stop_requested'}), 200


def run_holo_training(epochs, imgsz, batch):
    """Run HOLO training using train.py script in background."""
    global training_progress
    import subprocess
    import sys
    import shutil
    from datetime import datetime
    
    try:
        training_progress['status'] = 'training'
        training_progress['message'] = 'Cleaning old training data...'
        training_progress['progress'] = 1
        
        print("\n" + "="*80)
        print("🚀 TRAINING STARTED")
        print("="*80)
        
        # Step 0: Clean old runs folder
        runs_folder = PROJECT / 'runs' / 'detect' / 'train'
        if runs_folder.exists():
            print(f"\n🧹 Removing old training data from {runs_folder}")
            sys.stdout.flush()
            shutil.rmtree(runs_folder)
            print("✓ Old training data removed")
            sys.stdout.flush()
        
        training_progress['progress'] = 2
        
        # Step 1: Auto-convert JSON to HOLO format text files
        training_progress['message'] = 'Converting JSON labels to text format...'
        try:
            from json_to_holo import main as convert_json
            print("\n📝 Auto-converting JSON annotations to HOLO text format...")
            sys.stdout.flush()
            convert_json()
            print("✓ JSON conversion complete")
            sys.stdout.flush()
        except Exception as e:
            print(f"⚠️  JSON conversion warning (non-fatal): {e}")
            sys.stdout.flush()
        
        training_progress['progress'] = 5
        training_progress['message'] = 'Starting training using train.py...'
        
        # Get training mode from progress dict
        train_mode = training_progress.get('mode', 'start')
        device = training_progress.get('device', '0')
        
        # Build command to run train.py
        train_script = PROJECT / 'train.py'
        
        if not train_script.exists():
            training_progress['status'] = 'error'
            training_progress['message'] = f'Error: train.py not found at {train_script}'
            print(f"❌ {training_progress['message']}")
            sys.stdout.flush()
            return
        
        # Build command line arguments
        cmd = [
            sys.executable,  # Use same Python interpreter
            str(train_script),
            '--epochs', str(epochs),
            '--batch', str(batch),
            '--imgsz', str(imgsz),
            '--device', str(device)
        ]
        
        mode_label = '♻️ Fine-tune' if train_mode == 'finetune' else '🆕 Train from Start'
        training_progress['message'] = f'{mode_label}: {epochs} epochs, {imgsz}px, batch {batch}'
        training_progress['progress'] = 10
        
        print(f"\n📊 {training_progress['message']}")
        print(f"📄 Command: {' '.join(cmd)}")
        print(f"📁 Working directory: {PROJECT}")
        sys.stdout.flush()
        
        # Run train.py as subprocess - DON'T capture output, let it stream
        print("\n" + "-"*80)
        print("TRAINING OUTPUT:")
        print("-"*80 + "\n")
        sys.stdout.flush()
        
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT),
            capture_output=False,  # Don't capture - let output stream
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        print("\n" + "-"*80)
        
        # Check result
        if result.returncode == 0:
            training_progress['message'] = 'Training completed! Backing up old model...'
            training_progress['progress'] = 95
            print("\n✓ Training completed successfully!")
            print("📦 Backing up old model and moving new model to scan_models...")
            sys.stdout.flush()
            
            # Step 2: Backup old model and move new one
            try:
                scan_models_dir = PROJECT / 'scan_models'
                scan_models_dir.mkdir(parents=True, exist_ok=True)
                old_model_path = scan_models_dir / 'scan_model.pt'
                
                # Backup old model if it exists
                if old_model_path.exists():
                    backup_dir = scan_models_dir / 'backups'
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    backup_path = backup_dir / f'scan_model_backup_{timestamp}.pt'
                    shutil.copy2(old_model_path, backup_path)
                    print(f"✓ Old model backed up to: {backup_path}")
                    sys.stdout.flush()
                
                # Copy new model
                new_model_path = runs_folder / 'weights' / 'best.pt'
                if new_model_path.exists():
                    shutil.copy2(new_model_path, old_model_path)
                    print(f"✓ New model moved to: {old_model_path}")
                    sys.stdout.flush()
                else:
                    raise FileNotFoundError(f"New model not found at {new_model_path}")
                
            except Exception as e:
                print(f"⚠️  Model backup/move warning: {e}")
                sys.stdout.flush()
            
            training_progress['status'] = 'completed'
            training_progress['progress'] = 100
            training_progress['message'] = '✅ Training completed and model updated!'
        else:
            training_progress['status'] = 'error'
            training_progress['progress'] = 0
            training_progress['message'] = f'Training failed with return code {result.returncode}'
            print(f"❌ {training_progress['message']}")
        
        sys.stdout.flush()
        print("="*80)
        
    except subprocess.TimeoutExpired:
        training_progress['status'] = 'error'
        training_progress['progress'] = 0
        training_progress['message'] = 'Training timeout (exceeded 1 hour)'
        print(f"❌ {training_progress['message']}")
    except Exception as e:
        training_progress['status'] = 'error'
        training_progress['progress'] = 0
        training_progress['message'] = f'Error: {str(e)}'
        print(f"❌ {training_progress['message']}")
        import traceback
        traceback.print_exc()


@APP.route('/api/models/info', methods=['GET'])
def get_models_info():
    """Get current model and available backups."""
    try:
        scan_models_dir = PROJECT / 'scan_models'
        current_model = scan_models_dir / 'scan_model.pt'
        backups_dir = scan_models_dir / 'backups'
        
        models = {
            'current': None,
            'backups': [],
            'backups_dir': str(backups_dir)
        }
        
        # Current model info
        if current_model.exists():
            stat = current_model.stat()
            models['current'] = {
                'name': 'scan_model.pt',
                'path': str(current_model),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        
        # Backups info
        if backups_dir.exists():
            for backup_file in sorted(backups_dir.glob('scan_model_backup_*.pt'), reverse=True):
                stat = backup_file.stat()
                models['backups'].append({
                    'name': backup_file.name,
                    'path': str(backup_file),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'timestamp': backup_file.stem.replace('scan_model_backup_', '')
                })
        
        return jsonify(models), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@APP.route('/api/models/restore', methods=['POST'])
def restore_model():
    """Restore a backup model as the current model."""
    try:
        data = request.get_json() or {}
        backup_name = data.get('backup_name')
        
        if not backup_name:
            return jsonify({'error': 'backup_name is required'}), 400
        
        scan_models_dir = PROJECT / 'scan_models'
        backups_dir = scan_models_dir / 'backups'
        backup_path = backups_dir / backup_name
        current_model = scan_models_dir / 'scan_model.pt'
        
        # Validate backup exists
        if not backup_path.exists():
            return jsonify({'error': f'Backup not found: {backup_name}'}), 404
        
        # Backup current model before restoring
        if current_model.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_backup = backups_dir / f'scan_model_backup_{timestamp}.pt'
            shutil.copy2(current_model, new_backup)
            print(f"✓ Current model backed up to: {new_backup}")
        
        # Restore backup
        shutil.copy2(backup_path, current_model)
        print(f"✓ Model restored from: {backup_path}")
        
        return jsonify({
            'success': True,
            'message': f'Model restored from backup: {backup_name}',
            'restored_from': backup_name,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@APP.route('/api/models/delete-backup', methods=['POST'])
def delete_backup():
    """Delete a backup model."""
    try:
        data = request.get_json() or {}
        backup_name = data.get('backup_name')
        
        if not backup_name:
            return jsonify({'error': 'backup_name is required'}), 400
        
        backups_dir = PROJECT / 'scan_models' / 'backups'
        backup_path = backups_dir / backup_name
        
        # Validate backup exists
        if not backup_path.exists():
            return jsonify({'error': f'Backup not found: {backup_name}'}), 404
        
        # Delete backup
        backup_path.unlink()
        print(f"✓ Backup deleted: {backup_path}")
        
        return jsonify({
            'success': True,
            'message': f'Backup deleted: {backup_name}',
            'deleted': backup_name,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@APP.route('/api/detect', methods=['POST'])
def detect_products():
    """Detect products in an uploaded image, crop them, and match to SKUs.
    
    Returns:
        {
            "success": bool,
            "product_count": int,
            "detections": [
                {
                    "id": 0,
                    "confidence": 0.95,
                    "box": [x, y, w, h],
                    "matched_sku": "BIRDS",
                    "sku_similarity": 0.87
                }
            ],
            "image_size": [width, height],
            "image_url": "/tmp/scan_...",
            "crops_url": "/tmp/scan_..._detected_crops.jpg",
            "sku_matches": {"BIRDS": 0.87, ...},
            "message": string
        }
    """
    try:
        # Check for image file
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'product_count': 0,
                'detections': [],
                'error': 'No image file provided'
            }), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({
                'success': False,
                'product_count': 0,
                'detections': [],
                'error': 'No file selected'
            }), 400
        
        # Load image
        from io import BytesIO
        import torch
        import uuid
        from datetime import datetime
        from sku_embeddings import SKUEmbeddingMatcher, load_sku_info
        
        img_data = image_file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'product_count': 0,
                'detections': [],
                'error': 'Could not decode image'
            }), 400
        
        height, width = image.shape[:2]
        
        # Save image to tmp/ folder
        tmp_dir = PROJECT / 'tmp'
        tmp_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        img_filename = f'scan_{timestamp}_{unique_id}.jpg'
        img_path = tmp_dir / img_filename
        cv2.imwrite(str(img_path), image)
        image_url = f'/tmp/{img_filename}'
        
        # Load trained model
        from hyperimagedetect import HOLO
        scan_model_path = PROJECT / 'scan_models' / 'scan_model.pt'
        
        if not scan_model_path.exists():
            return jsonify({
                'success': False,
                'product_count': 0,
                'detections': [],
                'error': f'Model not found at {scan_model_path}. Please train first.'
            }), 404
        
        # Load model and run inference
        model = HOLO(str(scan_model_path))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        # Run detection
        results = model.predict(image, conf=0.5, verbose=False)
        
        # Parse results
        detections = []
        product_count = 0
        
        if results and len(results) > 0:
            for i, result in enumerate(results):
                # Get boxes
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    for j, box in enumerate(boxes):
                        # Get box coordinates and confidence
                        try:
                            # box.xyxy = [x1, y1, x2, y2]
                            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else None
                            conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                            
                            if xyxy is not None:
                                x1, y1, x2, y2 = xyxy
                                detections.append({
                                    'id': product_count,
                                    'confidence': round(conf, 3),
                                    'box': [
                                        round(float(x1), 2),
                                        round(float(y1), 2),
                                        round(float(x2 - x1), 2),  # width
                                        round(float(y2 - y1), 2)   # height
                                    ]
                                })
                                product_count += 1
                        except Exception as e:
                            print(f"Error parsing detection: {e}")
                            continue
        
        # Initialize SKU matching
        crops_url = None
        sku_matches = {}
        
        if detections:
            try:
                # Initialize OpenCLIP matcher
                print("[OPENCLIP] Initializing OpenCLIP for SKU matching...")
                matcher = SKUEmbeddingMatcher(model_name="ViT-B-32", pretrained="openai")
                
                # Get SKU embeddings from dataset
                print("[OPENCLIP] Generating SKU embeddings...")
                sku_embeddings = matcher.get_sku_embeddings_from_dataset(
                    str(PROJECT / 'openclip_dataset')
                )
                
                # Build FAISS index for fast search (50-100x speedup)
                if sku_embeddings:
                    print("[FAISS] Building fast search index...")
                    matcher.build_faiss_index(sku_embeddings)
                
                # Crop detected regions
                print(f"[OPENCLIP] Cropping {len(detections)} detected regions...")
                crop_data = matcher.crop_detections(image, detections)
                crops = crop_data['crops']
                
                # Save crops
                image_base = img_filename.replace('.jpg', '')
                crops_path = matcher.save_crops(crops, str(tmp_dir), image_base)
                if crops_path:
                    crops_url = f'/tmp/{os.path.basename(crops_path)}'
                
                # Match each crop to SKU (uses FAISS if available, falls back to linear)
                print("[OPENCLIP] Matching crops to SKUs...")
                for i, crop_info in enumerate(crops):
                    crop_img = crop_info['image']
                    
                    # Save temporary crop for embedding generation
                    temp_crop_path = tmp_dir / f'crop_temp_{i}.jpg'
                    cv2.imwrite(str(temp_crop_path), crop_img)
                    
                    # Generate embedding
                    crop_embedding = matcher.get_image_embedding(str(temp_crop_path))
                    
                    if crop_embedding is not None:
                        # Find best match (FAISS or linear fallback)
                        match = matcher.match_sku(crop_embedding, sku_embeddings, threshold=0.3)
                        matched_sku = match.get('matched_sku')
                        similarity = match.get('similarity', 0.0)
                        
                        # Update detection with SKU info
                        if matched_sku:
                            detection_idx = crop_info['index']
                            if detection_idx < len(detections):
                                detections[detection_idx]['matched_sku'] = matched_sku
                                detections[detection_idx]['sku_similarity'] = round(similarity, 3)
                                
                                if matched_sku not in sku_matches:
                                    sku_matches[matched_sku] = similarity
                                else:
                                    sku_matches[matched_sku] = max(sku_matches[matched_sku], similarity)
                        
                        print(f"[OPENCLIP] Crop {i}: Matched to {matched_sku} (similarity: {similarity:.3f})")
                    
                    # Clean up temporary crop
                    try:
                        temp_crop_path.unlink()
                    except:
                        pass
                
                print(f"[OPENCLIP] SKU matching complete: {sku_matches}")
                
            except ImportError:
                print("[WARNING] OpenCLIP not installed. Skipping SKU matching.")
                print("  Run: pip install open-clip-torch")
            except Exception as e:
                print(f"[ERROR] SKU matching error: {e}")
                import traceback
                traceback.print_exc()
        
        response = {
            'success': True,
            'product_count': product_count,
            'detections': detections,
            'image_size': [width, height],
            'image_url': image_url,
            'message': f'Detected {product_count} product(s)'
        }
        
        if crops_url:
            response['crops_url'] = crops_url
        
        if sku_matches:
            response['sku_matches'] = {k: round(v, 3) for k, v in sku_matches.items()}
        
        def cleanup_detection_files():
            import time
            time.sleep(2)
            cleanup_errors = []
            try:
                if img_path.exists():
                    try:
                        img_path.unlink()
                    except Exception as e:
                        cleanup_errors.append(f"image file: {e}")
                if crops_path and Path(crops_path).exists():
                    try:
                        Path(crops_path).unlink()
                    except Exception as e:
                        cleanup_errors.append(f"crops file: {e}")
                for temp_file in tmp_dir.glob(f'crop_temp_*.jpg'):
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        cleanup_errors.append(f"temp file {temp_file.name}: {e}")
            except Exception as e:
                cleanup_errors.append(f"cleanup process: {e}")
            
            if cleanup_errors:
                print(f"[CLEANUP] Warnings during temp file cleanup: {', '.join(cleanup_errors)}")
        
        cleanup_thread = threading.Thread(target=cleanup_detection_files)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
        return jsonify(response), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'product_count': 0,
            'detections': [],
            'error': f'Detection error: {str(e)}'
        }), 500


@APP.route('/scan', methods=['GET'])
def scan_image_simple():
    """Simple GET endpoint for scanning images via URL - returns JSON API response.
    
    Usage:
        /scan?image=IMG_1445.jpeg                          (from dataset/images/)
        /scan?image=/uploads/image/IMG_1445.jpeg          (from uploads)
        /scan?image=/tmp/scan_20231230_145030.jpg         (from tmp)
        /scan?image=http://localhost:5002/uploads/image/IMG_1445.jpeg (full URL)
        /scan?image=http://example.com/photo.jpg           (remote URL)
    
    Returns: JSON with detections, SKU matches, and image URLs
    """
    image_param = request.args.get('image', '').strip()
    
    # Remove URL-encoded quotes and normalize
    image_param = image_param.replace('%22', '').replace('"', '').strip()
    
    if not image_param:
        return jsonify({
            'success': False,
            'product_count': 0,
            'detections': [],
            'error': 'No image provided. Usage: /scan?image=IMG_1445.jpeg or /scan?image=http://...'
        }), 400

    
    try:
        import cv2
        import numpy as np
        from pathlib import Path
        from io import BytesIO
        from datetime import datetime
        import uuid
        
        image = None
        filename = None
        
        # Load image based on parameter type
        if image_param.startswith('http://') or image_param.startswith('https://'):
            # Remote URL - download and process
            import requests
            response = requests.get(image_param, timeout=10)
            if response.status_code != 200:
                return jsonify({
                    'success': False,
                    'product_count': 0,
                    'detections': [],
                    'error': f'Could not fetch image from {image_param}'
                }), 400
            
            img_data = response.content
            img_array = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            filename = image_param.split('/')[-1]
        
        elif image_param.startswith('/'):
            # Server relative URL path - map to local file
            # SECURITY: Prevent directory traversal attacks
            if '..' in image_param or image_param.count('/') > 10:
                return jsonify({
                    'success': False,
                    'product_count': 0,
                    'detections': [],
                    'error': 'Invalid path - directory traversal not allowed'
                }), 400
            
            if image_param.startswith('/uploads/image/'):
                filename = image_param.replace('/uploads/image/', '')
                if '/' in filename:  # No nested directories
                    return jsonify({
                        'success': False,
                        'product_count': 0,
                        'detections': [],
                        'error': 'Invalid path - nested directories not allowed'
                    }), 400
                local_path = UPLOAD_IMAGES / filename
            elif image_param.startswith('/tmp/'):
                filename = image_param.replace('/tmp/', '')
                if '/' in filename:  # No nested directories
                    return jsonify({
                        'success': False,
                        'product_count': 0,
                        'detections': [],
                        'error': 'Invalid path - nested directories not allowed'
                    }), 400
                local_path = (PROJECT / 'tmp') / filename
            else:
                return jsonify({
                    'success': False,
                    'product_count': 0,
                    'detections': [],
                    'error': f'Invalid path format: {image_param}'
                }), 400
            
            # Verify resolved path is within allowed directories
            try:
                local_path = local_path.resolve()
                if image_param.startswith('/uploads/image/'):
                    allowed_dir = UPLOAD_IMAGES.resolve()
                else:
                    allowed_dir = (PROJECT / 'tmp').resolve()
                
                # Ensure the resolved path starts with the allowed directory
                if not str(local_path).startswith(str(allowed_dir)):
                    return jsonify({
                        'success': False,
                        'product_count': 0,
                        'detections': [],
                        'error': 'Access denied - path outside allowed directory'
                    }), 403
            except (ValueError, RuntimeError):
                return jsonify({
                    'success': False,
                    'product_count': 0,
                    'detections': [],
                    'error': 'Invalid path'
                }), 400
            
            if not local_path.exists():
                return jsonify({
                    'success': False,
                    'product_count': 0,
                    'detections': [],
                    'error': f'File not found: {image_param}'
                }), 400
            
            image = cv2.imread(str(local_path))
            filename = local_path.name
        
        else:
            # Simple filename - look in standard locations
            filename_to_find = Path(image_param).name
            
            # Try dataset/images first
            candidates = [
                UPLOAD_IMAGES / filename_to_find,
                (PROJECT / 'tmp') / filename_to_find,
                (PROJECT / 'dataset' / 'images') / filename_to_find,
            ]
            
            local_path = None
            for candidate in candidates:
                if candidate.exists():
                    local_path = candidate
                    break
            
            if not local_path:
                return jsonify({
                    'success': False,
                    'product_count': 0,
                    'detections': [],
                    'error': f'File not found: {filename_to_find}\nSearched in: dataset/images, uploads, tmp'
                }), 400
            
            image = cv2.imread(str(local_path))
            filename = local_path.name
        
        if image is None:
            return jsonify({
                'success': False,
                'product_count': 0,
                'detections': [],
                'error': f'Could not decode image: {image_param}'
            }), 400
        
        height, width = image.shape[:2]
        
        # Save image to tmp/ folder
        tmp_dir = PROJECT / 'tmp'
        tmp_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        img_filename = f'scan_{timestamp}_{unique_id}.jpg'
        img_path = tmp_dir / img_filename
        cv2.imwrite(str(img_path), image)
        image_url = f'/tmp/{img_filename}'
        
        # Load trained model and run detection
        from hyperimagedetect import HOLO
        scan_model_path = PROJECT / 'scan_models' / 'scan_model.pt'
        
        model = HOLO(str(scan_model_path))
        results = model.predict(image, conf=0.3, iou=0.5)
        
        detections = []
        product_count = 0
        crops = []
        
        for result in results:
            if result.boxes is not None:
                for idx, box in enumerate(result.boxes):
                    conf = float(box.conf[0]) if box.conf is not None else 0
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    
                    detections.append({
                        'id': product_count,
                        'confidence': round(conf, 3),
                        'box': [x, y, w, h],
                        'matched_sku': 'Unknown',
                        'sku_similarity': 0.0
                    })
                    
                    # Crop detection
                    crop_img = image[int(y1):int(y2), int(x1):int(x2)]
                    crops.append({'image': crop_img, 'index': product_count})
                    product_count += 1
        
        # Draw boxes
        crops_path = None
        if crops:
            import copy
            image_with_boxes = copy.deepcopy(image)
            for det in detections:
                x, y, w, h = det['box']
                cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image_with_boxes, f"{det['matched_sku']} {det['confidence']}", 
                           (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            crops_filename = f'scan_{timestamp}_{unique_id}_detected_crops.jpg'
            crops_path = tmp_dir / crops_filename
            cv2.imwrite(str(crops_path), image_with_boxes)
            crops_url = f'/tmp/{crops_filename}'
        else:
            crops_url = None
        
        # SKU Matching (using cached embeddings)
        sku_matches = {}
        try:
            # Initialize SKU embeddings if not already done
            if not SKU_EMBEDDINGS_INITIALIZED:
                initialize_sku_embeddings()
            
            if SKU_EMBEDDINGS_CACHE and MATCHER_INSTANCE:
                # Match detected crops to cached SKU embeddings
                for i, crop_info in enumerate(crops):
                    crop_img = crop_info['image']
                    temp_crop_path = tmp_dir / f'crop_temp_{i}.jpg'
                    
                    try:
                        cv2.imwrite(str(temp_crop_path), crop_img)
                        crop_embedding = MATCHER_INSTANCE.get_image_embedding(str(temp_crop_path))
                        
                        if crop_embedding is not None:
                            # Find best matching SKU
                            best_match = None
                            best_similarity = -1
                            
                            for sku_name, sku_embedding in SKU_EMBEDDINGS_CACHE.items():
                                # Cosine similarity
                                similarity = np.dot(crop_embedding, sku_embedding) / (
                                    np.linalg.norm(crop_embedding) * np.linalg.norm(sku_embedding) + 1e-8
                                )
                                
                                if similarity > best_similarity and similarity >= 0.3:
                                    best_similarity = similarity
                                    best_match = sku_name
                            
                            if best_match and best_similarity > 0:
                                detection_idx = crop_info['index']
                                if detection_idx < len(detections):
                                    detections[detection_idx]['matched_sku'] = best_match
                                    detections[detection_idx]['sku_similarity'] = round(float(best_similarity), 3)
                                
                                if best_match not in sku_matches or best_similarity > sku_matches.get(best_match, 0):
                                    sku_matches[best_match] = best_similarity
                    finally:
                        try:
                            temp_crop_path.unlink(missing_ok=True)
                        except:
                            pass
        except Exception as e:
            print(f"[SKU MATCH] Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Build JSON response
        response_data = {
            'success': True,
            'product_count': product_count,
            'detections': detections,
            'image_size': [width, height],
            'image_url': image_url,
            'message': f'Detected {product_count} product(s)'
        }
        
        if crops_url:
            response_data['crops_url'] = crops_url
        
        if sku_matches:
            response_data['sku_matches'] = {k: round(float(v), 3) for k, v in sku_matches.items()}
        
        # Schedule cleanup in background without waiting
        def cleanup_detection_files():
            try:
                import time
                import os
                time.sleep(3)
                # Clean up temp files
                try:
                    if img_path.exists():
                        img_path.unlink(missing_ok=True)
                except:
                    pass
                try:
                    if crops_path and Path(crops_path).exists():
                        Path(crops_path).unlink(missing_ok=True)
                except:
                    pass
                try:
                    for temp_file in tmp_dir.glob('crop_temp_*.jpg'):
                        temp_file.unlink(missing_ok=True)
                except:
                    pass
            except Exception as e:
                print(f"[CLEANUP] Error: {e}")
        
        import threading
        cleanup_thread = threading.Thread(target=cleanup_detection_files, daemon=True)
        cleanup_thread.start()
        
        return jsonify(response_data), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'product_count': 0,
            'detections': [],
            'error': f'Detection error: {str(e)}'
        }), 500


@APP.route('/api/dataset/skus', methods=['GET'])
def get_dataset_skus():
    """Get list of all SKUs from openclip_dataset folder."""
    try:
        dataset_dir = PROJECT / 'openclip_dataset'
        skus = []
        
        if dataset_dir.exists():
            for sku_dir in dataset_dir.iterdir():
                if sku_dir.is_dir():
                    # Count images in this SKU folder
                    image_count = len([f for f in sku_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']])
                    skus.append({
                        'name': sku_dir.name,
                        'count': image_count
                    })
        
        # Sort alphabetically
        skus.sort(key=lambda x: x['name'])
        
        return jsonify({
            'ok': True,
            'skus': skus,
            'total': len(skus)
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'skus': []}), 500


@APP.route('/api/dataset/sku/<sku_name>/images', methods=['GET'])
def get_sku_images(sku_name):
    """Get paginated images for a specific SKU.
    
    Query parameters:
    - page: page number (1-indexed, default=1)
    - per_page: images per page (default=25)
    """
    try:
        # Security: prevent directory traversal
        if '..' in sku_name or '/' in sku_name or '\\' in sku_name:
            return jsonify({'ok': False, 'error': 'invalid sku_name'}), 400
        
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 25, type=int)
        
        if page < 1:
            page = 1
        if per_page < 1 or per_page > 100:
            per_page = 25
        
        sku_dir = PROJECT / 'openclip_dataset' / sku_name
        
        if not sku_dir.exists() or not sku_dir.is_dir():
            return jsonify({
                'ok': False,
                'error': f'SKU folder not found: {sku_name}',
                'sku': sku_name,
                'images': [],
                'page': page,
                'per_page': per_page,
                'total': 0,
                'total_pages': 0
            }), 404
        
        # Get all image files, sorted by name
        all_images = sorted([f for f in sku_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']], key=lambda x: x.name)
        
        total = len(all_images)
        total_pages = (total + per_page - 1) // per_page
        
        # Paginate
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_images = all_images[start_idx:end_idx]
        
        images = []
        for img_file in page_images:
            images.append({
                'name': img_file.name,
                'size_bytes': img_file.stat().st_size,
                'url': f'/api/dataset/image/{sku_name}/{img_file.name}'
            })
        
        return jsonify({
            'ok': True,
            'sku': sku_name,
            'images': images,
            'page': page,
            'per_page': per_page,
            'total': total,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@APP.route('/api/dataset/image/<sku_name>/<image_name>')
def serve_dataset_image(sku_name, image_name):
    """Serve an image from a specific SKU folder."""
    try:
        # Security: prevent directory traversal
        if '..' in sku_name or '/' in sku_name or '\\' in sku_name:
            return jsonify({'error': 'invalid sku_name'}), 400
        if '..' in image_name or '/' in image_name or '\\' in image_name:
            return jsonify({'error': 'invalid image_name'}), 400
        
        sku_dir = PROJECT / 'openclip_dataset' / sku_name
        image_path = sku_dir / image_name
        
        if not image_path.exists() or not image_path.is_file():
            return jsonify({'error': 'image not found'}), 404
        
        # Verify it's in the correct directory
        if not str(image_path).startswith(str(sku_dir)):
            return jsonify({'error': 'access denied'}), 403
        
        return send_file(str(image_path), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@APP.route('/api/dataset/delete-image', methods=['POST'])
def delete_dataset_image():
    """Delete an image from a SKU folder.
    
    JSON body:
    - sku: SKU name
    - image: image filename
    """
    try:
        data = request.get_json()
        sku = data.get('sku', '').strip()
        image = data.get('image', '').strip()
        
        if not sku or not image:
            return jsonify({'ok': False, 'error': 'Missing sku or image'}), 400
        
        # Security: prevent directory traversal
        if '..' in sku or '/' in sku or '\\' in sku:
            return jsonify({'ok': False, 'error': 'invalid sku'}), 400
        if '..' in image or '/' in image or '\\' in image:
            return jsonify({'ok': False, 'error': 'invalid image'}), 400
        
        sku_dir = PROJECT / 'openclip_dataset' / sku
        image_path = sku_dir / image
        
        if not image_path.exists():
            return jsonify({'ok': False, 'error': 'image not found'}), 404
        
        # Verify it's in the correct directory
        if not str(image_path).startswith(str(sku_dir)):
            return jsonify({'ok': False, 'error': 'access denied'}), 403
        
        # Delete the image
        image_path.unlink()
        
        return jsonify({
            'ok': True,
            'message': f'Deleted {image} from {sku}',
            'sku': sku,
            'deleted_image': image
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@APP.route('/api/dataset/list', methods=['GET'])
def list_dataset_images():
    """Get dataset information with images, labels, and SKU mappings.
    
    Query params:
    - sku: Optional SKU filter
    - search: Optional image name search filter
    """
    try:
        dataset_dir = PROJECT / 'dataset'
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        json_data_dir = dataset_dir / 'json_data'
        
        sku_filter = request.args.get('sku', '').strip()
        search_filter = request.args.get('search', '').strip().lower()
        
        # Get all images
        images = {}
        if images_dir.exists():
            for img_file in sorted(images_dir.glob('*.*')):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_name = img_file.stem
                    
                    # Check search filter
                    if search_filter and search_filter not in img_name.lower():
                        continue
                    
                    # Get corresponding label
                    label_file = labels_dir / f'{img_name}.txt'
                    label_data = None
                    if label_file.exists():
                        with open(label_file, 'r') as f:
                            label_data = f.read().strip()
                    
                    # Get corresponding JSON metadata
                    json_file = json_data_dir / f'{img_name}.json'
                    json_data = None
                    skus = []
                    if json_file.exists():
                        try:
                            with open(json_file, 'r') as f:
                                json_data = json.load(f)
                                # Extract SKUs from boxes if available
                                if isinstance(json_data, dict) and 'boxes' in json_data:
                                    for box in json_data['boxes']:
                                        if 'sku' in box:
                                            skus.append(box['sku'])
                        except:
                            pass
                    
                    # Apply SKU filter
                    if sku_filter and sku_filter not in skus:
                        continue
                    
                    images[img_name] = {
                        'name': img_name,
                        'filename': img_file.name,
                        'size': img_file.stat().st_size,
                        'has_label': label_file.exists(),
                        'has_json': json_file.exists(),
                        'skus': sorted(set(skus)),
                        'label_content': label_data,
                        'json_data': json_data
                    }
        
        return jsonify({
            'total': len(images),
            'filter': {
                'sku': sku_filter,
                'search': search_filter
            },
            'images': images
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@APP.route('/api/dataset/stats', methods=['GET'])
def get_dataset_stats_v2():
    """Get dataset summary with statistics for labeling."""
    try:
        dataset_dir = PROJECT / 'dataset'
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'
        json_data_dir = dataset_dir / 'json_data'
        
        # Count files
        image_count = len(list(images_dir.glob('*.*'))) if images_dir.exists() else 0
        label_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        json_count = len(list(json_data_dir.glob('*.json'))) if json_data_dir.exists() else 0
        
        # Get unique SKUs from json_data (labeled boxes)
        skus_from_labels = set()
        if json_data_dir.exists():
            for json_file in json_data_dir.glob('*.json'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'boxes' in data:
                            for box in data['boxes']:
                                if 'sku' in box and box['sku']:
                                    skus_from_labels.add(box['sku'])
                except:
                    pass
        
        # Count boxes
        total_boxes = 0
        if json_data_dir.exists():
            for json_file in json_data_dir.glob('*.json'):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'boxes' in data:
                            total_boxes += len(data['boxes'])
                except:
                    pass
        
        return jsonify({
            'images': image_count,
            'labels': label_count,
            'json_files': json_count,
            'total_boxes': total_boxes,
            'unique_skus': len(skus_from_labels),
            'skus': sorted(skus_from_labels)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@APP.route('/api/dataset/sku/<sku_name>/image', methods=['DELETE'])
@safe_json
def delete_sku_image(sku_name):
    """Delete a specific image from a SKU folder."""
    openclip_dir = PROJECT / 'openclip_dataset'
    sku_folder = openclip_dir / sku_name
    
    data = request.get_json() or {}
    image_name = data.get('image_name') or data.get('image_path', '')
    
    # Extract just the filename from the path if full path provided
    image_name = Path(image_name).name
    image_file = sku_folder / image_name
    
    if not image_file.exists():
        return jsonify({'ok': False, 'error': f'Image not found: {image_name}'}), 404
    
    try:
        image_file.unlink()
        return jsonify({'ok': True, 'message': f'Deleted: {image_name}'}), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@APP.route('/api/dataset/sku/<sku_name>/image/<image_name>')
@safe_json
def serve_sku_image(sku_name, image_name):
    """Serve an image from a SKU folder."""
    openclip_dir = PROJECT / 'openclip_dataset'
    sku_folder = openclip_dir / sku_name
    image_path = sku_folder / image_name
    
    if not image_path.exists() or not image_path.is_file():
        return jsonify({'error': 'Image not found'}), 404
    
    return send_file(str(image_path), mimetype='image/jpeg')


@APP.route('/api/dataset/sku/<sku_name>/upload', methods=['POST'])
@safe_json
def upload_sku_image(sku_name):
    """Upload image(s) to a SKU folder."""
    openclip_dir = PROJECT / 'openclip_dataset'
    sku_folder = openclip_dir / sku_name
    
    # Create SKU folder if it doesn't exist
    sku_folder.mkdir(parents=True, exist_ok=True)
    
    if 'files' not in request.files:
        return jsonify({'ok': False, 'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    uploaded = []
    errors = []
    
    try:
        for file in files:
            if file.filename == '':
                continue
            
            # Allow only image files
            allowed_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
            file_ext = Path(file.filename).suffix.lower()
            
            if file_ext not in allowed_ext:
                errors.append(f'Invalid format: {file.filename}')
                continue
            
            # Save file
            file_path = sku_folder / file.filename
            file.save(str(file_path))
            uploaded.append(file.filename)
        
        return jsonify({
            'ok': True,
            'uploaded': uploaded,
            'errors': errors,
            'count': len(uploaded)
        }), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@APP.route('/api/dataset/sku/<sku_name>', methods=['DELETE'])
@safe_json
def delete_sku_folder(sku_name):
    """Delete entire SKU folder."""
    openclip_dir = PROJECT / 'openclip_dataset'
    sku_folder = openclip_dir / sku_name
    
    if not sku_folder.exists():
        return jsonify({'ok': False, 'error': f'SKU folder not found: {sku_name}'}), 404
    
    try:
        import shutil
        shutil.rmtree(str(sku_folder))
        return jsonify({'ok': True, 'message': f'Deleted SKU folder: {sku_name}'}), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500



if __name__ == '__main__':
    # t = threading.Thread(target=monitor_models, daemon=True)
    # t.start()
    APP.run(host='0.0.0.0', port=5002)