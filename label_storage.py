#!/usr/bin/env python3
"""
Simplified Label Storage
========================

Saves images and labels directly:
- Images: dataset/images/image_{counter}.jpg
- Labels: dataset/labels/image_{counter}.txt (HOLO format, class=0)
- SKU Map: dataset/sku_mapping.json (maps image → SKU with box indices)
"""

import json
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent
DATASET = PROJECT / 'dataset'
IMAGES_DIR = DATASET / 'images'
LABELS_DIR = DATASET / 'labels'
SKU_MAPPING_FILE = DATASET / 'sku_mapping.json'


def ensure_dirs():
    """Create necessary directories."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)


def get_next_image_id():
    """Get next image counter."""
    ensure_dirs()
    jpg_files = list(IMAGES_DIR.glob('image_*.jpg'))
    if not jpg_files:
        return 1
    
    numbers = []
    for f in jpg_files:
        try:
            num = int(f.stem.split('_')[1])
            numbers.append(num)
        except:
            pass
    
    return max(numbers) + 1 if numbers else 1


def save_image_and_labels(image_bytes, boxes_with_sku):
    """
    Save image and labels.
    
    Args:
        image_bytes: Image file bytes (JPEG)
        boxes_with_sku: List of {
            'x_center': float (0-1),
            'y_center': float (0-1),
            'width': float (0-1),
            'height': float (0-1),
            'sku_id': str
        }
    
    Returns:
        {
            'success': bool,
            'image_id': int,
            'image_file': str,
            'label_file': str,
            'box_count': int,
            'message': str
        }
    """
    try:
        ensure_dirs()
        
        # Get image ID
        image_id = get_next_image_id()
        image_name = f"image_{image_id}"
        image_file = IMAGES_DIR / f"{image_name}.jpg"
        label_file = LABELS_DIR / f"{image_name}.txt"
        
        # Save image
        with open(image_file, 'wb') as f:
            f.write(image_bytes)
        
        # Save HOLO labels (class=0 only)
        holo_lines = []
        sku_entries = []
        
        for idx, box in enumerate(boxes_with_sku):
            x_center = box.get('x_center', 0)
            y_center = box.get('y_center', 0)
            width = box.get('width', 0)
            height = box.get('height', 0)
            sku_id = box.get('sku_id', 'UNKNOWN')
            
            # HOLO format: class_id x_center y_center width height
            holo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Track SKU assignment
            sku_entries.append({
                'box_index': idx,
                'sku_id': sku_id
            })
        
        # Save label file
        with open(label_file, 'w') as f:
            f.write('\n'.join(holo_lines))
        
        # Update SKU mapping
        sku_mapping = load_sku_mapping()
        sku_mapping[image_name] = sku_entries
        save_sku_mapping(sku_mapping)
        
        return {
            'success': True,
            'image_id': image_id,
            'image_file': f"image_{image_id}.jpg",
            'label_file': f"image_{image_id}.txt",
            'box_count': len(boxes_with_sku),
            'message': f'Saved image_{image_id} with {len(boxes_with_sku)} boxes'
        }
    
    except Exception as e:
        return {
            'success': False,
            'message': f'Error: {str(e)}'
        }


def load_sku_mapping():
    """Load SKU mapping from JSON."""
    ensure_dirs()
    if SKU_MAPPING_FILE.exists():
        try:
            with open(SKU_MAPPING_FILE) as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_sku_mapping(mapping):
    """Save SKU mapping to JSON."""
    ensure_dirs()
    with open(SKU_MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=2)


def get_dataset_stats():
    """Get current dataset statistics."""
    ensure_dirs()
    
    jpg_files = list(IMAGES_DIR.glob('image_*.jpg'))
    txt_files = list(LABELS_DIR.glob('image_*.txt'))
    
    sku_mapping = load_sku_mapping()
    
    # Count boxes per SKU
    sku_counts = {}
    for image_name, entries in sku_mapping.items():
        for entry in entries:
            sku_id = entry.get('sku_id', 'UNKNOWN')
            sku_counts[sku_id] = sku_counts.get(sku_id, 0) + 1
    
    return {
        'total_images': len(jpg_files),
        'total_labels': len(txt_files),
        'total_boxes': sum(len(entries) for entries in sku_mapping.values()),
        'sku_distribution': sku_counts,
        'image_ids': [int(f.stem.split('_')[1]) for f in jpg_files if '_' in f.stem]
    }


if __name__ == '__main__':
    stats = get_dataset_stats()
    print(json.dumps(stats, indent=2))
