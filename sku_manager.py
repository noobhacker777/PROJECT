"""
SKU Management Utility Module
Handles CRUD operations for SKU list stored in JSON file
"""

import json
from pathlib import Path

# SKU storage file
SKU_FILE = Path(__file__).parent / 'dataset' / 'sku_list.json'

def ensure_sku_file():
    """Ensure SKU file exists."""
    SKU_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not SKU_FILE.exists():
        with open(SKU_FILE, 'w') as f:
            json.dump([], f)

def load_skus():
    """Load all SKUs from file."""
    ensure_sku_file()
    try:
        with open(SKU_FILE, 'r') as f:
            return sorted(json.load(f))
    except (json.JSONDecodeError, IOError):
        return []

def save_skus(skus):
    """Save SKUs to file."""
    ensure_sku_file()
    with open(SKU_FILE, 'w') as f:
        json.dump(sorted(skus), f, indent=2)

def create_sku(sku_id):
    """Create a new SKU.
    
    Args:
        sku_id (str): SKU identifier
        
    Returns:
        dict: Status and message
    """
    skus = load_skus()
    
    if sku_id in skus:
        return {'status': 'exists', 'message': f'SKU {sku_id} already exists'}
    
    skus.append(sku_id)
    save_skus(skus)
    return {'status': 'created', 'sku_id': sku_id}

def read_skus():
    """Read all SKUs.
    
    Returns:
        list: List of SKU identifiers
    """
    return load_skus()

def update_sku(old_sku_id, new_sku_id):
    """Update an existing SKU.
    
    Args:
        old_sku_id (str): Current SKU identifier
        new_sku_id (str): New SKU identifier
        
    Returns:
        dict: Status and message
    """
    skus = load_skus()
    
    if old_sku_id not in skus:
        return {'status': 'not_found', 'message': f'SKU {old_sku_id} not found'}
    
    if new_sku_id in skus and new_sku_id != old_sku_id:
        return {'status': 'exists', 'message': f'SKU {new_sku_id} already exists'}
    
    # Update the SKU and update sku_mapping.json if needed
    idx = skus.index(old_sku_id)
    skus[idx] = new_sku_id
    save_skus(skus)
    
    # Update sku_mapping.json to reflect the change
    from label_storage import load_sku_mapping, save_sku_mapping
    mapping = load_sku_mapping()
    for image, boxes in mapping.items():
        for box in boxes:
            if box.get('sku_id') == old_sku_id:
                box['sku_id'] = new_sku_id
    save_sku_mapping(mapping)
    
    return {'status': 'updated', 'old_sku_id': old_sku_id, 'new_sku_id': new_sku_id}

def delete_sku(sku_id):
    """Delete a SKU.
    
    Args:
        sku_id (str): SKU identifier to delete
        
    Returns:
        dict: Status and message
    """
    skus = load_skus()
    
    if sku_id not in skus:
        return {'status': 'not_found', 'message': f'SKU {sku_id} not found'}
    
    skus.remove(sku_id)
    save_skus(skus)
    
    # Remove from sku_mapping.json as well
    from label_storage import load_sku_mapping, save_sku_mapping
    mapping = load_sku_mapping()
    for image, boxes in mapping.items():
        boxes[:] = [box for box in boxes if box.get('sku_id') != sku_id]
    save_sku_mapping(mapping)
    
    return {'status': 'deleted', 'sku_id': sku_id}
