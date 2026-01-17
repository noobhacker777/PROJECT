#!/usr/bin/env python3
"""Convert JSON annotation files to HOLO/holo txt format.

Converts bounding boxes from JSON format (class_id, center_x, center_y, width, height)
to HOLO txt format with space-separated values.
"""

import json
from pathlib import Path


def convert_json_to_HOLO(json_file, output_dir):
    """Convert a JSON annotation file to HOLO format.
    
    Args:
        json_file: Path to JSON file
        output_dir: Directory to save output txt file
    
    Returns:
        Number of boxes processed, or None on error
    """
    try:
        # Read JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract image stem for output filename
        image_stem = data.get('image_stem', json_file.stem)
        output_file = output_dir / f'{image_stem}.txt'
        
        # Process boxes
        lines = []
        boxes = data.get('boxes', [])
        
        for box in boxes:
            bbox = box.get('bbox', [])
            
            # bbox should be [class_id, center_x, center_y, width, height]
            if len(bbox) >= 5:
                class_id = int(bbox[0])
                center_x = float(bbox[1])
                center_y = float(bbox[2])
                width = float(bbox[3])
                height = float(bbox[4])
                
                # Convert to HOLO format: class_id center_x center_y width height
                line = f"{class_id} {center_x} {center_y} {width} {height}"
                lines.append(line)
        
        # Write output file
        if lines:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            print(f"✓ {json_file.name} → {output_file.name} ({len(lines)} boxes)")
            return len(lines)
        else:
            print(f"⚠ {json_file.name}: No boxes found")
            return 0
    
    except Exception as e:
        print(f"✗ {json_file.name}: Error - {str(e)}")
        return None


def main():
    """Convert all JSON files to HOLO format."""
    # Set up paths
    project_root = Path(__file__).parent
    json_dir = project_root / 'dataset' / 'json_data'
    labels_dir = project_root / 'dataset' / 'labels'
    
    # Validate paths
    if not json_dir.exists():
        print(f"Error: JSON directory not found: {json_dir}")
        return False
    
    # Create output directory if needed
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = sorted(json_dir.glob('*.json'))
    
    if not json_files:
        print(f"Error: No JSON files found in {json_dir}")
        return False
    
    print(f"Converting {len(json_files)} JSON files to HOLO format...\n")
    
    # Convert each file
    total_boxes = 0
    successful = 0
    
    for json_file in json_files:
        result = convert_json_to_HOLO(json_file, labels_dir)
        if result is not None:
            successful += 1
            total_boxes += result
    
    print(f"\n{'='*60}")
    print(f"Conversion complete: {successful}/{len(json_files)} files processed")
    print(f"Total bounding boxes: {total_boxes}")
    print(f"Output directory: {labels_dir}")
    print(f"{'='*60}")
    
    return successful == len(json_files)


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
