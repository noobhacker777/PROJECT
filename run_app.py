#!/usr/bin/env python3
"""
HOLO SKU Detection API Server
Simple Flask launcher without PyInstaller complications
Run: python run_app.py
"""

import sys
import os
from pathlib import Path

# Add project to path
PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))

if __name__ == '__main__':
    try:
        from image_model_app import APP
        
        print("=" * 80)
        print("HOLO SKU Detection API Server")
        print("=" * 80)
        print(f"Starting Flask server on http://localhost:5002")
        print(f"Project: {PROJECT}")
        print("")
        print("API Endpoints:")
        print("  POST /api/detect       - Upload image for detection + SKU matching")
        print("  GET  /api/train/status - Check training status")
        print("  POST /api/tmp/cleanup  - Clean temporary files")
        print("")
        print("Web Interface:")
        print("  GET  / - Dashboard")
        print("")
        print("Press Ctrl+C to stop server")
        print("=" * 80)
        print("")
        
        # Run Flask
        APP.run(
            host='0.0.0.0',
            port=5002,
            debug=False,
            use_reloader=False,
            threaded=True
        )
        
    except Exception as e:
        print(f"ERROR: Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
