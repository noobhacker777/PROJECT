#!/bin/bash

################################################################################
# HOLO SKU Detection API - Linux/WSL Installer
# Run this script to setup the environment and start the server
# Usage: bash start.sh
################################################################################

set -e

echo ""
echo "================================================================================"
echo "  HOLO SKU Detection API - Linux/WSL Setup"
echo "================================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Install with: sudo apt-get install python3 python3-pip"
    exit 1
fi

python3 --version
echo ""

# Run setup script
echo "Step 1: Installing dependencies..."
echo ""
python3 setup.py install

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Setup failed"
    exit 1
fi

echo ""
echo "Step 2: Setup complete! Starting Flask server..."
echo ""
echo "================================================================================"
echo "  API Server Starting"
echo "================================================================================"
echo ""
echo "Open in browser: http://localhost:5002"
echo "API endpoint:    http://localhost:5002/scan?image=IMG_1445.jpeg"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask app
python3 run_app.py
