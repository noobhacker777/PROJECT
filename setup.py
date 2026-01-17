#!/usr/bin/env python3
"""
Setup script for HOLO SKU Detection API
Installs all dependencies and prepares the environment to run run_app.py

Usage:
    python setup.py                 # Interactive setup
    python setup.py install         # Install dependencies
    python setup.py check           # Check if environment is ready
"""

import subprocess
import sys
from pathlib import Path


class Setup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / 'venv'
        self.requirements_file = self.project_root / 'requirements.txt'
    
    def print_header(self, text):
        """Print formatted header."""
        print(f"\n{'='*80}")
        print(f"  {text}")
        print(f"{'='*80}\n")
    
    def print_section(self, text):
        """Print section header."""
        print(f"\n→ {text}")
        print("-" * 60)
    
    def run_command(self, cmd, description=""):
        """Run a shell command and report status."""
        if description:
            print(f"  {description}...", end=" ")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                if description:
                    print("✓")
                return True
            else:
                if description:
                    print("✗")
                print(f"  Error: {result.stderr}")
                return False
        except Exception as e:
            if description:
                print("✗")
            print(f"  Error: {e}")
            return False
    
    def check_python_version(self):
        """Check if Python version is compatible."""
        self.print_section("Checking Python Version")
        
        version = sys.version_info
        print(f"  Python {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 9):
            print("\n  ⚠ Warning: Python 3.9+ is recommended for best compatibility")
            print(f"  Current: Python {version.major}.{version.minor}")
            return False
        
        print("  ✓ Python version OK")
        return True
    
    def check_requirements_file(self):
        """Check if requirements.txt exists."""
        if not self.requirements_file.exists():
            print(f"\n✗ Error: requirements.txt not found at {self.requirements_file}")
            return False
        
        print(f"  ✓ Found requirements.txt ({self.requirements_file})")
        return True
    
    def install_dependencies(self):
        """Install packages from requirements.txt."""
        self.print_section("Installing Dependencies")
        
        if not self.check_requirements_file():
            return False
        
        with open(self.requirements_file, 'r') as f:
            package_count = len(f.readlines())
        print(f"  Installing {package_count} packages...")
        print(f"  From: {self.requirements_file}\n")
        
        # Use pip install with requirements file
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file), "--upgrade"]
        
        # Convert list to string for display
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("  ✓ Dependencies installed successfully")
                return True
            else:
                print("  ✗ Failed to install dependencies")
                print(f"  Error: {result.stderr}")
                return False
        except Exception as e:
            print("  ✗ Failed to install dependencies")
            print(f"  Error: {e}")
            return False
        
        if False:  # This block will never execute, replacing the old logic
            print("\n  ✗ Failed to install dependencies")
            print("\n  Try manually running:")
            print(f"  {sys.executable} -m pip install -r {self.requirements_file}\n")
            return False
        
        print("  ✓ Dependencies installed successfully")
        return True
    
    def check_imports(self):
        """Check if critical packages are importable."""
        self.print_section("Checking Critical Imports")
        
        critical_packages = [
            ('flask', 'Flask'),
            ('torch', 'PyTorch'),
            ('cv2', 'OpenCV'),
            ('numpy', 'NumPy'),
            ('PIL', 'Pillow'),
            ('pydantic', 'Pydantic'),
        ]
        
        all_ok = True
        for module, name in critical_packages:
            try:
                __import__(module)
                print(f"  ✓ {name}")
            except ImportError:
                print(f"  ✗ {name} - NOT FOUND")
                all_ok = False
        
        return all_ok
    
    def create_directories(self):
        """Create necessary directories."""
        self.print_section("Creating Directories")
        
        dirs_to_create = [
            self.project_root / 'tmp',
            self.project_root / 'runs',
            self.project_root / 'dataset' / 'images',
            self.project_root / 'dataset' / 'labels',
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {dir_path.relative_to(self.project_root)}")
    
    def check_models(self):
        """Check if required model files exist."""
        self.print_section("Checking Model Files")
        
        model_files = [
            (self.project_root / 'scan_models' / 'scan_model.pt', 'scan_model.pt'),
        ]
        
        for model_path, model_name in model_files:
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ {model_name} ({size_mb:.1f} MB)")
            else:
                print(f"  ⚠ {model_name} - NOT FOUND")
                print(f"    Expected at: {model_path}")
    
    def show_next_steps(self):
        """Show next steps to run the app."""
        self.print_header("Setup Complete!")
        
        print("✓ Your environment is ready to run the application\n")
        print("Next Steps:")
        print("-" * 60)
        print(f"1. Start the API server:")
        print(f"   {sys.executable} run_app.py\n")
        print(f"2. Open in browser:")
        print(f"   http://localhost:5002\n")
        print(f"3. Use the API:")
        print(f"   http://localhost:5002/scan?image=IMG_1445.jpeg\n")
        print(f"4. View documentation:")
        print(f"   See SCAN_API_GUIDE.md for full API documentation\n")
    
    def verify_environment(self):
        """Verify the environment is ready without installing."""
        self.print_header("Verifying Environment")
        
        checks = [
            (self.check_python_version, "Python Version"),
            (self.check_requirements_file, "Requirements File"),
            (self.check_imports, "Package Imports"),
        ]
        
        all_ok = True
        for check_func, name in checks:
            if not check_func():
                all_ok = False
        
        if all_ok:
            print("\n✓ Environment is ready!")
        else:
            print("\n✗ Environment needs setup")
        
        return all_ok
    
    def full_setup(self):
        """Run full setup."""
        self.print_header("HOLO SKU Detection API - Environment Setup")
        
        print("This script will:")
        print("  1. Check Python version")
        print("  2. Install all required packages")
        print("  3. Verify critical imports")
        print("  4. Create necessary directories")
        print("  5. Check for model files")
        print("  6. Prepare environment to run run_app.py\n")
        
        # Run setup steps
        if not self.check_python_version():
            print("\n⚠ Continuing despite Python version warning...\n")
        
        if not self.install_dependencies():
            print("\n✗ Setup failed during dependency installation")
            return False
        
        if not self.check_imports():
            print("\n⚠ Warning: Some packages failed to import")
            print("  Try running setup again or check network connection\n")
        
        self.create_directories()
        self.check_models()
        self.show_next_steps()
        
        return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='HOLO SKU Detection API - Environment Setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup.py                 # Full interactive setup
  python setup.py install         # Install dependencies
  python setup.py check           # Check environment
  python setup.py help            # Show this help
        """
    )
    
    parser.add_argument(
        'action',
        nargs='?',
        default='install',
        choices=['install', 'check', 'help', 'verify'],
        help='Action to perform (default: install)'
    )
    
    args = parser.parse_args()
    
    setup = Setup()
    
    if args.action == 'install':
        success = setup.full_setup()
        sys.exit(0 if success else 1)
    
    elif args.action == 'check':
        success = setup.verify_environment()
        sys.exit(0 if success else 1)
    
    elif args.action == 'verify':
        success = setup.verify_environment()
        sys.exit(0 if success else 1)
    
    elif args.action == 'help':
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
