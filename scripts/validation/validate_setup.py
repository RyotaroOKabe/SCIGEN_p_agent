#!/usr/bin/env python
"""Validation script to check SCIGEN environment and dependencies.

This script verifies that the environment is properly configured for running
SCIGEN, including Python version, required packages, CUDA availability, and
data file existence.
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.12+.
    
    Returns:
        Tuple of (success, message)
    """
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.12+)"


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed.
    
    Args:
        package_name: Name of the package (for display)
        import_name: Name to import (if different from package_name)
    
    Returns:
        Tuple of (success, message)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, f"{package_name} ✓"
    except ImportError:
        return False, f"{package_name} ✗ (not installed)"


def check_cuda() -> Tuple[bool, str]:
    """Check if CUDA is available.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return True, f"CUDA available ({device_count} GPU(s): {device_name}) ✓"
        else:
            return False, "CUDA not available (CPU-only mode)"
    except ImportError:
        return False, "PyTorch not installed, cannot check CUDA"


def check_env_variables() -> List[Tuple[bool, str]]:
    """Check required environment variables.
    
    Returns:
        List of (success, message) tuples
    """
    results = []
    required_vars = {
        'PROJECT_ROOT': 'Project root directory',
        'HYDRA_JOBS': 'Hydra output directory',
    }
    optional_vars = {
        'WANDB_DIR': 'WandB directory',
    }
    
    for var, description in required_vars.items():
        value = os.environ.get(var)
        if value:
            path = Path(value)
            if path.exists():
                results.append((True, f"{var}={value} ✓"))
            else:
                results.append((False, f"{var}={value} ✗ (path does not exist)"))
        else:
            results.append((False, f"{var} ✗ (not set)"))
    
    for var, description in optional_vars.items():
        value = os.environ.get(var)
        if value:
            results.append((True, f"{var}={value} ✓ (optional)"))
        else:
            results.append((True, f"{var} not set (optional)"))
    
    return results


def check_data_files() -> List[Tuple[bool, str]]:
    """Check if common data files exist.
    
    Returns:
        List of (success, message) tuples
    """
    results = []
    project_root = os.environ.get('PROJECT_ROOT')
    
    if not project_root:
        return [(False, "PROJECT_ROOT not set, cannot check data files")]
    
    data_paths = [
        ('data/mp_20/train.csv', 'MP-20 training data'),
        ('data/mp_20/val.csv', 'MP-20 validation data'),
        ('data/alex_2d/train.csv', 'Alexandria 2D training data'),
        ('data/alex_2d/val.csv', 'Alexandria 2D validation data'),
    ]
    
    for rel_path, description in data_paths:
        full_path = Path(project_root) / rel_path
        if full_path.exists():
            results.append((True, f"{description}: {rel_path} ✓"))
        else:
            results.append((False, f"{description}: {rel_path} ✗ (not found)"))
    
    return results


def check_config_files() -> List[Tuple[bool, str]]:
    """Check if configuration files exist.
    
    Returns:
        List of (success, message) tuples
    """
    results = []
    project_root = os.environ.get('PROJECT_ROOT')
    
    if not project_root:
        return [(False, "PROJECT_ROOT not set, cannot check config files")]
    
    config_files = [
        ('config_scigen.py', 'User configuration file'),
        ('.env', 'Environment variables file'),
        ('conf/default.yaml', 'Default Hydra config'),
        ('conf/data/mp_20.yaml', 'MP-20 data config'),
        ('conf/model/diffusion_w_type.yaml', 'Diffusion model config'),
    ]
    
    for rel_path, description in config_files:
        full_path = Path(project_root) / rel_path
        if full_path.exists():
            results.append((True, f"{description}: {rel_path} ✓"))
        else:
            results.append((False, f"{description}: {rel_path} ✗ (not found)"))
    
    return results


def main():
    """Run all validation checks and print results."""
    print("=" * 70)
    print("SCIGEN Environment Validation")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # Python version
    print("Python Version:")
    success, msg = check_python_version()
    print(f"  {GREEN if success else RED}{msg}{RESET}")
    if not success:
        all_passed = False
    print()
    
    # Required packages
    print("Required Packages:")
    required_packages = [
        ('torch', 'torch'),
        ('pytorch_lightning', 'pytorch_lightning'),
        ('torch_geometric', 'torch_geometric'),
        ('hydra-core', 'hydra'),
        ('omegaconf', 'omegaconf'),
        ('pymatgen', 'pymatgen'),
        ('wandb', 'wandb'),
        ('numpy', 'numpy'),
    ]
    
    for pkg_name, import_name in required_packages:
        success, msg = check_package(pkg_name, import_name)
        print(f"  {GREEN if success else RED}{msg}{RESET}")
        if not success:
            all_passed = False
    print()
    
    # CUDA
    print("CUDA:")
    success, msg = check_cuda()
    print(f"  {GREEN if success else YELLOW}{msg}{RESET}")
    print()
    
    # Environment variables
    print("Environment Variables:")
    env_results = check_env_variables()
    for success, msg in env_results:
        print(f"  {GREEN if success else RED}{msg}{RESET}")
        if not success and 'PROJECT_ROOT' in msg or 'HYDRA_JOBS' in msg:
            all_passed = False
    print()
    
    # Configuration files
    print("Configuration Files:")
    config_results = check_config_files()
    for success, msg in config_results:
        print(f"  {GREEN if success else YELLOW}{msg}{RESET}")
    print()
    
    # Data files
    print("Data Files:")
    data_results = check_data_files()
    for success, msg in data_results:
        print(f"  {GREEN if success else YELLOW}{msg}{RESET}")
    print()
    
    # Summary
    print("=" * 70)
    if all_passed:
        print(f"{GREEN}All critical checks passed! ✓{RESET}")
        return 0
    else:
        print(f"{RED}Some checks failed. Please fix the issues above.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())



