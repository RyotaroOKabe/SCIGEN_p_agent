#!/usr/bin/env python
"""Quick start example for training a SCIGEN model.

This script demonstrates the minimal setup required to train a diffusion model
on the MP-20 dataset. It uses debug mode (fast_dev_run) for quick testing.

Usage:
    python examples/quick_start_training.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess


def main():
    """Run quick start training example."""
    print("=" * 70)
    print("SCIGEN Quick Start: Training Example")
    print("=" * 70)
    print()
    print("This example will:")
    print("  1. Train a diffusion model on MP-20 dataset")
    print("  2. Use debug mode (fast_dev_run) for quick testing")
    print("  3. Run for 1 batch to verify setup")
    print()
    
    # Check if data exists
    data_path = project_root / "data" / "mp_20" / "train.csv"
    if not data_path.exists():
        print(f"Warning: Data file not found: {data_path}")
        print("Please run data preparation first:")
        print("  python data_prep/load_mp20.py  # or appropriate script")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return 1
    
    # Run training command
    cmd = [
        sys.executable,
        "scigen/run.py",
        "data=mp_20",
        "model=diffusion_w_type",
        "expname=quick_start_example",
        "train.pl_trainer.fast_dev_run=True",
        "train.pl_trainer.accelerator=cpu",  # Use CPU for quick test
        "data.datamodule.num_workers.train=0",
    ]
    
    print("Running training command:")
    print(" ".join(cmd))
    print()
    
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        print()
        print("=" * 70)
        print("Training completed successfully!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Remove fast_dev_run=True for full training")
        print("  2. Use train.pl_trainer.accelerator=gpu for GPU training")
        print("  3. Adjust batch_size and other hyperparameters")
        return 0
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print(f"Training failed with error code {e.returncode}")
        print("=" * 70)
        print()
        print("Troubleshooting:")
        print("  1. Check that data files exist")
        print("  2. Verify environment is set up correctly")
        print("  3. Run: python scripts/validation/validate_setup.py")
        return 1
    except KeyboardInterrupt:
        print()
        print("Training interrupted by user.")
        return 1


if __name__ == "__main__":
    sys.exit(main())



