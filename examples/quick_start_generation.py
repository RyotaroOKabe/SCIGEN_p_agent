#!/usr/bin/env python
"""Quick start example for generating crystal structures.

This script demonstrates how to generate crystal structures from a trained model.
It requires a checkpoint file from a previous training run.

Usage:
    python examples/quick_start_generation.py --checkpoint path/to/checkpoint.ckpt
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run quick start generation example."""
    parser = argparse.ArgumentParser(
        description="Quick start example for structure generation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mp_20",
        help="Dataset name (for loading scalers)"
    )
    parser.add_argument(
        "--label",
        type=str,
        default="quick_start_gen",
        help="Label for generated structures"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of structures to generate per batch"
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=2,
        help="Number of batches to generate"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SCIGEN Quick Start: Generation Example")
    print("=" * 70)
    print()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Label: {args.label}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {args.num_batches}")
    print()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        print()
        print("Please provide a valid checkpoint file from a training run.")
        print("Example:")
        print("  python examples/quick_start_generation.py \\")
        print("    --checkpoint hydra_outputs/.../checkpoint.ckpt")
        return 1
    
    # Check if generation script exists
    gen_script = project_root / "script" / "generation.py"
    if not gen_script.exists():
        print(f"Error: Generation script not found: {gen_script}")
        return 1
    
    # Construct command
    cmd = [
        sys.executable,
        str(gen_script),
        "--model_path", str(checkpoint_path.parent),
        "--dataset", args.dataset,
        "--label", args.label,
        "--sc", "van",  # Vanilla (no constraint)
        "--batch_size", str(args.batch_size),
        "--num_batches_to_samples", str(args.num_batches),
        "--natm_range", "1", "20",
    ]
    
    print("Running generation command:")
    print(" ".join(cmd))
    print()
    
    try:
        import subprocess
        subprocess.run(cmd, check=True, cwd=project_root)
        print()
        print("=" * 70)
        print("Generation completed successfully!")
        print("=" * 70)
        print()
        print("Generated structures are saved in the Hydra output directory.")
        print("Next steps:")
        print("  1. Check generated structures in output directory")
        print("  2. Use script/save_cif.py to convert to CIF format")
        print("  3. Use script/eval_screen.py to evaluate structures")
        return 0
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 70)
        print(f"Generation failed with error code {e.returncode}")
        print("=" * 70)
        return 1
    except KeyboardInterrupt:
        print()
        print("Generation interrupted by user.")
        return 1


if __name__ == "__main__":
    sys.exit(main())



