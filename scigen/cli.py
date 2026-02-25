#!/usr/bin/env python
"""Command-line interface for SCIGEN operations.

This module provides a unified CLI for training, generation, and evaluation.
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def train(
    data: str = "mp_20",
    model: str = "diffusion_w_type",
    expname: str = "default",
    devices: Optional[int] = None,
    strategy: Optional[str] = None,
    max_epochs: Optional[int] = None,
    **kwargs
) -> int:
    """Train a SCIGEN model.
    
    Args:
        data: Dataset configuration name (e.g., 'mp_20', 'alex_2d')
        model: Model configuration name (e.g., 'diffusion_w_type')
        expname: Experiment name
        devices: Number of GPU devices (None for default from config)
        strategy: Training strategy ('ddp' for multi-GPU, None for auto)
        max_epochs: Maximum number of training epochs
        **kwargs: Additional configuration overrides
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    cmd = [
        sys.executable,
        "scigen/run.py",
        f"data={data}",
        f"model={model}",
        f"expname={expname}",
    ]
    
    if devices is not None:
        cmd.append(f"train.pl_trainer.devices={devices}")
    
    if strategy is not None:
        cmd.append(f"train.pl_trainer.strategy={strategy}")
    
    if max_epochs is not None:
        cmd.append(f"train.pl_trainer.max_epochs={max_epochs}")
    
    # Add any additional overrides
    for key, value in kwargs.items():
        cmd.append(f"{key}={value}")
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=project_root).returncode


def generate(
    checkpoint: str,
    dataset: str = "mp_20",
    label: str = "generation",
    batch_size: int = 20,
    num_batches: int = 50,
    sc: str = "van",
    **kwargs
) -> int:
    """Generate crystal structures from a trained model.
    
    Args:
        checkpoint: Path to model checkpoint directory
        dataset: Dataset name (for loading scalers)
        label: Label for generated structures
        batch_size: Number of structures per batch
        num_batches: Number of batches to generate
        sc: Structure constraint ('van', 'pyc', 'kag', etc.)
        **kwargs: Additional generation parameters
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    gen_script = project_root / "script" / "generation.py"
    
    if not gen_script.exists():
        print(f"Error: Generation script not found: {gen_script}")
        return 1
    
    cmd = [
        sys.executable,
        str(gen_script),
        "--model_path", checkpoint,
        "--dataset", dataset,
        "--label", label,
        "--sc", sc,
        "--batch_size", str(batch_size),
        "--num_batches_to_samples", str(num_batches),
    ]
    
    # Add any additional parameters
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=project_root).returncode


def evaluate(
    results_dir: str,
    label: Optional[str] = None,
    **kwargs
) -> int:
    """Evaluate generated structures.
    
    Args:
        results_dir: Directory containing generated structures
        label: Label for evaluation (if None, uses default)
        **kwargs: Additional evaluation parameters
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    eval_script = project_root / "script" / "eval_screen.py"
    
    if not eval_script.exists():
        print(f"Error: Evaluation script not found: {eval_script}")
        return 1
    
    cmd = [
        sys.executable,
        str(eval_script),
        "--job_dir", results_dir,
    ]
    
    if label is not None:
        cmd.extend(["--label", label])
    
    # Add any additional parameters
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=project_root).returncode


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SCIGEN Command-Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python -m scigen.cli train --data mp_20 --model diffusion_w_type --expname my_experiment
  
  # Train with 4 GPUs
  python -m scigen.cli train --data alex_2d --devices 4 --strategy ddp
  
  # Generate structures
  python -m scigen.cli generate --checkpoint path/to/checkpoint --label my_gen
  
  # Evaluate structures
  python -m scigen.cli evaluate --results_dir path/to/results
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data", default="mp_20", help="Dataset config")
    train_parser.add_argument("--model", default="diffusion_w_type", help="Model config")
    train_parser.add_argument("--expname", default="default", help="Experiment name")
    train_parser.add_argument("--devices", type=int, help="Number of GPUs")
    train_parser.add_argument("--strategy", help="Training strategy (e.g., 'ddp')")
    train_parser.add_argument("--max_epochs", type=int, help="Maximum epochs")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate structures")
    gen_parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    gen_parser.add_argument("--dataset", default="mp_20", help="Dataset name")
    gen_parser.add_argument("--label", default="generation", help="Generation label")
    gen_parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    gen_parser.add_argument("--num_batches", type=int, default=50, help="Number of batches")
    gen_parser.add_argument("--sc", default="van", help="Structure constraint")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate structures")
    eval_parser.add_argument("--results_dir", required=True, help="Results directory")
    eval_parser.add_argument("--label", help="Evaluation label")
    
    args = parser.parse_args()
    
    if args.command == "train":
        return train(
            data=args.data,
            model=args.model,
            expname=args.expname,
            devices=args.devices,
            strategy=args.strategy,
            max_epochs=args.max_epochs,
        )
    elif args.command == "generate":
        return generate(
            checkpoint=args.checkpoint,
            dataset=args.dataset,
            label=args.label,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            sc=args.sc,
        )
    elif args.command == "evaluate":
        return evaluate(
            results_dir=args.results_dir,
            label=args.label,
        )
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())



