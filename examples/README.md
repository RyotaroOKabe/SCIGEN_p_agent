# Examples

This directory contains example scripts to help you get started with SCIGEN.

## Quick Start Examples

### Training Example

**File**: `quick_start_training.py`

Minimal example for training a model:

```bash
python examples/quick_start_training.py
```

This script:
- Trains a diffusion model on MP-20 dataset
- Uses debug mode (fast_dev_run) for quick testing
- Runs on CPU for compatibility
- Verifies data files exist

**Customization**:
- Edit the script to change dataset, model, or parameters
- Remove `fast_dev_run=True` for full training
- Change `accelerator=cpu` to `accelerator=gpu` for GPU training

### Generation Example

**File**: `quick_start_generation.py`

Example for generating structures from a trained model:

```bash
python examples/quick_start_generation.py \
    --checkpoint path/to/checkpoint.ckpt \
    --dataset mp_20 \
    --label my_generation
```

**Options**:
- `--checkpoint`: Path to model checkpoint (required)
- `--dataset`: Dataset name for loading scalers (default: mp_20)
- `--label`: Label for generated structures (default: quick_start_gen)
- `--batch_size`: Number of structures per batch (default: 5)
- `--num_batches`: Number of batches to generate (default: 2)

## Multi-GPU Example

For multi-GPU training, use the job scripts:

```bash
# 4-GPU training
sbatch scripts/jobs/training/multi_gpu_4gpu.sh

# Or with command line
python scigen/run.py \
    data=alex_2d \
    model=diffusion_w_type \
    expname=multi_gpu_example \
    train.pl_trainer.devices=4 \
    train.pl_trainer.strategy=ddp
```

## Custom Examples

You can create your own examples by:

1. Copying an existing example script
2. Modifying parameters for your use case
3. Adding comments explaining your changes

## See Also

- [Workflow Guide](../docs/WORKFLOW.md): Complete workflow documentation
- [Configuration Guide](../docs/CONFIGURATION.md): Configuration options
- [API Documentation](../docs/API.md): Function reference



