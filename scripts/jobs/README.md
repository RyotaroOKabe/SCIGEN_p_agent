# Job Scripts

This directory contains SLURM job scripts for running SCIGEN on HPC systems.

## Directory Structure

- **`training/`**: Training job scripts
- **`data_prep/`**: Data preparation job scripts

## Training Scripts

### Single GPU
- **`job_run_premium.sh`**: Single GPU training on premium queue

### Multi-GPU
- **`job_run_premium_2gpu.sh`**: 2-GPU training
- **`job_run_premium_multi_gpu.sh`**: 4-GPU training (default)
- **`job_run_premium_alex2d_multi_gpu.sh`**: 4-GPU training for Alexandria 2D dataset
- **`job_train_alex_2d.sh`**: Single GPU training for Alexandria 2D

## Data Preparation Scripts

- **`job_alex_load.sh`**: Load Alexandria dataset
- **`job_alex_process.sh`**: Process and split Alexandria data

## Usage

Submit jobs using `sbatch`:

```bash
# Training
sbatch scripts/jobs/training/job_run_premium_multi_gpu.sh

# Data preparation
sbatch scripts/jobs/data_prep/job_alex_load.sh
```

## Configuration

Edit the job script to modify:
- Job name
- Wall time
- Number of GPUs
- Dataset and model configurations
- Experiment name

## See Also

- [Multi-GPU Setup Guide](../../docs/multi_gpu_setup_report_2026-01-13.md)
- [Workflow Guide](../../docs/WORKFLOW.md)



