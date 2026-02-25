# Configuration Guide

This document explains all configuration options in SCIGEN_p_agent, including Hydra configurations and user-specific settings.

## Overview

SCIGEN uses two types of configuration:

1. **Hydra Configurations** (`conf/`): YAML files for training, models, data, and optimization
2. **User Configurations** (`config_scigen*.py`): Python files for user-specific paths and settings

## Hydra Configuration

Hydra configurations are organized in the `conf/` directory and control training, models, data loading, and optimization.

### Configuration Structure

```
conf/
├── default.yaml          # Main config (includes all sub-configs)
├── data/                 # Dataset configurations
│   ├── default.yaml
│   ├── mp_20.yaml
│   └── alex_2d.yaml
├── model/                # Model architectures
│   ├── diffusion.yaml
│   ├── diffusion_w_type.yaml
│   └── diffusion_w_type_stop.yaml
├── train/                # Training settings
│   ├── default.yaml
│   └── multi_gpu.yaml
├── optim/                # Optimizer settings
│   └── default.yaml
└── logging/              # Logging settings
    └── default.yaml
```

### Data Configuration

**File**: `conf/data/<dataset>.yaml`

**Key Parameters**:

```yaml
root_path: ${oc.env:PROJECT_ROOT}/data/mp_20  # Dataset root directory
prop: formation_energy_per_atom              # Target property
num_targets: 1                                # Number of target properties
niggli: true                                  # Apply Niggli reduction
primitive: false                              # Use primitive cell
graph_method: crystalnn                       # Graph building method
lattice_scale_method: scale_length           # Lattice scaling method
preprocess_workers: 30                       # Number of preprocessing workers
max_atoms: 20                                # Maximum atoms per structure
otf_graph: false                             # On-the-fly graph building
tolerance: 0.1                               # Symmetry tolerance
use_space_group: true                        # Use space group information
use_pos_index: false                         # Use position index

datamodule:
  _target_: scigen.pl_data.datamodule.CrystDataModule
  datasets:
    train:
      _target_: scigen.pl_data.dataset.CrystDataset
      path: ${data.root_path}/train.csv
      save_path: ${data.root_path}/train_sym.pt
      # ... other dataset parameters
  num_workers:
    train: 0
    val: 0
    test: 0
  batch_size:
    train: 256
    val: 128
    test: 128
```

**Graph Methods**:
- `crystalnn`: CrystalNN algorithm (default)
- `voronoi`: Voronoi tessellation
- `cutoff`: Distance cutoff

**Lattice Scale Methods**:
- `scale_length`: Scale lattice lengths
- `scale_volume`: Scale lattice volume

### Model Configuration

**File**: `conf/model/<model_name>.yaml`

**Example**: `diffusion_w_type.yaml`

```yaml
_target_: scigen.pl_modules.diffusion_w_type.CSPDiffusion
time_dim: 256              # Time embedding dimension
latent_dim: 0             # Latent dimension (0 for no latent)
cost_coord: 1.0           # Coordinate loss weight
cost_lattice: 1.0         # Lattice loss weight
cost_type: 20.0           # Type loss weight
max_neighbors: 20         # Maximum neighbors for graph building
radius: 7.0               # Search radius for graph building
timesteps: 1000           # Number of diffusion timesteps

defaults:
  - decoder: cspnet        # Decoder architecture
  - beta_scheduler: cosine # Noise schedule
  - sigma_scheduler: wrapped # Sigma schedule
```

**Available Models**:
- `diffusion`: Basic diffusion model
- `diffusion_w_type`: Diffusion with type constraints
- `diffusion_w_type_stop`: Diffusion with type constraints and stop conditions

### Training Configuration

**File**: `conf/train/default.yaml`

```yaml
# Reproducibility
deterministic: true
random_seed: 42

# PyTorch Lightning Trainer
pl_trainer:
  fast_dev_run: false      # Debug mode (runs 1 batch)
  accelerator: gpu         # Device: gpu or cpu
  devices: 1               # Number of devices
  strategy: null           # DDP strategy (auto-selected if null)
  precision: 32            # Training precision (16, 32)
  max_epochs: ${data.train_max_epochs}
  accumulate_grad_batches: 1
  num_sanity_val_steps: 2
  gradient_clip_val: 0.5
  gradient_clip_algorithm: value
  profiler: simple

# Monitoring
monitor_metric: 'val_loss'
monitor_metric_mode: 'min'

# Early Stopping
early_stopping:
  patience: ${data.early_stopping_patience}
  verbose: false

# Model Checkpoints
model_checkpoints:
  save_top_k: 1           # Save top K models
  verbose: false
  save_last: false        # Save last checkpoint
```

**Multi-GPU Configuration**: `conf/train/multi_gpu.yaml`

```yaml
# Extends default.yaml
pl_trainer:
  devices: 4              # Number of GPUs
  strategy: ddp            # Distributed Data Parallel
```

### Optimizer Configuration

**File**: `conf/optim/default.yaml`

```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-4
  weight_decay: 1e-6

use_lr_scheduler: true

lr_scheduler:
  _target_: pytorch_lightning.callbacks.LearningRateMonitor
  # Or use ReduceLROnPlateau, etc.
```

### Logging Configuration

**File**: `conf/logging/default.yaml`

```yaml
wandb:
  project: scigen
  entity: your_entity
  mode: online              # online, offline, disabled
  name: ${expname}
  tags: ${core.tags}

wandb_watch:
  log: gradients           # gradients, parameters, all
  log_freq: 100           # Logging frequency

lr_monitor:
  logging_interval: step
  log_momentum: false

val_check_interval: 1      # Validate every N epochs
```

## User-Specific Configuration

### `config_scigen.py`

User-specific configuration file for paths and settings.

**Location**: Project root

**Variables**:

```python
home_dir = '/path/to/SCIGEN_p_agent'  # Project root (same as PROJECT_ROOT)
hydra_dir = '/path/to/hydra/singlerun'  # Hydra output directory
job_dir = 'yyyy-mm-dd/expname'  # Model checkpoint directory
out_name = 'sc_pyc1000_000'  # Generated materials filename
gnn_eval_path = './gnn_eval'  # GNN evaluation models path
stab_pred_name_A = "stab_240409-113155"  # Stability model A
stab_pred_name_B = "stab_240402-111754"  # Stability model B
mag_pred_name = "mag_240815-085301"  # Magnetic model
seedn = 42  # Random seed
```

**Usage**: Many scripts import from this file:
```python
from config_scigen import hydra_dir, job_dir, out_name
```

### `config_scigen_template.py`

Template file for new users. Copy to `config_scigen.py` and customize.

## Configuration Overrides

### Command-Line Overrides

Override any configuration value via command line:

```bash
python scigen/run.py \
    data=alex_2d \
    model=diffusion_w_type \
    train.pl_trainer.max_epochs=500 \
    optim.optimizer.lr=1e-5 \
    data.datamodule.batch_size.train=128
```

### Configuration Composition

Hydra supports configuration composition:

```bash
# Use multi-GPU training config
python scigen/run.py \
    data=alex_2d \
    model=diffusion_w_type \
    train=multi_gpu
```

### Environment Variables

Some values use environment variables:

```yaml
root_path: ${oc.env:PROJECT_ROOT}/data/mp_20
```

Set in `.env` file:
```bash
PROJECT_ROOT=/path/to/project
HYDRA_JOBS=/path/to/hydra/outputs
WANDB_DIR=/path/to/wandb
```

## Common Configuration Patterns

### Single GPU Training

```bash
python scigen/run.py \
    data=mp_20 \
    model=diffusion_w_type \
    expname=single_gpu_experiment
```

### Multi-GPU Training (4 GPUs)

```bash
python scigen/run.py \
    data=alex_2d \
    model=diffusion_w_type \
    expname=multi_gpu_experiment \
    train.pl_trainer.devices=4 \
    train.pl_trainer.strategy=ddp
```

### Debug Mode

```bash
python scigen/run.py \
    data=mp_20 \
    model=diffusion_w_type \
    train.pl_trainer.fast_dev_run=True
```

### Custom Batch Size

```bash
python scigen/run.py \
    data=mp_20 \
    model=diffusion_w_type \
    data.datamodule.batch_size.train=512
```

### Resume from Checkpoint

Checkpoints are automatically detected in the Hydra output directory. To resume:

```bash
python scigen/run.py \
    data=mp_20 \
    model=diffusion_w_type \
    expname=previous_experiment
```

## Configuration Best Practices

1. **Use Environment Variables**: For paths that vary between systems, use environment variables in `.env` file

2. **Version Control**: 
   - Commit `conf/` directory (Hydra configs)
   - Do NOT commit `config_scigen.py` (user-specific)
   - Commit `config_scigen_template.py` as a template

3. **Experiment Names**: Use descriptive experiment names:
   ```bash
   expname=alex2d_diffusion_w_type_4gpu
   ```

4. **Configuration Files**: Create dataset-specific configs in `conf/data/` for new datasets

5. **Multi-GPU**: Use `train=multi_gpu` config or override `devices` and `strategy` on command line

## Troubleshooting

### Configuration Not Found

**Error**: `MissingConfigException: Could not find 'data/default'`

**Solution**: Ensure `conf/data/default.yaml` exists, or specify a valid data config:
```bash
python scigen/run.py data=mp_20 ...
```

### Environment Variable Not Set

**Error**: `KeyError: 'PROJECT_ROOT not defined'`

**Solution**: Create `.env` file with required variables:
```bash
PROJECT_ROOT=/path/to/project
```

### Invalid Configuration Value

**Error**: `ValidationError: Invalid value for parameter`

**Solution**: Check configuration file syntax and valid values. Use Hydra's validation:
```bash
python scigen/run.py --cfg job  # Print full config
```

## See Also

- [Hydra Documentation](https://hydra.cc/)
- [PyTorch Lightning Configuration](https://pytorch-lightning.readthedocs.io/)
- [Workflow Guide](WORKFLOW.md)
- [Multi-GPU Setup](multi_gpu_setup_report_2026-01-13.md)



