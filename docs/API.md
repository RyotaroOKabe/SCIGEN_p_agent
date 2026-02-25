# API Documentation

This document provides detailed API documentation for the core modules and functions in SCIGEN_p_agent.

## Table of Contents

- [Training Module (`scigen/run.py`)](#training-module)
- [Model Modules (`scigen/pl_modules/`)](#model-modules)
- [Data Modules (`scigen/pl_data/`)](#data-modules)
- [Common Utilities (`scigen/common/`)](#common-utilities)
- [Data Preparation (`data_prep/`)](#data-preparation)

## Training Module

### `scigen.run.run()`

Main training function that orchestrates the training process.

**Signature**:
```python
def run(cfg: DictConfig) -> None
```

**Parameters**:
- `cfg` (DictConfig): Complete Hydra configuration object containing all training parameters

**What it does**:
1. Sets random seed if deterministic mode is enabled
2. Configures debug mode if `fast_dev_run` is enabled
3. Instantiates data module and model
4. Sets up callbacks (early stopping, checkpointing, LR monitoring)
5. Configures WandB logger
6. Creates PyTorch Lightning Trainer
7. Runs training, validation, and testing

**Example**:
```python
from omegaconf import DictConfig
from scigen.run import run

# cfg is provided by Hydra decorator
run(cfg)
```

### `scigen.run.build_callbacks()`

Builds list of PyTorch Lightning callbacks based on configuration.

**Signature**:
```python
def build_callbacks(cfg: DictConfig) -> List[Callback]
```

**Parameters**:
- `cfg` (DictConfig): Configuration object

**Returns**:
- `List[Callback]`: List of configured callbacks

**Callbacks**:
- `LearningRateMonitor`: Monitors learning rate during training
- `EarlyStopping`: Stops training if validation metric doesn't improve
- `ModelCheckpoint`: Saves model checkpoints

## Model Modules

### `scigen.pl_modules.diffusion_w_type.CSPDiffusion`

Main diffusion model for crystal structure generation with type constraints.

**Inheritance**: `BaseModule` → `pl.LightningModule`

**Key Components**:
- `decoder`: GNN-based decoder (CSPNet)
- `beta_scheduler`: Noise schedule for diffusion process
- `sigma_scheduler`: Sigma schedule for wrapped normal distribution
- `time_embedding`: Sinusoidal time embeddings
- `crystal_family`: Crystal family constraints

**Methods**:

#### `forward()`

Forward pass for training.

**Signature**:
```python
def forward(self, batch, batch_idx=None) -> Dict[str, torch.Tensor]
```

**Parameters**:
- `batch`: Batch of crystal graph data
- `batch_idx`: Batch index (optional)

**Returns**:
- Dictionary containing loss and other metrics

#### `training_step()`

PyTorch Lightning training step.

**Signature**:
```python
def training_step(self, batch, batch_idx) -> torch.Tensor
```

**Returns**:
- Training loss tensor

#### `validation_step()`

PyTorch Lightning validation step.

**Signature**:
```python
def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]
```

**Returns**:
- Dictionary of validation metrics

#### `sample_scigen()`

Generate crystal structures using the diffusion model.

**Signature**:
```python
def sample_scigen(
    self,
    batch,
    step_lr: float = 1e-5,
    num_steps: int = 1000,
    save_traj: bool = False
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]
```

**Parameters**:
- `batch`: Input batch with composition constraints
- `step_lr`: Step size for Langevin dynamics
- `num_steps`: Number of diffusion steps
- `save_traj`: Whether to save generation trajectory

**Returns**:
- Tuple of (outputs, trajectory)
  - `outputs`: Dictionary with `frac_coords`, `atom_types`, `lattices`, etc.
  - `trajectory`: Optional trajectory dictionary if `save_traj=True`

**Example**:
```python
model = CSPDiffusion(...)
outputs, traj = model.sample_scigen(batch, step_lr=1e-5, num_steps=1000)
```

### `scigen.pl_modules.diffusion_w_type.BaseModule`

Base PyTorch Lightning module with optimizer configuration.

**Methods**:

#### `configure_optimizers()`

Configures optimizer and learning rate scheduler.

**Signature**:
```python
def configure_optimizers(self) -> Union[List[Optimizer], Dict]
```

**Returns**:
- Optimizer configuration dictionary or list

## Data Modules

### `scigen.pl_data.datamodule.CrystDataModule`

PyTorch Lightning data module for crystal structure datasets.

**Inheritance**: `pl.LightningDataModule`

**Initialization**:
```python
def __init__(
    self,
    datasets: DictConfig,
    num_workers: DictConfig,
    batch_size: DictConfig,
    scaler_path: Optional[str] = None
)
```

**Parameters**:
- `datasets`: Configuration for train/val/test datasets
- `num_workers`: Number of data loading workers per split
- `batch_size`: Batch sizes for each split
- `scaler_path`: Optional path to pre-computed scalers

**Methods**:

#### `setup()`

Sets up datasets for training/validation/testing.

**Signature**:
```python
def setup(self, stage: Optional[str] = None) -> None
```

**Parameters**:
- `stage`: One of `"fit"`, `"test"`, or `None` (both)

#### `train_dataloader()`

Returns training data loader.

**Signature**:
```python
def train_dataloader(self, shuffle: bool = True) -> DataLoader
```

**Returns**:
- PyTorch Geometric DataLoader for training data

#### `val_dataloader()`

Returns validation data loaders.

**Signature**:
```python
def val_dataloader(self) -> Sequence[DataLoader]
```

**Returns**:
- List of DataLoaders for validation sets

#### `test_dataloader()`

Returns test data loaders.

**Signature**:
```python
def test_dataloader(self) -> Sequence[DataLoader]
```

**Returns**:
- List of DataLoaders for test sets

#### `get_scaler()`

Computes or loads data scalers.

**Signature**:
```python
def get_scaler(self, scaler_path: Optional[str]) -> None
```

**Parameters**:
- `scaler_path`: Path to saved scalers, or None to compute from training data

**Attributes**:
- `lattice_scaler`: Scaler for lattice parameters
- `scaler`: Scaler for target property (e.g., formation energy)

### `scigen.pl_data.dataset.CrystDataset`

Dataset class for crystal structures.

**Inheritance**: `torch.utils.data.Dataset`

**Key Features**:
- Loads crystal structures from CSV files
- Builds crystal graphs using specified method (CrystalNN, Voronoi, etc.)
- Applies Niggli reduction and primitive cell conversion
- Caches processed structures

**Methods**:

#### `__getitem__()`

Gets a single crystal structure sample.

**Signature**:
```python
def __getitem__(self, idx: int) -> Data
```

**Returns**:
- PyTorch Geometric `Data` object with:
  - `x`: Node features (atom types)
  - `edge_index`: Graph edges
  - `edge_attr`: Edge attributes
  - `frac_coords`: Fractional coordinates
  - `lattice`: Lattice matrix
  - `y`: Target property value

## Common Utilities

### `scigen.common.utils`

Utility functions for environment and logging.

#### `get_env()`

Safely read an environment variable.

**Signature**:
```python
def get_env(env_name: str, default: Optional[str] = None) -> str
```

**Parameters**:
- `env_name`: Name of environment variable
- `default`: Optional default value

**Returns**:
- Environment variable value

**Raises**:
- `KeyError`: If variable not found and no default provided
- `ValueError`: If variable is empty and no default provided

#### `load_envs()`

Load environment variables from `.env` file.

**Signature**:
```python
def load_envs(env_file: Optional[str] = None) -> None
```

**Parameters**:
- `env_file`: Path to `.env` file, or None to search automatically

**Behavior**:
- Searches for `.env` in project root, current directory, or parent directories
- Loads variables using `python-dotenv`

#### `log_hyperparameters()`

Log hyperparameters to all configured loggers.

**Signature**:
```python
def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer
) -> None
```

**Parameters**:
- `cfg`: Hydra configuration
- `model`: PyTorch Lightning model
- `trainer`: PyTorch Lightning trainer

**What it logs**:
- All configuration parameters
- Total model parameters
- Trainable parameters
- Non-trainable parameters

### `scigen.common.data_utils`

Utilities for crystal structure data processing.

#### `build_crystal_graph()`

Builds crystal graph from structure.

**Signature**:
```python
def build_crystal_graph(
    structure: Structure,
    graph_method: str = "crystalnn",
    cutoff: float = 8.0,
    max_neighbors: int = 20
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

**Parameters**:
- `structure`: pymatgen Structure object
- `graph_method`: Method for building graph ("crystalnn", "voronoi", "cutoff")
- `cutoff`: Distance cutoff for graph building
- `max_neighbors`: Maximum number of neighbors per atom

**Returns**:
- Tuple of (edge_index, edge_attr, to_jimages)
  - `edge_index`: Graph edge indices
  - `edge_attr`: Edge attributes (distances, offsets)
  - `to_jimages`: Periodic image offsets

#### `get_scaler_from_data_list()`

Computes scaler (mean/std) from list of data dictionaries.

**Signature**:
```python
def get_scaler_from_data_list(
    data_list: List[Dict],
    key: str
) -> Tuple[float, float]
```

**Parameters**:
- `data_list`: List of data dictionaries
- `key`: Key to extract from each dictionary

**Returns**:
- Tuple of (mean, std)

#### `cart_to_frac_coords()`

Convert Cartesian to fractional coordinates.

**Signature**:
```python
def cart_to_frac_coords(
    cart_coords: torch.Tensor,
    lattice: torch.Tensor
) -> torch.Tensor
```

**Parameters**:
- `cart_coords`: Cartesian coordinates [N, 3]
- `lattice`: Lattice matrix [3, 3]

**Returns**:
- Fractional coordinates [N, 3]

#### `frac_to_cart_coords()`

Convert fractional to Cartesian coordinates.

**Signature**:
```python
def frac_to_cart_coords(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor
) -> torch.Tensor
```

**Parameters**:
- `frac_coords`: Fractional coordinates [N, 3]
- `lattice`: Lattice matrix [3, 3]

**Returns**:
- Cartesian coordinates [N, 3]

#### `lattice_params_to_matrix_torch()`

Convert lattice parameters to matrix.

**Signature**:
```python
def lattice_params_to_matrix_torch(
    lengths: torch.Tensor,
    angles: torch.Tensor
) -> torch.Tensor
```

**Parameters**:
- `lengths`: Lattice lengths [a, b, c]
- `angles`: Lattice angles [alpha, beta, gamma] (in radians)

**Returns**:
- Lattice matrix [3, 3]

#### `min_distance_sqr_pbc()`

Compute minimum distance with periodic boundary conditions.

**Signature**:
```python
def min_distance_sqr_pbc(
    coords1: torch.Tensor,
    coords2: torch.Tensor,
    lattice: torch.Tensor,
    to_jimages: torch.Tensor
) -> torch.Tensor
```

**Parameters**:
- `coords1`: First set of coordinates [N1, 3]
- `coords2`: Second set of coordinates [N2, 3]
- `lattice`: Lattice matrix [3, 3]
- `to_jimages`: Periodic image offsets [E, 3]

**Returns**:
- Minimum squared distances [E]

## Data Preparation

### `data_prep.alex_load`

Script for loading Alexandria dataset.

**Main Function**: Script execution (no function exports)

**What it does**:
1. Downloads compressed JSON files from Alexandria database
2. Extracts crystal structure information
3. Converts to CSV format with required columns
4. Saves individual CSV files per data chunk

**Output**: CSV files in `data/alex_2d/` or `data/alex_3d/`

### `data_prep.alex_process`

Script for processing and splitting Alexandria data.

**Main Function**: Script execution (no function exports)

**What it does**:
1. Loads all CSV files from data directory
2. Applies filters (formation energy, energy above hull, number of atoms)
3. Splits data into train/val/test sets
4. Generates visualization plots
5. Saves final CSV files

**Output**:
- `train.csv`, `val.csv`, `test.csv`
- Visualization plots

## Constants

### `scigen.common.data_utils`

- `EPSILON`: Small epsilon value (1e-5) for numerical stability
- `OFFSET_LIST`: List of 27 periodic image offsets for 3D crystals
- `chemical_symbols`: List of chemical element symbols
- `CrystalNN`: Pre-configured CrystalNN for graph building

### `scigen.common.utils`

- `PROJECT_ROOT`: Path to project root directory (from environment)
- `STATS_KEY`: Key prefix for statistics in hyperparameter logging

## Type Definitions

Common type hints used throughout the codebase:

```python
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from omegaconf import DictConfig
import torch
from torch_geometric.data import Data, DataLoader
```

## Error Handling

Most functions raise standard Python exceptions:

- `KeyError`: Missing required configuration or environment variable
- `ValueError`: Invalid parameter values
- `FileNotFoundError`: Missing required files
- `RuntimeError`: Runtime errors during model execution

## Examples

### Training a Model

```python
from scigen.run import run
from omegaconf import DictConfig

# Configuration is provided by Hydra
# Command: python scigen/run.py data=mp_20 model=diffusion_w_type
run(cfg)
```

### Generating Structures

```python
import torch
from scigen.pl_modules.diffusion_w_type import CSPDiffusion

# Load model
model = CSPDiffusion.load_from_checkpoint("checkpoint.ckpt")
model.eval()

# Prepare batch
batch = prepare_batch(compositions, num_atoms)

# Generate
with torch.no_grad():
    outputs, traj = model.sample_scigen(
        batch,
        step_lr=1e-5,
        num_steps=1000,
        save_traj=True
    )
```

### Using Data Module

```python
from scigen.pl_data.datamodule import CrystDataModule
from omegaconf import DictConfig

# Instantiate from config
datamodule = CrystDataModule(
    datasets=cfg.data.datamodule.datasets,
    num_workers=cfg.data.datamodule.num_workers,
    batch_size=cfg.data.datamodule.batch_size
)

# Setup
datamodule.setup("fit")

# Get data loaders
train_loader = datamodule.train_dataloader()
val_loaders = datamodule.val_dataloader()
```

## See Also

- [Workflow Guide](WORKFLOW.md): Complete workflow documentation
- [Configuration Guide](CONFIGURATION.md): Configuration options
- [Architecture](ARCHITECTURE.md): System architecture details



