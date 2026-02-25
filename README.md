# SCIGEN_p_agent: Targeted Materials Discovery

A PyTorch Lightning-based crystal structure generation system using diffusion models. This codebase enables training, generation, and evaluation of crystal structures for materials discovery.

## 🚀 Project Resources (For Investors & Partners)
- **[RESEARCH.md](RESEARCH.md)**: The core innovation (Structural Constraint Manifold) and Agentic Strategy.
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Deep dive into the CSPNet Diffusion Decoder system design.
- **[ROADMAP.md](ROADMAP.md)**: Strategic milestones from Foundation to Automomous Discovery.
- **[Colab Demo](demo_colab.ipynb)**: Interactive reproducibility notebook (Coming Soon).

## Overview

## Overview

SCIGEN_p_agent is a deep learning framework for generating crystal structures using denoising diffusion probabilistic models. It supports:

- **Training**: Train diffusion models on crystal structure datasets (Materials Project, Alexandria, etc.)
- **Generation**: Generate novel crystal structures with specified compositions and constraints
- **Evaluation**: Assess generated structures using various metrics
- **Multi-GPU Training**: Distributed training with PyTorch Lightning DDP

## Features

- **Diffusion Models**: Multiple diffusion model variants (with/without type constraints, stop conditions)
- **Crystal Graph Neural Networks**: GNN-based encoders for crystal structure representation
- **Flexible Data Loading**: Support for multiple datasets (MP-20, Alexandria 2D/3D)
- **Hydra Configuration**: Flexible configuration management using Hydra
- **WandB Integration**: Experiment tracking and logging
- **Multi-GPU Support**: Distributed training with automatic process management

## Installation

### Prerequisites

- Python 3.12+
- CUDA 11.8+ (for GPU training)
- Conda (for environment management)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd SCIGEN_p_agent
```

2. **Create conda environment**:
```bash
conda env create -f setup/env.yml
conda activate scigen_py312
```

3. **Install additional dependencies** (if needed):
```bash
pip install -e .
```

4. **Set up environment variables**:
Create a `.env` file in the project root with:
```bash
PROJECT_ROOT=/path/to/SCIGEN_p_agent
HYDRA_JOBS=/path/to/hydra/outputs
WANDB_DIR=/path/to/wandb
```

## Quick Start

### 1. Data Preparation

Prepare your dataset (example for Alexandria 2D):

```bash
# Load raw data
sbatch scripts/jobs/data_prep/alex_load.sh

# Process and split data
sbatch scripts/jobs/data_prep/alex_process.sh
```

This creates train/val/test CSV files in `data/alex_2d/`.

### 2. Training

Train a model on a single GPU:

```bash
python scigen/run.py \
    data=mp_20 \
    model=diffusion_w_type \
    expname=my_experiment
```

Train with multiple GPUs (4 GPUs):

```bash
sbatch scripts/jobs/training/multi_gpu_4gpu.sh
```

Or use command-line overrides:

```bash
python scigen/run.py \
    data=alex_2d \
    model=diffusion_w_type \
    expname=alex2d_training \
    train.pl_trainer.devices=4 \
    train.pl_trainer.strategy=ddp
```

### 3. Generation

Generate crystal structures from a trained model:

```bash
python scripts/generation/generation.py \
    --model_path /path/to/checkpoint.ckpt \
    --dataset mp_20 \
    --label my_generation \
    --sc pyc \
    --batch_size 20 \
    --num_batches_to_samples 50
```

### 4. Evaluation

Evaluate generated structures:

```bash
python scripts/evaluation/eval_screen.py \
    --results_dir /path/to/generated/structures
```

## Project Structure

```
SCIGEN_p_agent/
├── scigen/                 # Core training code
│   ├── run.py             # Main training entry point
│   ├── pl_modules/        # PyTorch Lightning model modules
│   │   ├── diffusion_w_type.py
│   │   ├── diffusion.py
│   │   └── gnn.py
│   ├── pl_data/           # Data loading modules
│   │   ├── datamodule.py
│   │   └── dataset.py
│   └── common/            # Utilities
│       ├── utils.py
│       └── data_utils.py
├── data_prep/              # Data preparation scripts
│   ├── alex_load.py       # Load Alexandria dataset
│   └── alex_process.py    # Process and split data
├── scripts/                # Generation and evaluation scripts
│   ├── generation/        # Structure generation
│   ├── evaluation/       # Evaluation metrics
│   ├── screening/         # Structure screening
│   └── jobs/              # SLURM job scripts
├── conf/                   # Hydra configuration files
│   ├── data/              # Dataset configurations
│   ├── model/             # Model configurations
│   ├── train/             # Training configurations
│   └── optim/             # Optimizer configurations
├── docs/                   # Documentation
│   ├── WORKFLOW.md        # Complete workflow guide
│   ├── API.md             # API documentation
│   └── CONFIGURATION.md   # Configuration guide
└── examples/              # Example scripts
```

## Workflow

The typical workflow consists of four main phases:

1. **Data Preparation**: Load and preprocess raw crystal structure data
2. **Training**: Train diffusion models on prepared datasets
3. **Generation**: Generate novel crystal structures
4. **Evaluation**: Assess generated structures

For detailed workflow documentation, see [docs/WORKFLOW.md](docs/WORKFLOW.md).

## Configuration

SCIGEN uses Hydra for configuration management. Key configuration files:

- `conf/data/`: Dataset configurations (mp_20.yaml, alex_2d.yaml)
- `conf/model/`: Model architectures (diffusion_w_type.yaml, diffusion.yaml)
- `conf/train/`: Training settings (default.yaml, multi_gpu.yaml)
- `conf/optim/`: Optimizer and scheduler settings

Override configurations via command line:

```bash
python scigen/run.py \
    data=alex_2d \
    model=diffusion_w_type \
    train.pl_trainer.max_epochs=100 \
    optim.optimizer.lr=1e-4
```

For detailed configuration documentation, see [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Multi-GPU Training

SCIGEN supports distributed training with PyTorch Lightning DDP. Key points:

- **Single node, multiple GPUs**: Use `train.pl_trainer.devices=N` and `train.pl_trainer.strategy=ddp`
- **SLURM integration**: Job scripts handle process spawning automatically
- **Performance**: Near-linear speedup with multiple GPUs

For detailed multi-GPU setup, see [docs/multi_gpu_setup_report_2026-01-13.md](docs/multi_gpu_setup_report_2026-01-13.md).

## Documentation

- [Workflow Guide](docs/WORKFLOW.md): Complete workflow from data to evaluation
- [API Documentation](docs/API.md): Core modules and functions
- [Configuration Guide](docs/CONFIGURATION.md): Configuration options and examples
- [Architecture](docs/ARCHITECTURE.md): System architecture and data flow
- [Multi-GPU Setup](docs/multi_gpu_setup_report_2026-01-13.md): Multi-GPU training guide

## Examples

See the `examples/` directory for:

- `quick_start_training.py`: Minimal training example
- `quick_start_generation.py`: Minimal generation example
- `multi_gpu_example.py`: Multi-GPU training example

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Code style and formatting
- Pull request process
- Testing requirements
- Documentation standards

## Citation

If you use this code in your research, please cite:

```bibtex
@software{scigen_p_agent,
  title = {SCIGEN_p_agent: Crystal Structure Generation with Diffusion Models},
  author = {Your Name},
  year = {2026},
  url = {<repository-url>}
}
```

## License

[Specify your license here]

## Acknowledgments

- PyTorch Lightning for the training framework
- PyTorch Geometric for graph neural network operations
- Materials Project and Alexandria datasets
- All contributors and collaborators

## Support

For questions, issues, or contributions, please:

1. Check the [documentation](docs/)
2. Search existing issues
3. Create a new issue with detailed information

---

**Last Updated**: January 2026  
**Python Version**: 3.12+  
**PyTorch Lightning Version**: 2.1.0+
