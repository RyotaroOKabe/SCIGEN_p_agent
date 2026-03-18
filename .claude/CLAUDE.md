# CLAUDE.md: SCIGEN_p_agent

Main entrypoint for **SCIGEN — Structure-Constrained Crystal Generation** with diffusion models.

## Agent Context
- **Goal**: Train and deploy a Hydra/PyTorch-Lightning diffusion model for lattice-constrained crystal generation (kagome, lieb, honeycomb, pyrochlore, …).
- **Key Technologies**: DiffCSP-style diffusion, PyTorch Lightning 2.6, Hydra, WandB, SMACT, pyg.
- **Pipeline**: Train model → Generate constrained structures → SMACT+GNN screening → CIF outputs.

## Configuration Map
- [Project Context](./project_context.md): Scope, datasets, checkpoints, exclusions, overleaf papers.
- [Rules](./RULES.md): Hard constraints, naming conventions, SLURM/secrets policy.
- [Overleaf](./OVERLEAF.md): **Overleaf handling rules** — git sync, reading notes, implementation planning.
- [Skills](./SKILLS.md): Operational cheat sheet — one-liners for every common task.
- [Code Style](./code_style.md): Python formatting, logging, reproducibility conventions.

## Command Guides

### Core Pipeline
- [Train](./commands/train.md): Training from scratch or resuming; Hydra overrides.
- [Generate](./commands/generate.md): Constrained generation from a checkpoint.
- [Screen](./commands/screen.md): SMACT + GNN stability/mag screening.
- [HPC](./commands/hpc.md): SLURM submission, monitoring, log locations.
- [Debug](./commands/debug.md): Known errors and how to resolve them.

### Documentation
- [Progress](./commands/progress.md): **When and how to write progress reports** → `docs/progress/YYYY-MM-DD_<slug>.md`
- [Discussion](./commands/discussion.md): **Meeting notes and feedback reviews** → `docs/discussion/YYYY-MM-DD_<slug>.md`
- [Prompt](./commands/prompt.md): **Critical prompts for AI tools** → `docs/prompt/YYYY-MM-DD_<slug>.md`

### Overleaf
- [Overleaf Sync](./commands/overleaf-sync.md): **Safely sync overleaf git repos** with conflict checking
- [Overleaf Read](./commands/overleaf-read.md): **Start interactive reading session** for papers
- [Overleaf Plan](./commands/overleaf-plan.md): **Plan implementation** from paper concepts

## Quick Start

```bash
conda activate scigen_p_agent     # logical env name; actual: scigen_py312
export PYTHONNOUSERSITE=1         # REQUIRED — prevents ~/.local pkg pollution
export MPLBACKEND=Agg             # headless plotting on login nodes
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent

# Smoke test (CPU, ~2 min)
bash run_repro.sh --stage smoke

# Full training — submit to SLURM
sbatch hpc/job_repro_train_alex2d.sh     # alex_2d, 2 GPU, 24 h (main)
sbatch hpc/job_repro_train.sh            # mp_20,  1 GPU, 24 h

# Generate from best checkpoint
sbatch hpc/job_repro_generate.sh         # 1000 kagome structures

# Screen generated structures
sbatch hpc/job_repro_screen.sh
```

## Key Paths

| What | Path |
|---|---|
| Project root | `/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent` |
| Conda env (logical) | `scigen_p_agent` → actual: `scigen_py312` |
| Datasets (canonical) | `/pscratch/sd/r/ryotaro/datasets/SCIGEN_p_agent/` |
| Datasets (actual) | `data/alex_2d/`, `data/mp_20/` (project-local) |
| Hydra outputs | `/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun/<date>/<expname>/` |
| Outputs (canonical) | `/pscratch/sd/r/ryotaro/outputs/SCIGEN_p_agent/<run_id>/` |
| Best ckpt (alex_2d) | `hydra/singlerun/2026-01-14/alex2d_py312_2gpu/epoch=824-step=89100.ckpt` |
| Best ckpt (mp_20) | `hydra/singlerun/2026-01-12/test_py312/epoch=894-step=94870.ckpt` |
| SLURM logs | `./slurm/` |
| WandB artifacts | `/pscratch/sd/r/ryotaro/data/generative/wandb/` |
| Overleaf papers | `/pscratch/sd/r/ryotaro/data/generative/overleaf/` |
| Reading notes | `/pscratch/sd/r/ryotaro/data/generative/overleaf_notes/` |
