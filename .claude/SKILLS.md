# Claude Code Skills: SCIGEN_p_agent Operations

Operational cheat sheet. All commands assume CWD is the project root:
`/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent`

---

## Environment Setup (Always Required)

```bash
conda activate scigen_py312      # actual env name (logical alias: scigen_p_agent)
export PYTHONNOUSERSITE=1        # prevent ~/.local pkg pollution
export MPLBACKEND=Agg            # headless matplotlib on login nodes
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
```

---

## Skill Index

| Task | Command / File | Details |
|---|---|---|
| Smoke test | `bash run_repro.sh --stage smoke` | CPU, ~2 min, no GPU |
| Verify datasets | `bash run_repro.sh --stage verify` | Checks _sym.pt counts |
| Train (mp_20, 1 GPU) | `sbatch hpc/job_repro_train.sh` | [Train](./commands/train.md) |
| Train (alex_2d, 2 GPU) | `sbatch hpc/job_repro_train_alex2d.sh` | [Train](./commands/train.md) |
| Generate (batch) | `sbatch hpc/job_repro_generate.sh` | [Generate](./commands/generate.md) |
| Generate (interactive) | `python script/generation.py ...` | [Generate](./commands/generate.md) |
| Screen results | `sbatch hpc/job_repro_screen.sh` | [Screen](./commands/screen.md) |
| Monitor SLURM | `squeue -u $USER` | [HPC](./commands/hpc.md) |
| Check logs | `tail -f slurm/<jobid>.log` | [HPC](./commands/hpc.md) |
| Debug failures | Check `.err` + [Debug](./commands/debug.md) | [Debug](./commands/debug.md) |

---

## Key Paths

| What | Path |
|---|---|
| Project root | `/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent` |
| Datasets (actual) | `data/alex_2d/`, `data/mp_20/` |
| Datasets (canonical) | `/pscratch/sd/r/ryotaro/datasets/SCIGEN_p_agent/` |
| Hydra runs | `/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun/<date>/<expname>/` |
| Best ckpt (alex_2d) | `hydra/singlerun/2026-01-14/alex2d_py312_2gpu/epoch=824-step=89100.ckpt` |
| Best ckpt (mp_20) | `hydra/singlerun/2026-01-12/test_py312/epoch=894-step=94870.ckpt` |
| SLURM logs | `./slurm/` |
| GNN evaluators | `gnn_eval/models/stab_*.torch`, `gnn_eval/models/mag_*.torch` |
| Config (mp_20) | `config_scigen.py` |
| Config (alex_2d) | `config_scigen_2d.py` |

---

## Critical Known Issues

### 1. User site-package pollution (wrong torch version in SLURM)
`~/.local/lib/python3.12/site-packages/` overrides conda env without `PYTHONNOUSERSITE=1`.
**Fix applied**: all SLURM scripts export it. Do not remove this line.

### 2. `config_scigen` ModuleNotFoundError
Scripts in `scripts/generation/` import `config_scigen` from project root.
Python's `sys.path` doesn't include the root when invoked from subdirs.
**Fix applied**: `job_gen_premium_mp20.sh` exports `PYTHONPATH=<root>:...`.
For interactive use: always `cd` to project root first, or `PYTHONPATH=. python scripts/...`.

### 3. PL 2.6+ strategy rejection
`strategy: null` rejected by PL 2.6.1. **Fix applied**: `conf/train/default.yaml` → `strategy: auto`.

### 4. Missing `script/generation_stop.py`
Called by `scripts/generation/gen_mul_stop*.py` (stop-ratio generation variant).
**Does NOT exist in any archive** (`archive/260119/script/` does not have it).
Must be created from scratch: copy `script/generation.py`, add `--stop_ratio` arg,
pass to model via `scigen/pl_modules/diffusion_w_type_stop.py`.

### 5. CUBLAS batched GEMM (CUDA 12.8 / torch 2.10 / A100)
`cublasSgemmStridedBatched` is broken on this platform.
Applies to `magmom_guided_generation` (CDVAE). SCIGEN training is unaffected (torch 2.5.1).
See MEMORY.md for patches if you encounter this in another project.
