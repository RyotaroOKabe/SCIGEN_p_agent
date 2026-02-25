# REPRODUCE.md — SCIGEN_p_agent End-to-End Reproduction Guide

> Generated: 2026-02-24
> System: NERSC Perlmutter (A100 GPUs, CUDA 12.8, Python 3.12)

---

## Directory Tree (key paths)

```
SCIGEN_p_agent/               ← PROJECT_ROOT
├── .env                      ← env vars (PROJECT_ROOT, HYDRA_JOBS, WANDB_DIR)
├── conf/                     ← Hydra configs
│   ├── default.yaml
│   ├── data/alex_2d.yaml, mp_20.yaml, ...
│   ├── model/diffusion_w_type.yaml
│   └── train/default.yaml, multi_gpu.yaml
├── data/
│   ├── alex_2d/              ← train/val/test.csv + train_sym.pt, val_sym.pt, test_sym.pt
│   └── mp_20/                ← train/val/test.csv + *_sym.pt
├── scigen/                   ← library (NOT pip-installed; uses sys.path tricks)
│   └── run.py                ← training entry point (Hydra)
├── script/                   ← generation + evaluation scripts
│   ├── generation.py         ← inference script
│   ├── eval_screen.py        ← screening (SMACT, stability, mag)
│   └── eval_utils.py, gen_utils.py, eval_funcs.py, ...
├── scripts/                  ← orchestrator loops (call script/ internals)
│   ├── generation/gen_mul_natm.py, gen_mul.py, ...
│   └── screening/screen_mul.py, screen_mul_2d.py, ...
├── gnn_eval/models/          ← pre-trained GNN evaluators (already present)
│   ├── stab_240409-113155.torch
│   ├── stab_240402-111754.torch
│   └── mag_240815-085301.torch
├── config_scigen.py          ← user config (hydra_dir, job_dir, out_name, etc.)
├── hpc/                      ← SLURM job scripts
│   ├── job_repro_train.sh    ← Stage 2: training (1-GPU, mp_20)
│   ├── job_repro_train_alex2d.sh  ← Stage 2b: training (2-GPU, alex_2d)
│   ├── job_repro_generate.sh ← Stage 3: generation
│   └── job_repro_screen.sh   ← Stage 4: screening
├── run_repro.sh              ← master smoke-test (local, fast_dev_run)
└── slurm/                    ← SLURM stdout/stderr logs

/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun/
    <date>/<expname>/         ← Hydra run directory per experiment
        *.ckpt                ← checkpoints (top-k + last)
        hparams.yaml
        lattice_scaler.pt
        prop_scaler.pt
        eval_gen_<label>.pt   ← generation output
        eval_<label>.pt       ← screening output
```

---

## Prerequisites

### Conda Environment

The `scigen_py312` env already exists:

```bash
conda env list   # should show scigen_py312
```

To recreate from scratch:
```bash
mamba env create -f setup/env.yml -n scigen_py312
conda activate scigen_py312
# torch-geometric scatter/sparse extensions (match torch+cuda version)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

**Note**: `scigen` is NOT installed as a pip package. The scripts rely on `sys.path.append('.')` in
`script/eval_utils.py` to make `import scigen` work when running from the project root.

### Environment Variables

The file `.env` is already configured:
```bash
cat .env
# export PROJECT_ROOT="/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent"
# export HYDRA_JOBS="/pscratch/sd/r/ryotaro/data/generative/hydra"
# export WANDB_DIR="/pscratch/sd/r/ryotaro/data/generative/wandb"
```

These are loaded automatically by `scigen/common/utils.py` via `python-dotenv`.

### WandB

Training defaults to **online** WandB logging. Options:
- Set `WANDB_API_KEY` in your shell / `~/.bashrc`
- Or pass `logging.wandb.mode=offline` to disable online sync

### Datasets

Data is already preprocessed (train/val/test split + graph cache):
```
data/alex_2d/   — train.csv, val.csv, test.csv, train_sym.pt, val_sym.pt, test_sym.pt
data/mp_20/     — same structure
```

No download or preprocessing needed for these two datasets.

---

## Stage 0: Smoke Test (local, ~2 min)

Run from the project root. This runs 1 training batch and exits (no GPU required):

```bash
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
source /pscratch/sd/r/ryotaro/anaconda3/bin/activate /pscratch/sd/r/ryotaro/anaconda3/envs/scigen_py312
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline

python scigen/run.py \
    data=mp_20 \
    model=diffusion_w_type \
    expname=smoke_repro \
    train.pl_trainer.max_epochs=1 \
    +train.pl_trainer.limit_train_batches=2 \
    +train.pl_trainer.limit_val_batches=2 \
    +train.pl_trainer.limit_test_batches=2 \
    train.pl_trainer.accelerator=cpu \
    train.pl_trainer.devices=1 \
    train.pl_trainer.strategy=auto \
    logging.val_check_interval=1 \
    logging.wandb.mode=offline
```

Expected: prints test metrics table and exits with status 0.
Output dir: `$HYDRA_JOBS/singlerun/<today>/smoke_repro/`

> **PL 2.6.1 notes**: `fast_dev_run=True` is broken for `trainer.test()` — it needs a saved
> checkpoint for `ckpt_path="best"`. Use the approach above instead. Always pass
> `train.pl_trainer.strategy=auto` for single-device runs (PL 2.6.1 rejects `strategy=null`).

Or use the master script:
```bash
bash run_repro.sh --stage smoke
```

---

## Stage 1: Data Verification (optional)

The `_sym.pt` graph cache files are already built. To verify dataset integrity:

```bash
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
conda activate scigen_py312 && export PYTHONNOUSERSITE=1

python - <<'EOF'
import torch
for ds, path in [("alex_2d", "data/alex_2d"), ("mp_20", "data/mp_20")]:
    for split in ["train", "val", "test"]:
        data = torch.load(f"{path}/{split}_sym.pt", weights_only=False)
        print(f"{ds}/{split}: {len(data)} samples")
EOF
```

Expected output:
```
alex_2d/train: ~7600 samples
alex_2d/val:   ~950  samples
alex_2d/test:  ~950  samples
mp_20/train:   ~27000 samples
mp_20/val:     ~3000  samples
mp_20/test:    ~3000  samples
```

---

## Stage 2: Training

### 2a. Submit 1-GPU training job (mp_20, ~24 h)

```bash
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
mkdir -p slurm
sbatch hpc/job_repro_train.sh
```

Checkpoint saved to:
```
/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun/<date>/repro_mp20_1gpu/epoch=XXX-step=YYYY.ckpt
```

Monitor:
```bash
squeue -u $USER
tail -f slurm/repro_train_<JOBID>.log
```

### 2b. Submit 2-GPU training job (alex_2d, ~24 h) — main experiment

```bash
sbatch hpc/job_repro_train_alex2d.sh
```

### Existing checkpoints (skip training if already done)

Two well-trained checkpoints already exist:

| Dataset | Expname | Epochs | Path |
|---------|---------|--------|------|
| mp_20   | test_py312 | 894 | `hydra/singlerun/2026-01-12/test_py312/` |
| alex_2d | alex2d_py312_2gpu | 824 | `hydra/singlerun/2026-01-14/alex2d_py312_2gpu/` |

Set `config_scigen.py` accordingly before generation:
```python
# For mp_20 model:
hydra_dir = '/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun'
job_dir   = '2026-01-12/test_py312'

# For alex_2d model:
hydra_dir = '/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun'
job_dir   = '2026-01-14/alex2d_py312_2gpu'
```

---

## Stage 3: Generation

Generation runs from the project root. `script/generation.py` uses the trained checkpoint
to sample novel crystal structures via constrained Langevin dynamics.

### 3a. Interactive (small, for testing)

Edit `config_scigen.py`:
```python
hydra_dir = '/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun'
job_dir   = '2026-01-14/alex2d_py312_2gpu'
out_name  = 'sc_kag1000_000'
```

Then run:
```bash
conda activate scigen_py312 && export PYTHONNOUSERSITE=1
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent

MODEL_PATH=/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun/2026-01-14/alex2d_py312_2gpu

python script/generation.py \
    --model_path  $MODEL_PATH \
    --dataset     alex_2d \
    --label       sc_kag100_000 \
    --sc          kag \
    --batch_size  20 \
    --num_batches_to_samples 5 \
    --natm_range  1 12 \
    --known_species Mn Fe Co Ni \
    --frac_z      0.5 \
    --t_mask      True
```

Output saved to: `$MODEL_PATH/eval_gen_sc_kag100_000.pt`

### 3b. Full batch via SLURM

```bash
sbatch hpc/job_repro_generate.sh
```

The job uses `scripts/generation/gen_mul_natm.py` as the orchestrator.

**Note**: Generation is read from the checkpoint directory and writes output `.pt` files there.

---

## Stage 4: Screening / Evaluation

Screening filters generated structures by:
1. SMACT charge-balance validity
2. Space occupation ratio (< 1.7)
3. Stability score via GNN (stab_A, stab_B)
4. (Optional) Magnetic moment score via GNN (mag)

### 4a. Interactive

```bash
conda activate scigen_py312 && export PYTHONNOUSERSITE=1
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent

python script/eval_screen.py \
    --label sc_kag100_000 \
    --gen_cif True \
    --screen_mag False
```

Output: filtered structures + CIF files in the checkpoint directory.

### 4b. Via SLURM

```bash
sbatch hpc/job_repro_screen.sh
```

---

## Stage 5: Expected Final Artifacts

After a complete run you should find, in the checkpoint directory
(`hydra/singlerun/<date>/<expname>/`):

```
epoch=XXX-step=YYYY.ckpt       ← trained model weights
hparams.yaml                   ← full Hydra config snapshot
lattice_scaler.pt              ← lattice normalizer
prop_scaler.pt                 ← property normalizer
eval_gen_<label>.pt            ← raw generation output tensor
eval_<label>.pt                ← post-screening filtered structures
cif_<label>/                   ← CIF files of passed structures (if gen_cif=True)
```

---

## Known Issues & Workarounds

### Wrong PyTorch version in SLURM jobs

Symptom: log shows `PyTorch version: 2.5.1` even though `scigen_py312` may have a different version.
Cause: `~/.local/lib/python3.12/site-packages/` (user site-packages) overrides conda env packages.
**Fix already applied**: All job scripts export `PYTHONNOUSERSITE=1`.

### `config_scigen` ModuleNotFoundError in generation scripts

`scripts/generation/gen_mul_elem.py` etc. import `config_scigen` but Python adds
`scripts/generation/` to sys.path — not the project root.
**Fix already applied** to `job_gen_premium_mp20.sh` via `export PYTHONPATH=<root>:${PYTHONPATH}`.
For interactive use: always run from the project root or prefix with `PYTHONPATH=. python scripts/...`

### `strategy: null` rejected by PL 2.6+

**Fix already applied**: `conf/train/default.yaml` now uses `strategy: auto`.

### Missing `script/generation_stop.py`

Called by `scripts/generation/gen_mul_stop_2d.py` and `gen_mul_stop.py` but not in the repo.
**It does not exist in any archive** (`archive/260119/script/` does not have it either).
The stop-ratio generation scripts are currently non-functional.
To enable them, create `script/generation_stop.py` by copying `script/generation.py` and
adding `--stop_ratio` support via `scigen/pl_modules/diffusion_w_type_stop.py`.

### CUBLAS batched GEMM bug (CUDA 12.8 / torch 2.10 / A100)

`cublasSgemmStridedBatched` is broken. Training has been confirmed to succeed
(checkpoints prove it). Generation uses the same GPU — if you hit a CUBLAS error
during generation, see `MEMORY.md` for the workaround patches.

### WandB connectivity

If the compute node has no internet: add `logging.wandb.mode=offline` to the
`python scigen/run.py` command. Logs will be stored locally and can be synced later:
```bash
wandb sync $WANDB_DIR/offline-run-*/
```

### "No Voronoi neighbors found" error

Occurs when `_sym.pt` cache files are missing and the dataset is built from scratch.
The existing `_sym.pt` files prevent this. If regenerating from CSV, reduce `tolerance`
in the data config or set a larger CrystalNN cutoff.

### Hydra output directory collision

If two jobs write to the same `<date>/<expname>` directory, the second job will load
the first job's checkpoint and resume training. This is intentional (resume support).
Use distinct `expname` values to start fresh.

---

## Quick Reference Commands

```bash
# Activate environment
source /pscratch/sd/r/ryotaro/anaconda3/bin/activate /pscratch/sd/r/ryotaro/anaconda3/envs/scigen_py312
export PYTHONNOUSERSITE=1

# Always run from project root
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent

# Smoke test (CPU, 1 epoch — fast_dev_run=True is broken in PL 2.6)
bash run_repro.sh --stage smoke

# Submit training
sbatch hpc/job_repro_train.sh           # mp_20, 1 GPU
sbatch hpc/job_repro_train_alex2d.sh    # alex_2d, 2 GPU

# Submit generation (edit config_scigen.py first)
sbatch hpc/job_repro_generate.sh

# Submit screening
sbatch hpc/job_repro_screen.sh

# Monitor
squeue -u $USER
tail -f slurm/repro_train_<JOBID>.log
```

---

## How to Use Claude Code Here

This repo is configured for [Claude Code](https://claude.ai/claude-code) with a multi-file
`.claude/` directory adapted from the `ht4matgen` and `magmom_guided_generation` project patterns.

### Starting a session

```bash
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
claude   # Claude Code picks up .claude/CLAUDE.md automatically
```

### What the `.claude/` directory contains

| File | Purpose |
|---|---|
| `CLAUDE.md` | Index — links to all other docs; first thing Claude reads |
| `project_context.md` | Datasets, checkpoints, pipeline architecture, exclusions |
| `RULES.md` | Hard constraints: env setup, secrets, naming, reproducibility |
| `SKILLS.md` | Operational cheat sheet — one-liners for every common task |
| `code_style.md` | Python formatting, logging, CLI, reproducibility conventions |
| `settings.local.json` | Pre-approved Bash commands (no prompt for common ops) |
| `commands/train.md` | Training reference: smoke test, SLURM, Hydra overrides |
| `commands/generate.md` | Generation reference: args, constraints, batch scripts |
| `commands/screen.md` | Screening reference: filters, config, SLURM |
| `commands/hpc.md` | SLURM submission, monitoring, log locations |
| `commands/debug.md` | Error catalog with root causes and fixes |

### Recommended prompts

**Run smoke test and report**:
```
Run the smoke test (bash run_repro.sh --stage smoke) and tell me if it passes.
```

**Submit a training job**:
```
Submit a 2-GPU alex_2d training job. Use seed 42 and offline WandB.
Monitor the queue and report when it starts running.
```

**Debug a failed job**:
```
Job 49322529 failed. Read the SLURM logs and diagnose the root cause.
Check .claude/commands/debug.md for known error patterns.
```

**Add a new structural constraint**:
```
I want to add support for the 'maple-leaf' lattice (code: mpl).
Read .claude/project_context.md and .claude/commands/generate.md first,
then propose where to add it in the codebase.
```

### Normalized path conventions

| Convention | Path |
|---|---|
| Env name (logical) | `scigen_p_agent` (actual conda env: `scigen_py312`) |
| Datasets (canonical) | `/pscratch/sd/r/ryotaro/datasets/SCIGEN_p_agent/` |
| Outputs (canonical) | `/pscratch/sd/r/ryotaro/outputs/SCIGEN_p_agent/<run_id>/` |
| Datasets (actual, today) | `data/alex_2d/`, `data/mp_20/` (project-local) |
| Hydra outputs (actual, today) | `/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun/` |

> **Note**: The canonical paths above are the target convention for future runs.
> Current data and checkpoints remain in their existing locations until symlinks or
> explicit moves are set up.
