# Command: Train

Train or resume the diffusion model via Hydra + PyTorch Lightning.

---

## Environment

```bash
conda activate scigen_py312
export PYTHONNOUSERSITE=1
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
```

---

## Smoke Test (CPU, ~2 min — run this first)

```bash
bash run_repro.sh --stage smoke
```

Or manually:
```bash
python scigen/run.py \
    data=mp_20 model=diffusion_w_type expname=smoke_repro \
    train.pl_trainer.max_epochs=1 \
    +train.pl_trainer.limit_train_batches=2 \
    +train.pl_trainer.limit_val_batches=2 \
    +train.pl_trainer.limit_test_batches=2 \
    train.pl_trainer.accelerator=cpu \
    train.pl_trainer.strategy=auto \
    logging.val_check_interval=1 \
    logging.wandb.mode=offline
```

Expected: prints test metrics table and exits 0.

---

## Full Training — SLURM

```bash
# mp_20, 1 GPU, 24 h (reproducible)
sbatch hpc/job_repro_train.sh

# alex_2d, 2 GPU DDP, 24 h (main experiment)
sbatch hpc/job_repro_train_alex2d.sh

# Legacy scripts (also patched with PYTHONNOUSERSITE=1)
sbatch job_run_premium_mp20.sh
sbatch job_run_premium_mp20_2gpu.sh
sbatch job_run_premium_alex2d_2gpu.sh
```

Monitor:
```bash
squeue -u $USER
tail -f slurm/repro_train_<JOBID>.log
```

---

## Common Hydra Overrides

```bash
# Fixed seed (required for reproducibility)
train.random_seed=42

# Offline WandB (compute nodes have no internet)
logging.wandb.mode=offline

# Change max epochs
train.pl_trainer.max_epochs=1000

# Batch size
data.datamodule.batch_size.train=128

# 2-GPU DDP
train.pl_trainer.devices=2
train.pl_trainer.strategy=ddp

# Single-GPU (explicit)
train.pl_trainer.devices=1
train.pl_trainer.strategy=auto        # NOT null — PL 2.6+ rejects null
```

---

## Checkpoint Output

Checkpoints are saved to:
```
/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun/<date>/<expname>/
    epoch=XXX-step=YYYY.ckpt
    hparams.yaml
    lattice_scaler.pt
    prop_scaler.pt
    .hydra/config.yaml          ← full Hydra snapshot
```

Top-k=1 (only best epoch saved). If you need more, override `train.model_checkpoints.save_top_k`.

---

## Resuming a Run

Hydra detects an existing `<date>/<expname>/` directory and resumes from the last checkpoint.
Use a different `expname` to start a fresh run.

---

## Existing Checkpoints (skip training)

| Dataset | Expname | Epochs | Path |
|---|---|---|---|
| mp_20 | test_py312 | 894 | `hydra/singlerun/2026-01-12/test_py312/` |
| alex_2d | alex2d_py312_2gpu | 824 | `hydra/singlerun/2026-01-14/alex2d_py312_2gpu/` |
