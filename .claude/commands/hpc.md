# Command: HPC

SLURM job management for SCIGEN_p_agent on Perlmutter (NERSC).

---

## Environment

```bash
conda activate scigen_py312
export PYTHONNOUSERSITE=1    # always — prevents user-site package pollution
```

---

## Resource Configuration (Perlmutter)

| Purpose | Queue | Nodes | GPUs | Walltime |
|---|---|---|---|---|
| Training (1-GPU) | `premium` | 1 | 1 | 24 h |
| Training (2-GPU DDP) | `premium` | 1 | 2 | 24 h |
| Generation | `premium` | 1 | 1 | 24 h |
| Screening | `premium` | 1 | 1 | 4 h |
| Debug / smoke | login node | — | — | — |

All jobs: account `m4768`, constraint `gpu`.

---

## Canonical SLURM Scripts (`hpc/` — reproducible)

```bash
sbatch hpc/job_repro_train.sh           # mp_20, 1 GPU, seed=42
sbatch hpc/job_repro_train_alex2d.sh    # alex_2d, 2 GPU, seed=42
sbatch hpc/job_repro_generate.sh        # generation, 1000 kagome structs
sbatch hpc/job_repro_screen.sh          # SMACT+GNN screening
```

## Legacy Root-Level Scripts

```bash
sbatch job_run_premium_mp20.sh           # mp_20, 1 GPU
sbatch job_run_premium_mp20_2gpu.sh      # mp_20, 2 GPU DDP
sbatch job_run_premium_alex2d_2gpu.sh    # alex_2d, 2 GPU DDP
sbatch job_gen_premium_mp20.sh           # element-constrained generation
```

---

## Monitoring

```bash
squeue -u $USER                          # all your running jobs
squeue -j <JOBID>                        # specific job
tail -f slurm/<name>_<JOBID>.log         # live stdout
cat  slurm/<name>_<JOBID>.err            # stderr / errors
```

---

## Log File Locations

```
slurm/repro_train_<JOBID>.log      # hpc/job_repro_train.sh stdout
slurm/repro_train_<JOBID>.err      # stderr
slurm/scigen_py312_2gpu_<JOBID>.log  # legacy 2-GPU scripts
slurm/scigen_gen_<JOBID>.log       # generation jobs
slurm/scigen_screen_<JOBID>.log    # screening jobs
```

---

## Job Output Locations

After training:
```
hydra/singlerun/<date>/<expname>/
    epoch=XXX-step=YYYY.ckpt
    hparams.yaml
    lattice_scaler.pt
    prop_scaler.pt
    .hydra/config.yaml
```

After generation:
```
hydra/singlerun/<date>/<expname>/
    eval_gen_<label>.pt
```

After screening:
```
hydra/singlerun/<date>/<expname>/
    eval_<label>.pt
    cif_<label>/         ← CIF files (if gen_cif=True)
```

---

## Cancelling a Job

```bash
scancel <JOBID>
```

---

## Checking Available GPUs

```bash
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
nvidia-smi                          # on compute node (salloc or srun --pty)
```
