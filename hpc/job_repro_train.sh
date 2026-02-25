#!/bin/bash
# job_repro_train.sh — 1-GPU training on mp_20 dataset
#
# Submit: cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
#         mkdir -p slurm && sbatch hpc/job_repro_train.sh
#
#SBATCH -J SCIGEN_REPRO_TRAIN_MP20
#SBATCH -A m4768
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o ./slurm/repro_train_%j.log
#SBATCH -e ./slurm/repro_train_%j.err

# ── Performance tuning ────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1

# ── Setup ─────────────────────────────────────────────────────────────────────
REPO=/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
cd "$REPO"

module purge
module load PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia80

source /pscratch/sd/r/ryotaro/anaconda3/bin/activate \
    /pscratch/sd/r/ryotaro/anaconda3/envs/scigen_py312

# ── Environment info ──────────────────────────────────────────────────────────
echo "=== Job Info ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Start:    $(date)"
echo "CWD:      $(pwd)"
echo "Python:   $(python --version)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:     $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ── Training ──────────────────────────────────────────────────────────────────
EXPNAME=repro_mp20_1gpu
echo "Starting training: expname=$EXPNAME  dataset=mp_20  1 GPU"
echo ""

python scigen/run.py \
    data=mp_20 \
    model=diffusion_w_type \
    expname="$EXPNAME" \
    train.pl_trainer.devices=1 \
    train.pl_trainer.strategy=auto \
    train.random_seed=42

EXIT=$?

echo ""
echo "Training finished at: $(date)"
echo "Exit status: $EXIT"
echo ""
echo "=== Disk usage ==="
du -sh "$REPO"

exit $EXIT
