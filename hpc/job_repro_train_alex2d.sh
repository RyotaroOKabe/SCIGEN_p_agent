#!/bin/bash
# job_repro_train_alex2d.sh — 2-GPU training on alex_2d dataset (main experiment)
#
# Submit: cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
#         mkdir -p slurm && sbatch hpc/job_repro_train_alex2d.sh
#
#SBATCH -J SCIGEN_REPRO_TRAIN_ALEX2D
#SBATCH -A m4768
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH -o ./slurm/repro_train_alex2d_%j.log
#SBATCH -e ./slurm/repro_train_alex2d_%j.err

# ── Performance tuning ────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1

# Disable SLURM env so Lightning spawns sub-processes (not SLURM launcher)
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

# NCCL tuning for Perlmutter Infiniband
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=2

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
echo "GPUs:     $(python -c 'import torch; print(torch.cuda.device_count())')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── Training ──────────────────────────────────────────────────────────────────
EXPNAME=repro_alex2d_2gpu
echo "Starting training: expname=$EXPNAME  dataset=alex_2d  2 GPUs (DDP)"
echo ""

python scigen/run.py \
    data=alex_2d \
    model=diffusion_w_type \
    expname="$EXPNAME" \
    train.pl_trainer.devices=2 \
    train.pl_trainer.strategy=ddp \
    train.random_seed=42

EXIT=$?

echo ""
echo "Training finished at: $(date)"
echo "Exit status: $EXIT"
echo ""
echo "=== Disk usage ==="
du -sh "$REPO"

exit $EXIT
