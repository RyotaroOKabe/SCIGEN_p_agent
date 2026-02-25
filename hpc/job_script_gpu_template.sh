#!/bin/bash
#SBATCH -J TEMPLATE_JOB
#SBATCH -A m4768
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH -o ./slurm/template_job_%j.log
#SBATCH -e ./slurm/template_job_%j.err

mkdir -p ./slurm

# --- PERFORMANCE TUNING ---
export OMP_NUM_THREADS=1

# Avoid user-site packages (~/.local/...)
export PYTHONNOUSERSITE=1

# Disable SLURM env auto-detection (Lightning subprocess launcher)
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

# NCCL tuning (Perlmutter)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=2

# --- Activate env ---
source ~/.bashrc
conda activate mat_gen_all

which python
python --version

python - << 'PY'
import sys, site
print("sys.executable:", sys.executable)
print("user site:", site.getusersitepackages())
PY

# --- Run command (example) ---
# python -m src.surrogate.train --config configs/surrogate_v1.yaml --seed 0
