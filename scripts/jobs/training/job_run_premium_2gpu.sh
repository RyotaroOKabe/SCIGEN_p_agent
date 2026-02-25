#!/bin/bash
#SBATCH -J SCIGEN_PY312_2GPU    # Job name
#SBATCH -A m4768                     # NERSC project allocation
#SBATCH -C gpu                       # Use Perlmutter GPU nodes
#SBATCH -q premium                   # Use the 'premium' queue for high-priority jobs
#SBATCH -t 24:00:00                  # Wall-time limit (24 hours for training)
#SBATCH -N 1                         # Request 1 node
#SBATCH --ntasks-per-node=1          # 1 task - PyTorch Lightning will spawn 2 processes
#SBATCH --gpus-per-task=2            # 2 GPUs per task (PyTorch Lightning will distribute them)
#SBATCH -o ./slurm/scigen_py312_2gpu_%j.log
#SBATCH -e ./slurm/scigen_py312_2gpu_%j.err

# --- PERFORMANCE TUNING ---
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1  # Prevent ~/.local/site-packages from shadowing conda env
# Disable SLURM environment auto-detection to allow Lightning to spawn processes
# Lightning will use subprocess launcher instead of SLURM launcher
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE
# Set NCCL environment variables for better multi-GPU performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=2

# --- SETUP PHASE ---
# Change to the project directory
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent

# Load necessary modules
module purge
module load PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia80

# Activate conda environment
source /pscratch/sd/r/ryotaro/anaconda3/bin/activate /pscratch/sd/r/ryotaro/anaconda3/envs/scigen_py312

# Print environment info
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU devices:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# --- EXECUTION PHASE ---
echo "Starting SCIGEN 2-GPU training at: $(date)"
echo ""
echo "Configuration options:"
echo "  Option 1: Use command-line overrides (current)"
echo "  Option 2: Use config file: train=multi_gpu (modify devices=2)"
echo ""
echo "Command: python scigen/run.py data=mp_20 model=diffusion_w_type expname=test_py312_2gpu train.pl_trainer.devices=2 train.pl_trainer.strategy=ddp"
echo ""

# Run the training script with 2-GPU configuration
# With --ntasks-per-node=1 and --gpus-per-task=2, PyTorch Lightning will
# automatically spawn 2 processes (one per GPU) and coordinate them with DDP
python scigen/run.py \
    data=mp_20 \
    model=diffusion_w_type \
    expname=test_py312_2gpu \
    train.pl_trainer.devices=2 \
    train.pl_trainer.strategy=ddp

# Option 2: Alternative - use config file (need to create train=multi_gpu_2gpu.yaml):
# python scigen/run.py \
#     data=mp_20 \
#     model=diffusion_w_type \
#     expname=test_py312_2gpu \
#     train=multi_gpu_2gpu

# Capture exit status
EXIT_STATUS=$?

echo ""
echo "SCIGEN 2-GPU training finished at: $(date)"
echo "Exit status: $EXIT_STATUS"

# --- Data Collection ---
echo ""
echo "=== Disk Usage ==="
du -sh .

exit $EXIT_STATUS




