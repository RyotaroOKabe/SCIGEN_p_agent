#!/bin/bash
#SBATCH -J SCIGEN_ALEX_2D_TRAIN    # Job name
#SBATCH -A m4768                    # NERSC project allocation
#SBATCH -C gpu                      # Use Perlmutter GPU nodes
#SBATCH -q premium                  # Use the 'premium' queue for high-priority jobs
#SBATCH -t 24:00:00                 # Wall-time limit (24 hours for training)
#SBATCH -N 1                        # Request 1 node
#SBATCH --ntasks-per-node=1         # 1 task (Python script)
#SBATCH --gpus-per-task=1          # Request 1 GPU per task
#SBATCH -o ./slurm/scigen_alex_2d_%j.log
#SBATCH -e ./slurm/scigen_alex_2d_%j.err

# --- PERFORMANCE TUNING ---
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1  # Prevent ~/.local/site-packages from shadowing conda env

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
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Check if processed data files exist
echo "=== Checking Data Files ==="
DATA_DIR="data/alex_2d"
if [ ! -f "${DATA_DIR}/train.csv" ]; then
    echo "WARNING: ${DATA_DIR}/train.csv not found!"
    echo "Please run alex_process.py first to generate train/val/test splits."
    echo "You can submit: sbatch job_alex_process.sh"
    exit 1
fi
if [ ! -f "${DATA_DIR}/val.csv" ]; then
    echo "WARNING: ${DATA_DIR}/val.csv not found!"
    exit 1
fi
if [ ! -f "${DATA_DIR}/test.csv" ]; then
    echo "WARNING: ${DATA_DIR}/test.csv not found!"
    exit 1
fi
echo "✓ train.csv: $(wc -l < ${DATA_DIR}/train.csv) rows"
echo "✓ val.csv: $(wc -l < ${DATA_DIR}/val.csv) rows"
echo "✓ test.csv: $(wc -l < ${DATA_DIR}/test.csv) rows"
echo ""

# --- EXECUTION PHASE ---
echo "Starting SCIGEN training with Alexandria 2D dataset at: $(date)"
echo "Command: python scigen/run.py data=alex_2d model=diffusion_w_type expname=alex_2d_py312"
echo ""

# Run the training script
python scigen/run.py data=alex_2d model=diffusion_w_type expname=alex_2d_py312

# Capture exit status
EXIT_STATUS=$?

echo ""
echo "SCIGEN training finished at: $(date)"
echo "Exit status: $EXIT_STATUS"

# --- Data Collection ---
echo ""
echo "=== Disk Usage ==="
du -sh .

exit $EXIT_STATUS

