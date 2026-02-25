#!/bin/bash
#SBATCH -J SCIGEN_GEN_STOP          # Job name
#SBATCH -A m4768                    # NERSC project allocation
#SBATCH -C gpu                      # GPU partition
#SBATCH -q premium                  # Queue
#SBATCH -t 24:00:00                 # Walltime: 1 run × 5 stop_ratios × 1000 structs each
#SBATCH -N 1                        # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o ./slurm/gen_stop_%j.log
#SBATCH -e ./slurm/gen_stop_%j.err

# --- ENVIRONMENT ---
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent:${PYTHONPATH:-}"

# --- SETUP ---
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent

module purge
module load PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia80

source /pscratch/sd/r/ryotaro/anaconda3/bin/activate \
    /pscratch/sd/r/ryotaro/anaconda3/envs/scigen_py312

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "Start:     $(date)"
echo "Python:    $(python --version)"
echo "PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:      $(python -c 'import torch; print(torch.cuda.is_available())')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# --- PRE-FLIGHT CHECK ---
# gen_mul_stop.py calls script/generation_stop.py which must exist first.
# If missing, create it (copy generation.py and add --stop_ratio support) or abort.
if [ ! -f "script/generation_stop.py" ]; then
    echo "ERROR: script/generation_stop.py not found."
    echo "This script is required by gen_mul_stop.py but has not yet been created."
    echo "See .claude/commands/debug.md for instructions on how to create it."
    exit 1
fi

# --- GENERATION ---
# Reads: config_scigen.py  →  hydra_dir, job_dir  →  model_path
# Runs:  sc_list × num_run × stop_ratios  (stop-ratio variant, mp_20 model)
# Labels: n88s_<sc><N>_s<ratio>_<tag>.pt  in the checkpoint directory
echo "Starting stop-ratio generation (mp_20, pyrochlore natm=88): $(date)"
python scripts/generation/gen_mul_stop.py

EXIT_STATUS=$?
echo ""
echo "Finished: $(date)"
echo "Exit status: $EXIT_STATUS"
echo ""
echo "=== Disk Usage ==="
du -sh .
exit $EXIT_STATUS
