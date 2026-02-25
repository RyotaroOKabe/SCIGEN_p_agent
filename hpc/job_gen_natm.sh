#!/bin/bash
#SBATCH -J SCIGEN_GEN_NATM          # Job name
#SBATCH -A m4768                    # NERSC project allocation
#SBATCH -C gpu                      # GPU partition
#SBATCH -q premium                  # Queue
#SBATCH -t 24:00:00                 # Walltime: 5 runs × 1000 structs + save_cif=True
#SBATCH -N 1                        # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o ./slurm/gen_natm_%j.log
#SBATCH -e ./slurm/gen_natm_%j.err

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

# --- GENERATION ---
# Reads: config_scigen.py  →  hydra_dir, job_dir  →  model_path
# Runs:  sc_list × num_run generations  (fixed natm=88, save_cif=True)
# Output: eval_gen_n88_<sc><N>_<tag>.pt + cif_n88_<sc><N>_<tag>/  in checkpoint dir
echo "Starting natm-constrained generation (natm=88, save_cif=True): $(date)"
python scripts/generation/gen_mul_natm.py

EXIT_STATUS=$?
echo ""
echo "Finished: $(date)"
echo "Exit status: $EXIT_STATUS"
echo ""
echo "=== Disk Usage ==="
du -sh .
exit $EXIT_STATUS
