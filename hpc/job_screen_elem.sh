#!/bin/bash
#SBATCH -J SCIGEN_SCREEN_ELEM       # Job name
#SBATCH -A m4768                    # NERSC project allocation
#SBATCH -C gpu                      # GPU partition (GNN eval needs GPU)
#SBATCH -q premium                  # Queue
#SBATCH -t 04:00:00                 # Walltime: 5 elements × 1 run × 500 structs
#SBATCH -N 1                        # 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o ./slurm/screen_elem_%j.log
#SBATCH -e ./slurm/screen_elem_%j.err

# --- ENVIRONMENT ---
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
# config_scigen.py lives in project root; add it to PYTHONPATH
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

# --- SCREENING ---
# Reads: config_scigen.py  →  hydra_dir, job_dir, stab_pred_name_A/B, mag_pred_name
# Runs:  sc_list × atom_list × num_run  (SMACT + GNN stability screening, gen_cif=True)
# Input: eval_gen_sc_<sc><element><N>_<tag>.pt  must exist in the checkpoint directory
# Output: eval_<label>.pt + cif_<label>/  in the checkpoint directory
# Labels match gen_mul_elem.py: sc_liebMn500_000, sc_liebFe500_000, ...
echo "Starting element-constrained screening (lieb, Mn/Fe/Co/Ni/Cu, 1 run × 500 structs): $(date)"
python scripts/screening/screen_mul_elem.py

EXIT_STATUS=$?
echo ""
echo "Finished: $(date)"
echo "Exit status: $EXIT_STATUS"
echo ""
echo "=== Disk Usage ==="
du -sh .
exit $EXIT_STATUS
