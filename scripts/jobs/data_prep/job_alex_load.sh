#!/bin/bash
#SBATCH -J ALEX_LOAD_DATA        # Job name
#SBATCH -A m4768                 # NERSC project allocation
#SBATCH -C cpu                   # Use CPU nodes (no GPU needed for data prep)
#SBATCH -q regular               # Use the 'regular' queue
#SBATCH -t 12:00:00              # Wall-time limit (12 hours should be enough)
#SBATCH -N 1                     # Request 1 node
#SBATCH --ntasks-per-node=1      # 1 task (Python script)
#SBATCH --cpus-per-task=8        # Request 8 CPUs for parallel processing
#SBATCH --mem=64G                # Request 64GB memory for large JSON files
#SBATCH -o ./slurm/alex_load_%j.log
#SBATCH -e ./slurm/alex_load_%j.err

# --- PERFORMANCE TUNING ---
export OMP_NUM_THREADS=8

# --- SETUP PHASE ---
# Change to the project directory
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent

# Load necessary modules
module purge
module load PrgEnv-gnu
# Note: No GPU modules needed for data preparation

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
echo "Available CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo ""

# Check for required commands
echo "=== Checking Required Tools ==="
command -v wget >/dev/null 2>&1 || { echo "ERROR: wget not found. Please load appropriate module."; exit 1; }
command -v bzip2 >/dev/null 2>&1 || { echo "ERROR: bzip2 not found. Please load appropriate module."; exit 1; }
echo "wget: $(which wget)"
echo "bzip2: $(which bzip2)"
echo ""

# --- EXECUTION PHASE ---
echo "Starting Alexandria data loading at: $(date)"
echo "Script: data_prep/alex_load.py"
echo "Working directory: data_prep/"
echo ""

# Change to data_prep directory (script uses relative imports)
cd data_prep

# Run the data loading script
# The script uses sys.path.append('../') so we need to run from data_prep/
python alex_load.py

# Capture exit status
EXIT_STATUS=$?

# Return to project root
cd ..

echo ""
echo "Alexandria data loading finished at: $(date)"
echo "Exit status: $EXIT_STATUS"

# --- Data Collection ---
echo ""
echo "=== Disk Usage ==="
du -sh .
if [ -d "data/alex_2d" ]; then
    echo "Alexandria 2D data directory size:"
    du -sh data/alex_2d
fi

exit $EXIT_STATUS

