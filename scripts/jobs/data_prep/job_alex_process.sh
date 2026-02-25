#!/bin/bash
#SBATCH -J ALEX_PROCESS_DATA      # Job name
#SBATCH -A m4768                  # NERSC project allocation
#SBATCH -C cpu                    # Use CPU nodes (no GPU needed for data processing)
#SBATCH -q regular                # Use the 'regular' queue
#SBATCH -t 6:00:00                # Wall-time limit (6 hours should be enough)
#SBATCH -N 1                      # Request 1 node
#SBATCH --ntasks-per-node=1       # 1 task (Python script)
#SBATCH --cpus-per-task=8         # Request 8 CPUs for parallel processing
#SBATCH --mem=64G                 # Request 64GB memory for large CSV files
#SBATCH -o ./slurm/alex_process_%j.log
#SBATCH -e ./slurm/alex_process_%j.err

# --- PERFORMANCE TUNING ---
export OMP_NUM_THREADS=8

# --- SETUP PHASE ---
# Change to the project directory
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent

# Load necessary modules
module purge
module load PrgEnv-gnu
# Note: No GPU modules needed for data processing

# Activate conda environment
source /pscratch/sd/r/ryotaro/anaconda3/bin/activate /pscratch/sd/r/ryotaro/anaconda3/envs/scigen_py312

# Set matplotlib to use non-interactive backend (required for headless execution)
export MPLBACKEND=Agg

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
echo "Matplotlib backend: $MPLBACKEND"
echo ""

# Check for required Python packages
echo "=== Checking Required Python Packages ==="
python -c "import pandas; import numpy; import sklearn; import matplotlib; print('✓ All required packages available')" || { echo "ERROR: Missing required packages"; exit 1; }
echo ""

# Check if input CSV files exist
echo "=== Checking Input Data ==="
DATA_DIR="data/alex_2d"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory $DATA_DIR does not exist!"
    exit 1
fi

CSV_COUNT=$(ls -1 ${DATA_DIR}/alexandria_2d_*.csv 2>/dev/null | wc -l)
if [ "$CSV_COUNT" -eq 0 ]; then
    echo "ERROR: No CSV files found in $DATA_DIR"
    echo "Please run alex_load.py first to generate CSV files."
    exit 1
fi
echo "Found $CSV_COUNT CSV file(s) in $DATA_DIR"
ls -lh ${DATA_DIR}/alexandria_2d_*.csv
echo ""

# --- EXECUTION PHASE ---
echo "Starting Alexandria data processing at: $(date)"
echo "Script: data_prep/alex_process.py"
echo "Working directory: data_prep/"
echo ""

# Change to data_prep directory (script uses relative imports)
cd data_prep

# Run the data processing script
# The script will:
# 1. Load CSV files from data/alex_2d/
# 2. Filter data based on thresholds
# 3. Split into train/val/test sets
# 4. Save processed CSV files
# 5. Generate visualization plots
python alex_process.py

# Capture exit status
EXIT_STATUS=$?

# Return to project root
cd ..

echo ""
echo "Alexandria data processing finished at: $(date)"
echo "Exit status: $EXIT_STATUS"

# --- Data Collection ---
echo ""
echo "=== Processing Results ==="
if [ -f "${DATA_DIR}/train.csv" ]; then
    echo "✓ train.csv created"
    echo "  Size: $(du -h ${DATA_DIR}/train.csv | cut -f1)"
    echo "  Rows: $(wc -l < ${DATA_DIR}/train.csv)"
fi
if [ -f "${DATA_DIR}/val.csv" ]; then
    echo "✓ val.csv created"
    echo "  Size: $(du -h ${DATA_DIR}/val.csv | cut -f1)"
    echo "  Rows: $(wc -l < ${DATA_DIR}/val.csv)"
fi
if [ -f "${DATA_DIR}/test.csv" ]; then
    echo "✓ test.csv created"
    echo "  Size: $(du -h ${DATA_DIR}/test.csv | cut -f1)"
    echo "  Rows: $(wc -l < ${DATA_DIR}/test.csv)"
fi
if [ -f "${DATA_DIR}/data_config.json" ]; then
    echo "✓ data_config.json created"
fi

echo ""
echo "=== Visualization Files ==="
if [ -d "$DATA_DIR" ]; then
    PLOT_COUNT=$(ls -1 ${DATA_DIR}/*.png 2>/dev/null | wc -l)
    if [ "$PLOT_COUNT" -gt 0 ]; then
        echo "Generated $PLOT_COUNT plot file(s):"
        ls -lh ${DATA_DIR}/*.png
    else
        echo "No plot files found (may be normal if plotting failed)"
    fi
fi

echo ""
echo "=== Disk Usage ==="
du -sh .
if [ -d "$DATA_DIR" ]; then
    echo "Alexandria 2D data directory size:"
    du -sh ${DATA_DIR}
fi

exit $EXIT_STATUS

