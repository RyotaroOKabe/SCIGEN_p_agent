#!/bin/bash
# job_repro_screen.sh — screening and evaluation of generated crystal structures
#
# Runs SMACT validity + space occupation + GNN stability filtering.
# Requires generation output from job_repro_generate.sh.
#
# IMPORTANT: Update LABEL below to match the label used in generation.
# IMPORTANT: config_scigen.py must have the correct hydra_dir and job_dir
#            pointing to the checkpoint directory that contains the .pt files.
#
# Submit: cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
#         mkdir -p slurm && sbatch hpc/job_repro_screen.sh
#
#SBATCH -J SCIGEN_REPRO_SCREEN
#SBATCH -A m4768
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o ./slurm/repro_screen_%j.log
#SBATCH -e ./slurm/repro_screen_%j.err

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

# ── Configuration ─────────────────────────────────────────────────────────────
# Must match the label used in generation
LABEL=repro_kag1000_000
GEN_CIF=True      # save CIF files for passing structures
SCREEN_MAG=False  # set True to also filter by magnetic moment

# Screening reads hydra_dir/job_dir from config_scigen.py.
# Verify it points to the right checkpoint directory before running.
MODEL_PATH=/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun/2026-01-14/alex2d_py312_2gpu

# ── Environment info ──────────────────────────────────────────────────────────
echo "=== Job Info ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Start:    $(date)"
echo "Label:    $LABEL"
echo ""

# Verify generation output exists
GEN_PT="$MODEL_PATH/eval_gen_${LABEL}.pt"
if [[ ! -f "$GEN_PT" ]]; then
    echo "ERROR: Generation output not found: $GEN_PT"
    echo "Run job_repro_generate.sh first."
    exit 1
fi
echo "Input: $GEN_PT"
echo ""

# ── Screening ─────────────────────────────────────────────────────────────────
echo "Starting screening at: $(date)"

# Note: eval_screen.py reads config_scigen.py for hydra_dir/job_dir.
# Ensure config_scigen.py has:
#   hydra_dir = '/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun'
#   job_dir   = '2026-01-14/alex2d_py312_2gpu'

python script/eval_screen.py \
    --label   "$LABEL" \
    --gen_cif "$GEN_CIF" \
    --screen_mag "$SCREEN_MAG"

EXIT=$?

echo ""
echo "Screening finished at: $(date)"
echo "Exit status: $EXIT"
echo "Results in:  $MODEL_PATH/"

exit $EXIT
