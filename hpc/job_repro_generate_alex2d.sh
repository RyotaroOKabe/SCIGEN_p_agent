#!/bin/bash
# job_repro_generate.sh — crystal generation using trained alex_2d checkpoint
#
# Uses the existing best checkpoint (epoch=824, alex2d_py312_2gpu).
# To use a freshly trained checkpoint, update MODEL_PATH below.
#
# Submit: cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
#         mkdir -p slurm && sbatch hpc/job_repro_generate.sh
#
#SBATCH -J SCIGEN_REPRO_GEN
#SBATCH -A m4768
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o ./slurm/repro_generate_%j.log
#SBATCH -e ./slurm/repro_generate_%j.err

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
# Best existing checkpoint — update this if you trained a new model
MODEL_PATH=/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun/2026-02-24/repro_mp20_1gpu

DATASET=alex_2d
BATCH_SIZE=50
NUM_BATCHES=20          # total = BATCH_SIZE * NUM_BATCHES = 1000 structures
SC=kag                  # structural constraint: kagome lattice
NATM_MIN=1
NATM_MAX=12
LABEL=repro_kag1000_000
KNOWN_SPECIES="Mn Fe Co Ni Ru Nd Gd Tb Dy Yb Cu"
FRAC_Z=0.5
T_MASK=True
SEED=42

# ── Environment info ──────────────────────────────────────────────────────────
echo "=== Job Info ==="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Start:        $(date)"
echo "MODEL_PATH:   $MODEL_PATH"
echo "Label:        $LABEL"
echo "SC:           $SC"
echo "Total mats:   $((BATCH_SIZE * NUM_BATCHES))"
echo ""

# Verify checkpoint exists
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH not found: $MODEL_PATH"
    echo "Run training first or update MODEL_PATH in this script."
    exit 1
fi
CKPT=$(ls "$MODEL_PATH"/*.ckpt 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
    echo "ERROR: No .ckpt file found in $MODEL_PATH"
    exit 1
fi
echo "Using checkpoint: $CKPT"
echo ""

# ── Generation ────────────────────────────────────────────────────────────────
echo "Starting generation at: $(date)"

python script/generation.py \
    --model_path  "$MODEL_PATH" \
    --dataset     "$DATASET" \
    --label       "$LABEL" \
    --sc          $SC \
    --batch_size  $BATCH_SIZE \
    --num_batches_to_samples $NUM_BATCHES \
    --natm_range  $NATM_MIN $NATM_MAX \
    --known_species $KNOWN_SPECIES \
    --frac_z      $FRAC_Z \
    --t_mask      $T_MASK

EXIT=$?

echo ""
echo "Generation finished at: $(date)"
echo "Exit status: $EXIT"
echo "Output: $MODEL_PATH/eval_gen_${LABEL}.pt"

exit $EXIT
