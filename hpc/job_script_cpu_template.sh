#!/bin/bash
#SBATCH -J TEMPLATE_JOB
#SBATCH -A m4768
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -o ./slurm/template_job_%j.log
#SBATCH -e ./slurm/template_job_%j.err

mkdir -p ./slurm

# Avoid user-site packages (~/.local/...)
export PYTHONNOUSERSITE=1

# Threading controls (good defaults)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Activate env
source ~/.bashrc
conda activate mat_gen_all

which python
python --version

python - << 'PY'
import sys, site
print("sys.executable:", sys.executable)
print("user site:", site.getusersitepackages())
PY

# Example:
# python -m src.data.download_mp --out_raw data/raw/mp_dump.parquet
# python -m src.data.make_splits --in_processed data/processed/mp_magmom_dataset.parquet
