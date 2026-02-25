# Command: Generate

Generate novel crystal structures from a trained checkpoint.

---

## Environment

```bash
conda activate scigen_py312
export PYTHONNOUSERSITE=1
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
```

---

## Quick Interactive Generation (small batch)

```bash
MODEL_PATH=/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun/2026-01-14/alex2d_py312_2gpu

python script/generation.py \
    --model_path  $MODEL_PATH \
    --dataset     alex_2d \
    --label       sc_kag100_test \
    --sc          kag \
    --batch_size  20 \
    --num_batches_to_samples 5 \
    --natm_range  1 12 \
    --known_species Mn Fe Co Ni \
    --frac_z      0.5 \
    --t_mask      True
```

Output: `$MODEL_PATH/eval_gen_sc_kag100_test.pt`

---

## Full Batch via SLURM (1000 structures)

```bash
sbatch hpc/job_repro_generate.sh
```

Uses: checkpoint `2026-01-14/alex2d_py312_2gpu`, kagome constraint, 50×20 batches.

---

## Element-Constrained Generation (mp_20)

Edit `config_scigen.py` first:
```python
hydra_dir = '/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun'
job_dir   = '2026-01-12/test_py312'
out_name  = 'sc_elem1000_000'
```

Then:
```bash
sbatch job_gen_premium_mp20.sh    # runs scripts/generation/gen_mul_elem.py
```

---

## Multi-Run Batch Orchestration

```bash
# Multiple structural constraints, multiple runs
python scripts/generation/gen_mul.py         # mp_20 model
python scripts/generation/gen_mul_2d.py      # alex_2d model (if it exists)
python scripts/generation/gen_mul_elem.py    # element-constrained
```

These scripts read from `config_scigen.py` / `config_scigen_2d.py`.
Always run from project root — they import `config_scigen` without a package prefix.

---

## Key Arguments

| Arg | Description | Example |
|---|---|---|
| `--model_path` | Path to Hydra run dir (contains *.ckpt) | `hydra/singlerun/2026-01-14/alex2d_py312_2gpu` |
| `--dataset` | Dataset name | `alex_2d` or `mp_20` |
| `--label` | Output label (appended to `eval_gen_<label>.pt`) | `sc_kag1000_000` |
| `--sc` | Structural constraint code | `kag`, `lieb`, `hon`, `sqr`, `van`, … |
| `--batch_size` | Structures per batch | `50` |
| `--num_batches_to_samples` | Number of batches | `20` → 1000 total |
| `--natm_range` | Min/max atoms in unit cell | `1 12` |
| `--known_species` | Element whitelist | `Mn Fe Co Ni Ru Nd Gd Tb Dy Yb Cu` |
| `--frac_z` | Z-axis fraction for 2D mask | `0.5` |
| `--t_mask` | Enable atom-type masking | `True` |
| `--step_lr` | Langevin step size (-1 = auto) | `-1` |

## Structural Constraint Codes

| Code | Lattice | Recommended `natm_range` |
|---|---|---|
| `kag` | Kagome | `12 12` |
| `lieb` | Lieb | `1 12` |
| `hon` | Honeycomb | `1 8` |
| `sqr` | Square | `1 4` |
| `tri` | Triangle | `1 4` |
| `pyc` | Pyrochlore | `24 24` |
| `hkg` | Hyper-kagome | `18 30` |
| `van` | Vanilla (none) | `1 20` |
