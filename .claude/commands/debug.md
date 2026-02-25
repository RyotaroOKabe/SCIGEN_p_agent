# Command: Debug

Troubleshooting guide for common SCIGEN_p_agent failures.

---

## First Steps

1. Check the SLURM `.err` file: `cat slurm/<name>_<JOBID>.err`
2. Check the `.log` file for early failure signs: `head -50 slurm/<name>_<JOBID>.log`
3. Verify Python + torch versions in the log header:
   ```
   Python version: 3.12.12
   PyTorch version: 2.5.1      ← must NOT be a ~/.local version
   CUDA available: True
   ```

---

## Error Catalog

### `ModuleNotFoundError: No module named 'config_scigen'`

**Cause**: `scripts/generation/*.py` imports `config_scigen` from project root, but Python's
`sys.path` doesn't include the root when running from a subdirectory.

**Fix (interactive)**:
```bash
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
PYTHONPATH=. python scripts/generation/gen_mul_elem.py
```

**Fix (SLURM)**: `job_gen_premium_mp20.sh` already exports `PYTHONPATH=<root>:...`. Verify it's there:
```bash
grep PYTHONPATH job_gen_premium_mp20.sh
```

---

### Wrong torch version in log (`2.5.1` from `~/.local`)

**Cause**: Missing `export PYTHONNOUSERSITE=1` in the SLURM script.

**Fix**: All scripts in `hpc/` and root-level `job_*.sh` already have this line. If you added a new
script and forgot it, add: `export PYTHONNOUSERSITE=1` after `export OMP_NUM_THREADS=1`.

---

### `ValueError: You selected an invalid strategy name: strategy=None`

**Cause**: PL 2.6+ rejects `strategy: null` (was valid in PL 2.4).

**Fix applied**: `conf/train/default.yaml` → `strategy: auto`. Do not override back to `null`.

---

### `ValueError: You cannot execute .test(ckpt_path="best") with fast_dev_run=True`

**Cause**: `fast_dev_run=True` skips checkpoint saving, so `ckpt_path="best"` has nothing to load.

**Fix**: Use 1-epoch limited training instead:
```bash
python scigen/run.py ... \
    train.pl_trainer.max_epochs=1 \
    +train.pl_trainer.limit_train_batches=2 \
    +train.pl_trainer.limit_val_batches=2 \
    +train.pl_trainer.limit_test_batches=2 \
    logging.val_check_interval=1
```

---

### `ConfigCompositionException: Could not override 'train.pl_trainer.limit_train_batches'`

**Cause**: Hydra CLI override for a key not already in the config requires `+` prefix.

**Fix**: Use `+train.pl_trainer.limit_train_batches=2` (prefix with `+` to append new keys).

---

### `FileNotFoundError: _sym.pt` during dataset load

**Cause**: PyTorch graph cache files are missing.

**Fix**: The `_sym.pt` files already exist in `data/alex_2d/` and `data/mp_20/`. Verify:
```bash
ls data/alex_2d/*.pt
ls data/mp_20/*.pt
```
If missing, rebuild with `scigen/data/preprocess.py` (may take hours).

---

### Hydra output dir collision / accidental resume

**Cause**: Two runs with the same `expname` on the same date share a Hydra output dir.
The second run detects the first run's checkpoint and resumes from it.

**Fix**: Use a unique `expname`:
```bash
python scigen/run.py ... expname=mp20_seed99   # distinct from existing
```

---

### WandB connectivity error on compute nodes

**Cause**: Compute nodes have no outbound internet.

**Fix**:
```bash
python scigen/run.py ... logging.wandb.mode=offline
# Sync later from login node:
wandb sync /pscratch/sd/r/ryotaro/data/generative/wandb/offline-run-*/
```

---

### "No Voronoi neighbors found" during dataset loading

**Cause**: Graph cache (`_sym.pt`) is missing and dataset is being rebuilt from CSV.
The CrystalNN neighbor finder sometimes fails on unusual structures.

**Fix**: Do not delete the existing `_sym.pt` files. If you must rebuild:
- Increase neighbor cutoff in `conf/data/*.yaml`
- Or filter the problematic structures from the CSV.

---

### `script/generation_stop.py` not found

**Cause**: `scripts/generation/gen_mul_stop*.py` calls this script, but it was never committed.
It does NOT exist in `archive/260119/script/` either.

**Fix**: The stop-ratio generation scripts (`gen_mul_stop.py`, `gen_mul_stop_2d.py`) are
currently non-functional. Use `gen_mul_elem.py` or `script/generation.py` directly.
To enable stop-ratio generation: create `script/generation_stop.py` by copying
`script/generation.py` and adding `--stop_ratio` support via `diffusion_w_type_stop.py`.
