# Project Rules (Hard Constraints)

## 1. Environment

- Always activate with `conda activate scigen_py312` (logical alias: `scigen_p_agent`).
- Always set `export PYTHONNOUSERSITE=1` — without it, `~/.local/lib/python3.12/site-packages/`
  shadows the conda env and injects wrong package versions (confirmed: torch 2.5.1 vs conda version).
- Always `cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent` before running any script.
  `scigen` is NOT pip-installed; scripts rely on `sys.path.append('.')` from the project root.

## 2. SLURM / HPC

- Submit jobs only via scripts in `hpc/` (canonical) or root-level `job_*.sh` (legacy).
- Never embed API keys or credentials into job scripts or log output.
- All SLURM scripts must export `PYTHONNOUSERSITE=1` — already applied, do not remove.
- Use `premium` queue (`-q premium`) with account `m4768`.
- Log files go to `./slurm/` — never redirect them elsewhere.

## 3. Secrets & Version Control

- NEVER commit `.env`, API keys, or checkpoint files to git.
- `.env` at project root holds `PROJECT_ROOT`, `HYDRA_JOBS`, `WANDB_DIR` — already configured.
- Do NOT commit large binaries (`*.ckpt`, `*.pt`, `*.torch` GNN models) unless explicitly asked.

## 4. Configuration Files

- `config_scigen.py` — for mp_20 generation/screening paths; edit `hydra_dir`, `job_dir`.
- `config_scigen_2d.py` — for alex_2d generation/screening paths; same fields.
- These are Python modules imported at runtime (not Hydra configs). They live in project root.
- Do NOT move them to a subdirectory — scripts import them without a package prefix.

## 5. Code Integrity

- Do NOT modify `scigen/` source without understanding the Hydra/PL wiring.
- Do NOT modify `archive/` — these are frozen historical copies.
- Do NOT use `--force` git operations on any shared branch.
- Run the smoke test (`bash run_repro.sh --stage smoke`) after any change to `scigen/` or `conf/`.

## 6. Naming & Directories

- Progress notes: `progress/yyyy-mm-dd_<slug>.md` (follow if `progress/` dir exists).
- Do NOT create status files at the project root; prefer `progress/` or inline in CLAUDE.md.
- Canonical output path (new runs): `/pscratch/sd/r/ryotaro/outputs/SCIGEN_p_agent/<run_id>/`.
- Canonical dataset path: `/pscratch/sd/r/ryotaro/datasets/SCIGEN_p_agent/`.
  Current datasets are in `data/` (project-local); either is acceptable until symlinks are set up.

## 7. Reproducibility

- Every training run must use `train.random_seed=42` unless intentionally exploring variance.
- Use `logging.wandb.mode=offline` on compute nodes (no internet). Sync later if needed.
- Hydra saves full config snapshots in `<run_dir>/.hydra/` — do not delete them.

## 8. Known Incompatibilities (PL 2.6+)

- `strategy: null` is rejected → use `strategy: auto` (already in `conf/train/default.yaml`).
- `fast_dev_run=True` breaks `trainer.test(ckpt_path="best")` → use `max_epochs=1 +limit_*_batches=2`.
- Smoke test command in `run_repro.sh` already uses the correct override — do not change it.
