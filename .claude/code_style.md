# Code Style: SCIGEN_p_agent

## Python Standards
- **Formatter**: Black (line length 88).
- **Linter**: ruff (equivalent to flake8).
- **Type Hinting**: Use for all new public functions. Prefer `list[str]` / `dict[str, int]`
  (Python 3.10+ syntax) over `typing.List` / `typing.Dict`.
- **Docstrings**: Google Style for public functions; one-liner for simple internals.

## Logging
- Use `import logging; logger = logging.getLogger(__name__)` at module level.
- Do NOT use bare `print()` for progress in new code; use `logging.info()`.
- Always log the absolute path of output files so runs are auditable.
- Exception: existing `script/*.py` scripts use `print()` — do not refactor unless asked.

## Imports
- Group 1: Standard library (`os`, `sys`, `pathlib`, `logging`).
- Group 2: Scientific third-party (`numpy`, `pandas`, `torch`, `pymatgen`).
- Group 3: ML / Graph (`torch_geometric`, `pytorch_lightning`).
- Group 4: Local / project (`scigen.*`, `eval_utils`, `gen_utils`, `config_scigen`).

## CLI Conventions
- Every new runnable script must accept `--seed INT` and print its output directory at start.
- Use `argparse` (not `click`) for consistency with existing `script/*.py` scripts.
- Boolean CLI args use `type=lambda x: x.lower() == 'true'` (matches existing pattern).

## Reproducibility
- Set `torch.manual_seed(seed)` and `random.seed(seed)` at script start.
- Log git commit hash when available:
  ```python
  import subprocess
  try:
      sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
  except Exception:
      sha = "unknown"
  ```
- Save `config.yaml` and `metrics.json` to every new experiment directory.

## File Structure
- `scigen/`: core model library — do not add utility scripts here.
- `script/`: generation and evaluation entrypoints (flat, single scripts).
- `scripts/`: orchestrator loops that call `script/` internals in batch.
- `hpc/`: reproducible SLURM scripts.
- New utility code → prefer a new module in `script/` or a helper in `scripts/`.
