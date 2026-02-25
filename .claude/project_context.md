# Project Context: SCIGEN_p_agent

## Overview
**Structure-Constrained Crystal Generation with Diffusion Models**

SCIGEN trains a DiffCSP-style score-based diffusion model (CSPNet) on crystal databases
and generates novel structures subject to geometric lattice constraints (kagome, lieb,
honeycomb, pyrochlore, hyper-kagome, …). Generated candidates are then filtered by
SMACT charge-balance validity and GNN-based stability/magnetic-moment predictors.

## Datasets

| Name | Structures | Split | Source | Location |
|---|---|---|---|---|
| `alex_2d` | 68,826 (filtered from 137,833) | 55k/6.8k/6.8k | Alexandria 2D PBE | `data/alex_2d/` |
| `mp_20` | ~30,000 | 27k/3k/3k | Materials Project | `data/mp_20/` |

Filters: `e_above_hull < 0.3 eV/atom`, `natm < 41`, `formation_energy/atom < 2.0 eV`.

Preprocessed graph caches (`*_sym.pt`) already exist — **no download or rebuild needed**.

## Model Architecture
- **CSPNet**: equivariant transformer over periodic crystal graphs.
- **Diffusion**: joint score model over fractional coordinates, lattice vectors, and atom types.
- **Constraint mechanism**: `--sc` flag conditions the denoising trajectory on a target lattice
  symmetry (kagome = `kag`, lieb, honeycomb = `hon`, square = `sqr`, pyrochlore = `pyc`, …).
- **Type masking**: `--t_mask True` restricts atom-type sampling to `--known_species`.

## Trained Checkpoints

| Dataset | Expname | Epochs | Path |
|---|---|---|---|
| alex_2d | alex2d_py312_2gpu | 824 | `hydra/singlerun/2026-01-14/alex2d_py312_2gpu/` |
| mp_20 | test_py312 | 894 | `hydra/singlerun/2026-01-12/test_py312/` |

These are usable immediately — skip training if only generating or screening.

## Pipeline Architecture

```
Train (scigen/run.py, Hydra + PL)
   └── checkpoint → hydra/singlerun/<date>/<expname>/
          ↓
Generate (script/generation.py)
   └── eval_gen_<label>.pt in checkpoint dir
          ↓
Screen (script/eval_screen.py)
   └── eval_<label>.pt + cif_<label>/ in checkpoint dir
```

## GNN Evaluators (pre-trained, already in repo)

| Model | File | Purpose |
|---|---|---|
| stab_A | `gnn_eval/models/stab_240409-113155.torch` | Stability score (set A) |
| stab_B | `gnn_eval/models/stab_240402-111754.torch` | Stability score (set B) |
| mag | `gnn_eval/models/mag_240815-085301.torch` | Magnetic moment score |

## Structural Constraints (`--sc`)

| Code | Lattice | Typical `natm_range` |
|---|---|---|
| `kag` | Kagome | `[12, 12]` (fixed) |
| `lieb` | Lieb | `[1, 12]` |
| `hon` | Honeycomb | `[1, 8]` |
| `sqr` | Square | `[1, 4]` |
| `tri` | Triangle | `[1, 4]` |
| `pyc` | Pyrochlore | `[24, 24]` |
| `hkg` | Hyper-kagome | `[18, 30]` |
| `van` | Vanilla (unconstrained) | `[1, 20]` |

## Exclusions (What this is NOT)
- **Not a DFT validator**: No VASP/MLIP relaxation. Use `ht4matgen` for that.
- **Not interactive**: Scripts are CLI-only; no Jupyter notebooks.
- **Not dataset-agnostic**: Specific to `alex_2d` and `mp_20` preprocessed formats.

## Key External Dependencies
- `scigen/`: local library — NOT pip-installed; patched via `sys.path.append('.')`.
- `archive/`: frozen historical snapshots; do not modify.
