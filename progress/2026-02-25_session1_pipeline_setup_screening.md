# Session 1: Pipeline Setup, GitHub, SLURM Scripts, and First Screening — 2026-02-25

## Summary

First full working session for SCIGEN_p_agent. Established the Claude Code
configuration, cleaned the Git history and pushed to GitHub, wrote SLURM scripts
for all generation and screening orchestrators, ran the first element-constrained
Lieb generation job (5 × 500 structures), and completed SMACT + GNN screening
after fixing two missing-package bugs and one SMACT API incompatibility.

---

## Results

### Generation job 49337090 — `job_gen_elem.sh` → `gen_mul_elem.py`

| Label | Structures | Time | Output |
|-------|-----------|------|--------|
| sc_liebMn500_000 | 500 | ~192 s | `eval_gen_sc_liebMn500_000.pt` |
| sc_liebFe500_000 | 500 | ~192 s | `eval_gen_sc_liebFe500_000.pt` |
| sc_liebCo500_000 | 500 | ~192 s | `eval_gen_sc_liebCo500_000.pt` |
| sc_liebNi500_000 | 500 | ~192 s | `eval_gen_sc_liebNi500_000.pt` |
| sc_liebCu500_000 | 500 | ~192 s | `eval_gen_sc_liebCu500_000.pt` |

- Model: `hydra/singlerun/2026-02-24/mp_20_v1`
- Settings: `sc=lieb`, `natm_range=[1,12]`, `t_mask=True`, `frac_z=0.5`
- `save_cif` failed silently (see Bugs below); CIFs regenerated manually afterward.

### Screening job 49337590 — `job_screen_elem.sh` → `screen_mul_elem.py`

First run failed (opt_einsum missing). Screening re-run interactively after fixes.

| Element | SMACT valid | Density OK | GNN-A stable | GNN-B stable (final) |
|---------|------------|-----------|-------------|---------------------|
| Mn | 321 / 500 | 321 / 321 | 123 / 321 | **25** |
| Fe | 312 / 500 | 312 / 312 | 86 / 312 | **49** |
| Co | 367 / 500 | 367 / 367 | 179 / 367 | **98** |
| Ni | 297 / 500 | 297 / 297 | 134 / 297 | **47** |
| Cu | 268 / 500 | 268 / 268 | 149 / 268 | **38** |

- GNN-A: `stab_240409-113155`; GNN-B: `stab_240402-111754`
- Filtered CIFs saved to `mp_20_v1/gen_sc_lieb{elem}500_000_filtered/`

---

## Code Changes

| File | Change |
|------|--------|
| `.claude/CLAUDE.md` | Rewritten as multi-file index; added Progress command entry |
| `.claude/commands/progress.md` | New — progress report guide and template |
| `.claude/commands/{train,generate,screen,hpc,debug}.md` | New command guides |
| `.claude/project_context.md` | New — dataset and checkpoint reference |
| `.claude/RULES.md` | New — hard constraints, SLURM policy |
| `.claude/SKILLS.md` | New — operational cheat sheet |
| `.claude/code_style.md` | New — Python style conventions |
| `.claude/settings.local.json` | New — pre-approved Bash permissions |
| `scripts/screening/screen_mul_elem.py` | New — screens gen_mul_elem.py outputs |
| `scripts/screening/screen_mul.py` | Now tracked (gitignore fix) |
| `scripts/screening/screen_mul_natm.py` | Now tracked (gitignore fix) |
| `scripts/screening/screen_mul_2d.py` | Now tracked (gitignore fix) |
| `hpc/job_gen_elem.sh` | New SLURM script |
| `hpc/job_gen_natm.sh` | New SLURM script |
| `hpc/job_gen_mul.sh` | New SLURM script |
| `hpc/job_gen_stop.sh` | New SLURM script |
| `hpc/job_gen_stop_2d.sh` | New SLURM script |
| `hpc/job_screen_mul.sh` | New SLURM script |
| `hpc/job_screen_natm.sh` | New SLURM script |
| `hpc/job_screen_2d.sh` | New SLURM script |
| `hpc/job_screen_elem.sh` | New SLURM script |
| `script/mat_utils.py` | Added `None` guard in `smact_validity` (SMACT 3.2 compat) |
| `.gitignore` | Anchored `screening/` → `/screening/`; added `!hpc/*.sh` negation |

---

## Bugs Fixed

### Bug 1: `periodictable` not in `scigen_py312`

- **Root cause**: `script/mat_utils.py` imports `periodictable` at module level.
  Package was missing from the conda env; only present in `~/.local/` (user-site).
  With `PYTHONNOUSERSITE=1` set in SLURM scripts, it was invisible.
- **Symptom**: `save_cif.py` crashed with `ModuleNotFoundError: No module named 'periodictable'`.
  The crash was swallowed by `os.system()` in the generation orchestrator, so the SLURM
  job exited 0 but no CIF files were written.
- **Fix**: `pip install --target <scigen_py312 site-packages> periodictable`

### Bug 2: `opt_einsum` not in `scigen_py312`

- **Root cause**: `gnn_eval/utils/model_class.py` imports `e3nn`, which imports
  `opt_einsum_fx`, which requires `opt_einsum`. Package was in `~/.local/` only.
  With `PYTHONNOUSERSITE=1`, invisible in SLURM.
- **Symptom**: `eval_screen.py` crashed at import time; `os.system()` in the screening
  orchestrator swallowed the error; all labels printed "done" with no output files written.
- **Fix**: `pip install opt_einsum --ignore-installed --target <scigen_py312 site-packages>`

### Bug 3: `smact_validity` crashes on He (SMACT 3.2 API)

- **Root cause**: Probabilistic atom-type decoding (`atom_type_prob=True`) occasionally
  assigns helium to a site. SMACT 3.2 returns `oxidation_states = None` for noble gases.
  The loop `for oxc in ox_combos: oxn *= len(oxc)` raises `TypeError: 'NoneType' has no len()`.
- **Symptom**: `eval_screen.py` crashed mid-run on the first structure containing He.
- **Fix**: One-line guard added to `script/mat_utils.py:463`:
  ```python
  if oxc is None:   # noble gases / elements with no SMACT oxidation states
      return False
  ```

---

## Environment Notes

Both `periodictable` and `opt_einsum` must live inside the conda env, not just
`~/.local/`, because all SLURM scripts set `PYTHONNOUSERSITE=1`.

Standard install (`pip install <pkg>`) resolves to `~/.local/bin/pip` on Perlmutter
even when a conda env is active. Always use explicit path:

```bash
/pscratch/sd/r/ryotaro/anaconda3/envs/scigen_py312/bin/pip install <pkg>
# or for packages already "satisfied" in ~/.local:
/pscratch/sd/r/ryotaro/anaconda3/envs/scigen_py312/bin/pip install <pkg> \
    --ignore-installed \
    --target /pscratch/sd/r/ryotaro/anaconda3/envs/scigen_py312/lib/python3.12/site-packages/
```

---

## GitHub

- Repo: https://github.com/RyotaroOKabe/SCIGEN_p_agent
- Initial push by user included `wandb/` binaries (~480 MB `.git/objects/`).
  Cleaned via orphan branch + `git reflog expire --expire=now --all && git gc --prune=now` → 33 MB.
- All SLURM scripts, config files, and screening scripts now tracked.

---

## Next Steps

1. Run `job_screen_mul.sh` to screen the standard pyrochlore outputs.
2. Investigate why Co has the highest pass rate (98/500 = 19.6%) vs. Cu (38/500 = 7.6%).
3. Create `script/generation_stop.py` — needed by `gen_mul_stop.py` and `gen_mul_stop_2d.py`; currently missing from all archives.
4. Submit `job_gen_natm.sh` for natm=88 pyrochlore generation once QOS slots are free.

---

## Commits

```
ebb3c42  Initial commit: SCIGEN_p_agent crystal structure generation framework
804a116  Add SLURM scripts for all 5 generation orchestrators
a482c6e  Add SLURM scripts for 3 screening orchestrators; update gen scripts
3c56d4b  Fix gitignore to track scripts/screening/; add elem screening script
```
