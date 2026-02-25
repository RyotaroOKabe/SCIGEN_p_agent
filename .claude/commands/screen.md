# Command: Screen

Filter generated structures by SMACT charge-balance, space-occupation ratio,
and GNN-based stability / magnetic-moment predictions.

---

## Environment

```bash
conda activate scigen_py312
export PYTHONNOUSERSITE=1
cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
```

---

## Interactive Screening

```bash
python script/eval_screen.py \
    --label sc_kag100_test \
    --gen_cif True \
    --screen_mag False
```

Reads: `eval_gen_<label>.pt` from the checkpoint directory (set via `config_scigen.py`).
Writes:
- `eval_<label>.pt` — filtered structure tensors
- `cif_<label>/` — CIF files for structures that pass all filters (if `gen_cif=True`)

---

## Batch via SLURM

```bash
sbatch hpc/job_repro_screen.sh
```

Uses label `repro_kag1000_000` and reads `config_scigen.py` for checkpoint location.

---

## Batch Multi-Architecture Screening

```bash
# mp_20 model
python scripts/screening/screen_mul.py

# alex_2d model (uses config_scigen_2d.py)
python scripts/screening/screen_mul_2d.py
```

---

## Filters Applied

| Filter | Script | Description |
|---|---|---|
| Structural validity | `eval_screen.py` | Checks lattice + coordinate validity |
| SMACT charge balance | `eval_screen.py` | Requires charge-neutral composition |
| Space occupation ratio | `eval_screen.py` | Ratio < 1.7 (van der Waals radii) |
| Stability score A | GNN: `stab_240409-113155.torch` | Threshold: `screen_A > 0.5` |
| Stability score B | GNN: `stab_240402-111754.torch` | Threshold: `screen_B > 0.5` |
| Magnetic moment | GNN: `mag_240815-085301.torch` | Only if `--screen_mag True` |

---

## Key Arguments

| Arg | Description | Default |
|---|---|---|
| `--label` | Label matching the generation output | required |
| `--gen_cif` | Save passing structures as CIF files | `True` |
| `--screen_mag` | Apply magnetic moment filter | `False` |

---

## Config Files Used

- `config_scigen.py`: `hydra_dir`, `job_dir`, `stab_pred_name_A`, `stab_pred_name_B`, `mag_pred_name`
- `config_scigen_2d.py`: same fields, for alex_2d model runs
