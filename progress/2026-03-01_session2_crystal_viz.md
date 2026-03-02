# Session 2: Crystal Structure Visualizer — 2026-03-01

## Summary

Implemented a reusable, non-magnetic crystal structure visualizer (`crystal_viz.py`)
for CIF files using PyVista and pymatgen.  The tool is adapted from the existing
magnetic-structure visualizer at `MSN/utils/visualize/mplot3d.py`, with all
magnetic-moment logic stripped.  Ran a full batch of 203 kagome CIFs and
iteratively tuned the default view parameters based on user feedback.

---

## Deliverables

| File | Status | Notes |
|------|--------|-------|
| `utils/visualize/crystal_viz.py` | ✅ Created | 699 lines, full public API + CLI |
| `utils/visualize/jmol_param.csv` | ✅ Copied | 118 elements, colors + radii (pm) |

---

## Visualizer Design

### Public API

```python
visualize_crystal(cif_path, **kwargs)   # single CIF → PNG or interactive window
visualize_batch(cif_dir, output_dir, **kwargs)  # folder → one PNG per CIF
```

### Final Default Parameters (after tuning)

| Parameter | Default | Rationale |
|---|---|---|
| `atom_scale` | `0.25` | ~25% of Jmol vdW radii; atoms visually distinct without crowding |
| `elevation` | `75.0°` | Near bird's-eye — c-axis nearly perpendicular to image plane |
| `azimuth` | `30.0°` | Slight tilt to show 3D depth |
| `zoom_padding` | `0.45` | Zoomed out so full unit-cell wireframe fits in frame |
| `show_unit_cell` | `True` | Black wireframe, line_width=2, opacity=0.7 |
| `loop` | `True` | Periodic-image atoms within `_VIS_TOL=0.02` of cell faces |
| `lighting_preset` | `"standard"` | Single directional light + ambient |
| `background_color` | `"white"` | Clean white background for publications |
| `window_size` | `(1024, 768)` | |

### Key Technical Decisions

**CIF parsing** — uses `pymatgen.io.cif.CifParser.parse_structures()` (not
the deprecated `get_structures()`). Partial occupancy sites blend element
colors by weighted RGB average and use the dominant-species symbol.

**Atom rendering** — one `pv.Sphere` per atom with PBR (physically-based
rendering) properties: `metallic=0.1`, `roughness=0.5`, `ambient=0.3`,
`diffuse=0.7`, `specular=0.3`, `specular_power=30`.

**Camera setup** — spherical-coordinate camera:
```python
cam_pos = center + [dist*cos(el)*cos(az), dist*cos(el)*sin(az), dist*sin(el)]
```
Critical fix: at `elevation >= 60°`, the z-axis becomes nearly co-linear with
the view direction, causing gimbal lock (tilted images).  Fixed by switching
`camera.up = (0, 1, 0)` at high elevation:
```python
pl.camera.up = (0, 1, 0) if elevation >= 60.0 else (0, 0, 1)
```

**Boundary atoms** — for each atom within `_VIS_TOL=0.02` of a cell face,
a periodic image is added at the opposite face so the unit-cell wireframe
appears correctly closed.  Corner/edge atoms generate 3 or 7 images.

**Unit-cell wireframe** — 12-edge parallelepiped drawn from 8 fractional
corners `[0,0,0]…[1,1,1]` converted to Cartesian via `frac @ lattice_matrix`.

**Supercell** — na×nb×nc replication by translating Cartesian positions;
separate wireframe drawn for the supercell box in gray.

**Legend** — per-element color swatch in lower-right via `pl.add_legend()`.

---

## Iteration History

### Initial smoke test
- Default params from plan: `atom_scale=0.4`, `elevation=25°`, `zoom_padding=0.65`
- 203 CIFs rendered successfully: `gen_sc_kagMn500_000_filtered/images/`

### Round 1 improvements (user feedback: "unit cell too large, atoms too large")
- `atom_scale`: `0.4` → `0.25` (−37.5%)
- `zoom_padding`: `0.65` → `0.45` (zoom out)
- `elevation`: `25°` → `60°` (bird's-eye request)
- Re-ran 203 CIFs

### Round 2 fix (user clarification: "c-axis perpendicular to image plane")
- Problem: at `elevation=60°` with `camera.up=(0,0,1)`, gimbal lock caused
  images to appear tilted — the z-up vector is nearly co-linear with the
  view direction at high elevation
- Fix: `camera.up = (0,1,0)` when `elevation >= 60.0`
- `elevation`: `60°` → `75°` (stronger bird's-eye to make c-axis more perpendicular)
- Re-ran 203 CIFs — tilt eliminated, c-axis now visually perpendicular

---

## Batch Run Results

| Run | N CIFs | atom_scale | elevation | zoom_padding | Errors |
|-----|--------|-----------|-----------|-------------|--------|
| Initial | 203 | 0.4 | 25° | 0.65 | 0 |
| Round 1 | 203 | 0.25 | 60° | 0.45 | 0 |
| Round 2 (final) | 203 | 0.25 | 75° | 0.45 | 0 |

- Output: `hydra/singlerun/2026-02-24/mp_20_v1/gen_sc_kagMn500_000_filtered/images/`
- 203 PNG files, ~203 KB each, 1024×768

---

## CLI Usage

```bash
# Activate env (pyvista + pymatgen both available in mat_gen_all)
conda activate mat_gen_all
export PYTHONNOUSERSITE=1

# Single CIF
python utils/visualize/crystal_viz.py --cif structure.cif

# Single CIF, interactive window (requires X11)
python utils/visualize/crystal_viz.py --cif structure.cif --interactive

# Single CIF, 2×2×1 supercell, bird's-eye, save to specific path
python utils/visualize/crystal_viz.py \
    --cif KMn2S3.cif --supercell 2 2 1 --azimuth 45 --elevation 75 \
    --output KMn2S3_2x2x1.png

# Batch: all CIFs in a folder → PNGs in output_dir
python utils/visualize/crystal_viz.py \
    --dir gen_sc_kagMn500_000_filtered/ \
    --output_dir gen_sc_kagMn500_000_filtered/images/ \
    --atom_scale 0.25 --azimuth 30 --elevation 75 --zoom_padding 0.45
```

---

## Environment Note

`pyvista` was not installed in either environment at session start.
Installed into `mat_gen_all`:
```bash
/pscratch/sd/r/ryotaro/anaconda3/envs/mat_gen_all/bin/pip install pyvista
# → pyvista 0.47.1
```
`scigen_py312` does NOT have pyvista — only `mat_gen_all` is supported for
this tool (it also has pymatgen, which is required).

---

## Currently Running SLURM Jobs

| ID | Name | State | Runtime | Limit | Notes |
|----|------|-------|---------|-------|-------|
| 49542466 | GUIDED_DIFFUSION_SWEEP | RUNNING | 2:00 | 4:00 | magmom classifier v2 |
| 49539543 | GUIDED_PROJ_v2 | RUNNING | 3:16 | 12:00 | magmom projection v2 |
| 49538807 | ht4matgen_submit | RUNNING | 3:50 | 12:00 | DFT tier-1 candidates |

No SCIGEN_p_agent jobs are currently running.

---

## Pending Tasks

| Priority | Task | Notes |
|---|---|---|
| High | Submit pyrochlore generation | `gen_mul.py` already updated: `sc_list=['pyc'], idx_start=0`. Run `sbatch hpc/job_gen_mul.sh` (needs 1 free GPU slot on premium, currently 4/5 used by other projects) |
| Medium | Screen pyrochlore results | After generation: `sbatch hpc/job_screen_mul.sh` |
| Low | Create `script/generation_stop.py` | Missing; needed by `gen_mul_stop.py` and `gen_mul_stop_2d.py`. Adds `--stop_ratio` arg for early denoising termination |
| Low | Install pyvista in `scigen_py312` | Only needed if visualizer is called from that env |

---

## Reference

- Source reference: `/pscratch/sd/r/ryotaro/data/mag/MSN/utils/visualize/mplot3d.py`
  (~1800 lines, PyVista, magnetic structure visualizer with moment arrows)
- Adapted: stripped all magnetic-moment logic; replaced custom `MCifParser`
  with `pymatgen.io.cif.CifParser`
