"""crystal_viz.py — Crystal structure visualizer (non-magnetic), PyVista backend.

Adapted from /pscratch/sd/r/ryotaro/data/mag/MSN/utils/visualize/mplot3d.py
(magnetic-structure visualizer).  All magnetic-moment logic has been removed;
CIF parsing uses pymatgen instead of the custom MCifParser.

Public API
----------
visualize_crystal(cif_path, **kwargs)
    Visualize a single CIF file.  Saves a PNG or opens an interactive window.

visualize_batch(cif_dir, output_dir, **kwargs)
    Loop over every *.cif in *cif_dir* and save one PNG per file to *output_dir*.

CLI
---
# Single CIF → PNG
python crystal_viz.py --cif structure.cif

# Single CIF, 2×2×1 supercell, interactive window
python crystal_viz.py --cif structure.cif --supercell 2 2 1 --interactive

# Batch
python crystal_viz.py --dir cif_folder/ --output_dir out_pngs/

Dependencies
------------
  pyvista, pymatgen, numpy  (all available in mat_gen_all / scigen_py312 envs)
"""
from __future__ import annotations

import argparse
import csv
import os
from itertools import product as iproduct
from math import cos, radians, sin
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyvista as pv
from pymatgen.io.cif import CifParser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_JMOL_CSV: Path = Path(__file__).parent / "jmol_param.csv"
_DEFAULT_COLOR: Tuple[float, float, float] = (0.5, 0.5, 0.5)   # grey fallback
_DEFAULT_RADIUS: float = 1.5                                     # Å fallback
_VIS_TOL: float = 0.02   # fractional-coord tolerance for periodic-boundary atoms


# ===========================================================================
# Low-level utilities (adapted from mplot3d.py)
# ===========================================================================

def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert '#E06633' → (0.878, 0.40, 0.20)."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return (int(h[0:2], 16) / 255.0,
                int(h[2:4], 16) / 255.0,
                int(h[4:6], 16) / 255.0)
    return _DEFAULT_COLOR


def blend_colors(
    colors: List[Tuple[float, float, float]],
    fractions: List[float],
) -> Tuple[float, float, float]:
    """Weighted RGB average — used for partial-occupancy (alloy) sites."""
    total = sum(fractions)
    if total == 0:
        return _DEFAULT_COLOR
    f = [x / total for x in fractions]
    r = sum(c[0] * w for c, w in zip(colors, f))
    g = sum(c[1] * w for c, w in zip(colors, f))
    b = sum(c[2] * w for c, w in zip(colors, f))
    return (r, g, b)


def load_atom_colors(
    csv_path: Optional[Path] = None,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Load Jmol standard colors and van-der-Waals radii.

    Returns
    -------
    color_dict  : {symbol: hex_string}   e.g. {'Fe': '#E06633'}
    radius_dict : {symbol: radius_Å}     radii are converted pm → Å (/100)
    """
    if csv_path is None:
        csv_path = _JMOL_CSV
    color_dict: Dict[str, str] = {}
    radius_dict: Dict[str, float] = {}
    try:
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                sym = row["atom"].strip()
                color_dict[sym] = row["color"].strip()
                radius_dict[sym] = float(row["radii"].strip()) / 100.0  # pm → Å
        print(f"Loaded colors/radii for {len(color_dict)} elements from {csv_path}")
    except Exception as exc:
        print(f"Warning: could not load {csv_path}: {exc}. Using default grey/1.5 Å.")
    return color_dict, radius_dict


def frac_to_cart(frac: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Fractional → Cartesian.  *matrix* rows are lattice vectors."""
    return frac @ matrix


# ===========================================================================
# Structure preprocessing
# ===========================================================================

def _boundary_atom_extras(
    frac_coords: np.ndarray,       # (N, 3)  values in [0, 1)
    atom_types:  List[str],
    site_occus:  List[Dict[str, float]],
    lattice_mat: np.ndarray,       # (3, 3)
    vis_tol:     float = _VIS_TOL,
) -> Tuple[np.ndarray, List[str], List[Dict[str, float]]]:
    """Return Cartesian positions / types / occupancies for periodic-image atoms.

    For each atom sitting within *vis_tol* of a cell face, add a copy at the
    opposite face so the wireframe looks "closed".  Corner/edge atoms generate
    multiple images (one per needed face combination).
    """
    extra_pos:   List[np.ndarray]        = []
    extra_types: List[str]               = []
    extra_occus: List[Dict[str, float]]  = []

    for frac, atype, occ in zip(frac_coords, atom_types, site_occus):
        # For each axis collect which offsets are needed (+1 near 0, -1 near 1)
        axis_shifts: List[List[int]] = []
        for d in range(3):
            shifts = [0]
            if frac[d] < vis_tol:
                shifts.append(+1)
            if frac[d] > 1.0 - vis_tol:
                shifts.append(-1)
            axis_shifts.append(shifts)

        # All combinations of shifts, excluding the [0,0,0] original
        for combo in iproduct(*axis_shifts):
            if all(c == 0 for c in combo):
                continue
            new_frac = frac + np.array(combo, dtype=float)
            extra_pos.append(new_frac @ lattice_mat)
            extra_types.append(atype)
            extra_occus.append(occ)

    if extra_pos:
        return np.array(extra_pos), extra_types, extra_occus
    return np.empty((0, 3)), [], []


def _expand_supercell(
    cart_coords: np.ndarray,           # (N, 3)
    atom_types:  List[str],
    site_occus:  List[Dict[str, float]],
    lattice_mat: np.ndarray,
    supercell:   Tuple[int, int, int],
) -> Tuple[np.ndarray, List[str], List[Dict[str, float]]]:
    """Replicate na × nb × nc unit cells (including the [0,0,0] original)."""
    na, nb, nc = supercell
    exp_pos, exp_types, exp_occus = [], [], []
    for i, j, k in iproduct(range(na), range(nb), range(nc)):
        T = np.array([i, j, k], dtype=float) @ lattice_mat
        for pos, at, occ in zip(cart_coords, atom_types, site_occus):
            exp_pos.append(pos + T)
        exp_types.extend(atom_types)
        exp_occus.extend(site_occus)
    return np.array(exp_pos), exp_types, exp_occus


# ===========================================================================
# CIF parsing
# ===========================================================================

def _parse_cif(
    cif_path: Path,
    primitive: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, float]], np.ndarray, str]:
    """Parse a standard CIF file with pymatgen.

    Returns
    -------
    frac_coords : (N, 3) fractional coordinates
    cart_coords : (N, 3) Cartesian coordinates (Å)
    atom_types  : dominant-species symbol per site
    site_occus  : {element_symbol: fraction} per site
    lattice_mat : (3, 3) lattice matrix (rows = vectors)
    formula     : reduced formula string
    """
    struct = CifParser(str(cif_path)).parse_structures(primitive=primitive)[0]

    frac_coords = struct.frac_coords          # (N, 3)
    cart_coords = struct.cart_coords          # (N, 3)
    lattice_mat = struct.lattice.matrix       # (3, 3)
    formula     = struct.formula.replace(" ", "")

    atom_types: List[str]               = []
    site_occus: List[Dict[str, float]]  = []
    for site in struct.sites:
        occ = {str(el): frac for el, frac in site.species.items()}
        dominant = max(occ, key=lambda e: occ[e])
        atom_types.append(dominant)
        site_occus.append(occ)

    return frac_coords, cart_coords, atom_types, site_occus, lattice_mat, formula


# ===========================================================================
# PyVista rendering helpers
# ===========================================================================

def _site_color(
    atype: str,
    occ:   Dict[str, float],
    color_dict: Dict[str, str],
) -> Tuple[float, float, float]:
    """Return an RGB color for a site (blended if partially occupied)."""
    if len(occ) == 1:
        return hex_to_rgb(color_dict.get(atype, "#808080"))
    colors    = [hex_to_rgb(color_dict.get(el, "#808080")) for el in occ]
    fractions = list(occ.values())
    return blend_colors(colors, fractions)


def _draw_atoms(
    pl:          pv.Plotter,
    positions:   np.ndarray,
    atom_types:  List[str],
    site_occus:  List[Dict[str, float]],
    color_dict:  Dict[str, str],
    radius_dict: Dict[str, float],
    atom_scale:  float,
    opacity:     float,
    metallic:    float = 0.1,
    roughness:   float = 0.5,
    ambient:     float = 0.3,
    diffuse:     float = 0.7,
    specular:    float = 0.3,
    specular_power: float = 30.0,
) -> List[Tuple[float, float, float]]:
    """Add one sphere per atom site. Returns the RGB color used for each site."""
    atom_colors_out: List[Tuple[float, float, float]] = []
    for pos, atype, occ in zip(positions, atom_types, site_occus):
        color  = _site_color(atype, occ, color_dict)
        radius = radius_dict.get(atype, _DEFAULT_RADIUS) * atom_scale
        sphere = pv.Sphere(radius=radius, center=pos,
                           theta_resolution=40, phi_resolution=40)
        pl.add_mesh(sphere, color=color, opacity=opacity,
                    show_edges=False, smooth_shading=True,
                    metallic=metallic, roughness=roughness,
                    ambient=ambient, diffuse=diffuse,
                    specular=specular, specular_power=specular_power)
        atom_colors_out.append(color)
    return atom_colors_out


def _make_parallelepiped_lines() -> np.ndarray:
    """Return the 12-edge connectivity array for a parallelepiped (8 corners)."""
    return np.array([
        2, 0, 1,  2, 1, 2,  2, 2, 3,  2, 3, 0,   # bottom face
        2, 4, 5,  2, 5, 6,  2, 6, 7,  2, 7, 4,   # top face
        2, 0, 4,  2, 1, 5,  2, 2, 6,  2, 3, 7,   # vertical edges
    ], dtype=int)


def _draw_wireframe(
    pl:          pv.Plotter,
    frac_corners: np.ndarray,   # (8, 3)
    lattice_mat: np.ndarray,
    color:       str,
    line_width:  float,
    opacity:     float,
) -> None:
    corners = np.array([frac_to_cart(fc, lattice_mat) for fc in frac_corners])
    mesh = pv.PolyData()
    mesh.points = corners
    mesh.lines  = _make_parallelepiped_lines()
    pl.add_mesh(mesh, color=color, line_width=line_width, opacity=opacity)


def _draw_unit_cell(
    pl:          pv.Plotter,
    lattice_mat: np.ndarray,
    color:       str   = "black",
    line_width:  float = 2.0,
    opacity:     float = 0.7,
) -> None:
    frac_corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    _draw_wireframe(pl, frac_corners, lattice_mat, color, line_width, opacity)


def _draw_supercell_box(
    pl:          pv.Plotter,
    lattice_mat: np.ndarray,
    supercell:   Tuple[int, int, int],
    color:       str   = "gray",
    line_width:  float = 3.0,
    opacity:     float = 0.5,
) -> None:
    na, nb, nc = supercell
    frac_corners = np.array([
        [0,  0,  0 ], [na, 0,  0 ], [na, nb, 0 ], [0,  nb, 0 ],
        [0,  0,  nc], [na, 0,  nc], [na, nb, nc], [0,  nb, nc],
    ], dtype=float)
    _draw_wireframe(pl, frac_corners, lattice_mat, color, line_width, opacity)


def _add_element_legend(
    pl:          pv.Plotter,
    atom_types:  List[str],
    atom_colors: List[Tuple[float, float, float]],
) -> None:
    """Add a per-element color legend (lower-right corner)."""
    seen: Dict[str, Tuple[float, float, float]] = {}
    for atype, color in zip(atom_types, atom_colors):
        if atype not in seen:
            seen[atype] = color
    if not seen:
        return
    entries = [[sym, col] for sym, col in seen.items()]
    n = len(entries)
    height = min(0.04 * n + 0.04, 0.40)
    pl.add_legend(entries, bcolor="white", border=True,
                  size=(0.14, height), loc="lower right")


def _setup_camera(
    pl:           pv.Plotter,
    positions:    np.ndarray,
    azimuth:      float,
    elevation:    float,
    zoom_padding: float,
) -> None:
    center  = positions.mean(axis=0)
    bbox_min, bbox_max = positions.min(axis=0), positions.max(axis=0)
    max_dim = float(np.max(bbox_max - bbox_min))
    distance = max_dim * 1.8

    az_r = radians(azimuth)
    el_r = radians(elevation)
    cam_pos = center + np.array([
        distance * cos(el_r) * cos(az_r),
        distance * cos(el_r) * sin(az_r),
        distance * sin(el_r),
    ])

    # Auto-fit to get a reference distance, then apply custom view angle
    pl.reset_camera(bounds=(bbox_min[0], bbox_max[0],
                            bbox_min[1], bbox_max[1],
                            bbox_min[2], bbox_max[2]))
    auto_dist = float(np.linalg.norm(np.array(pl.camera.position) - center))

    pl.camera.position    = cam_pos
    pl.camera.focal_point = center
    # When looking nearly down the c-axis (high elevation), z-axis becomes
    # collinear with the view direction and causes gimbal lock / image tilt.
    # Use the b-axis (y) as "up" so the ab-plane reads naturally.
    pl.camera.up = (0, 1, 0) if elevation >= 60.0 else (0, 0, 1)

    new_dist = float(np.linalg.norm(cam_pos - center))
    pl.camera.zoom((auto_dist / new_dist) * zoom_padding)


def _setup_lighting(
    pl:       pv.Plotter,
    positions: np.ndarray,
    preset:   str   = "standard",
    intensity: float = 1.0,
    light_color: str | Tuple = "white",
) -> None:
    center  = positions.mean(axis=0)
    max_size = float(np.max(positions.max(axis=0) - positions.min(axis=0)))
    dist    = max_size * 2.5

    color_map = {
        "white":   (1.0, 1.0, 1.0),
        "warm":    (1.0, 0.95, 0.9),
        "cool":    (0.9, 0.95, 1.0),
        "daylight":(1.0, 1.0, 0.98),
    }
    rgb = (color_map.get(light_color.lower(), (1.0, 1.0, 1.0))
           if isinstance(light_color, str) else light_color)

    # Remove PyVista default lights
    try:
        renderer = pl.renderer
        for lt in list(renderer.GetLights()):
            renderer.RemoveLight(lt)
    except Exception:
        pass
    try:
        pl.disable_shadows()
    except Exception:
        pass

    preset = preset.lower()
    if preset == "diffused":
        for delta in ([1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]):
            lt = pv.Light(position=center + np.array(delta) * dist,
                          focal_point=center,
                          intensity=intensity * 0.6, color=rgb)
            pl.add_light(lt)
        amb = pv.Light(position=center + np.array([0, 0, dist * 2]),
                       focal_point=center,
                       intensity=intensity * 0.4, color=rgb)
        amb.positional = False
        pl.add_light(amb)
    elif preset == "top_left":
        lt = pv.Light(position=center + np.array([-dist, dist, dist * 1.5]),
                      focal_point=center, intensity=intensity, color=rgb)
        pl.add_light(lt)
    elif preset == "right":
        lt = pv.Light(position=center + np.array([dist * 1.5, 0, dist]),
                      focal_point=center, intensity=intensity, color=rgb)
        pl.add_light(lt)
    else:  # "standard"
        lt = pv.Light(position=center + np.array([dist, dist, dist]),
                      focal_point=center, intensity=intensity, color=rgb)
        pl.add_light(lt)


# ===========================================================================
# Public API
# ===========================================================================

def visualize_crystal(
    cif_path:         str | Path,
    *,
    supercell:        Tuple[int, int, int] = (1, 1, 1),
    atom_scale:       float = 0.25,
    show_unit_cell:   bool  = True,
    show_supercell:   bool  = True,
    loop:             bool  = True,
    interactive:      bool  = False,
    output_file:      Optional[str] = None,
    azimuth:          float = 30.0,
    elevation:        float = 75.0,
    zoom_padding:     float = 0.45,
    opacity:          float = 1.0,
    lighting_preset:  str   = "standard",
    window_size:      Tuple[int, int] = (1024, 768),
    primitive:        bool  = False,
    background_color: str   = "white",
    title:            Optional[str] = None,
    jmol_csv:         Optional[Path] = None,
) -> None:
    """Visualize a crystal structure from a CIF file.

    Parameters
    ----------
    cif_path        : Path to a standard (non-magnetic) CIF file.
    supercell       : (na, nb, nc) replication of the unit cell.
    atom_scale      : Multiply all atomic radii by this factor.
    show_unit_cell  : Draw the unit-cell wireframe.
    show_supercell  : Draw the supercell wireframe (only when supercell≠(1,1,1)).
    loop            : Add periodic images of atoms near cell faces.
    interactive     : Open a PyVista interactive window (requires X11 / display).
                      False → save PNG to *output_file*.
    output_file     : PNG path.  Auto-named ``{formula}_{stem}.png`` when None.
    azimuth         : Camera horizontal angle in degrees.
    elevation       : Camera vertical angle in degrees.
    zoom_padding    : Controls view tightness (lower = more zoomed out, try 0.4–0.9).
    opacity         : Atom sphere opacity (0–1).
    lighting_preset : ``"standard"`` | ``"diffused"`` | ``"top_left"`` | ``"right"``.
    window_size     : Render resolution in pixels (width, height).
    primitive       : Reduce to primitive cell with pymatgen before plotting.
    background_color: Background color string (``"white"``, ``"black"``, …).
    title           : Text shown in upper-left corner (default: reduced formula).
    jmol_csv        : Path to jmol_param.csv. Defaults to the copy alongside this script.
    """
    cif_path = Path(cif_path)
    color_dict, radius_dict = load_atom_colors(jmol_csv)

    frac_coords, cart_coords, atom_types, site_occus, lattice_mat, formula = \
        _parse_cif(cif_path, primitive=primitive)

    n_uc = len(atom_types)
    print(f"[crystal_viz] {formula}  ({n_uc} atoms in unit cell)  ← {cif_path.name}")

    # --- Boundary atoms ---
    all_cart  = list(cart_coords)
    all_types = list(atom_types)
    all_occus = list(site_occus)
    if loop:
        ex_pos, ex_types, ex_occus = _boundary_atom_extras(
            frac_coords, atom_types, site_occus, lattice_mat)
        if len(ex_pos) > 0:
            all_cart  = list(cart_coords) + list(ex_pos)
            all_types = all_types + ex_types
            all_occus = all_occus + ex_occus

    all_cart  = np.array(all_cart)
    all_types = list(all_types)
    all_occus = list(all_occus)

    # --- Supercell expansion ---
    if supercell != (1, 1, 1):
        all_cart, all_types, all_occus = _expand_supercell(
            all_cart, all_types, all_occus, lattice_mat, supercell)

    positions = all_cart
    print(f"[crystal_viz] Rendering {len(positions)} atoms "
          f"(supercell={supercell}, loop={loop})")

    # --- PyVista plotter ---
    pl = pv.Plotter(window_size=list(window_size), off_screen=not interactive)
    pl.background_color = background_color

    # Atoms
    atom_colors_list = _draw_atoms(
        pl, positions, all_types, all_occus,
        color_dict, radius_dict,
        atom_scale=atom_scale, opacity=opacity,
    )

    # Cell wireframes
    if show_unit_cell:
        _draw_unit_cell(pl, lattice_mat)
    if show_supercell and supercell != (1, 1, 1):
        _draw_supercell_box(pl, lattice_mat, supercell)

    # Legend + title
    _add_element_legend(pl, all_types, atom_colors_list)
    pl.add_text(title if title else formula, position="upper_left", font_size=12)
    pl.add_axes()

    # Camera + lighting
    _setup_camera(pl, positions, azimuth, elevation, zoom_padding)
    _setup_lighting(pl, positions, preset=lighting_preset)

    # Output
    if interactive:
        pl.show()
    else:
        if output_file is None:
            output_file = f"{formula}_{cif_path.stem}.png"
        pl.screenshot(output_file, transparent_background=False)
        print(f"[crystal_viz] Saved → {output_file}")
        pl.close()


def visualize_batch(
    cif_dir:    str | Path,
    output_dir: str | Path,
    glob:       str  = "*.cif",
    verbose:    bool = True,
    **kwargs,
) -> None:
    """Visualize every CIF file in *cif_dir* and save PNGs to *output_dir*.

    Parameters
    ----------
    cif_dir    : Folder containing *.cif files.
    output_dir : Destination for PNG files (created if it does not exist).
    glob       : File-name pattern (default ``"*.cif"``).
    verbose    : Print per-file progress.
    **kwargs   : Forwarded to :func:`visualize_crystal`
                 (``interactive`` is forced ``False``).
    """
    cif_dir    = Path(cif_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs["interactive"] = False   # batch is always off-screen

    cif_files = sorted(cif_dir.glob(glob))
    if not cif_files:
        print(f"[crystal_viz] No CIF files matching '{glob}' in {cif_dir}")
        return

    print(f"[crystal_viz] Batch: {len(cif_files)} files → {output_dir}")
    errors = []
    for idx, cif_path in enumerate(cif_files, 1):
        out_png = output_dir / f"{cif_path.stem}.png"
        if verbose:
            print(f"  [{idx}/{len(cif_files)}] {cif_path.name}")
        try:
            visualize_crystal(cif_path, output_file=str(out_png), **kwargs)
        except Exception as exc:
            msg = f"    ERROR processing {cif_path.name}: {exc}"
            print(msg)
            errors.append(msg)

    n_ok = len(cif_files) - len(errors)
    print(f"[crystal_viz] Done. {n_ok}/{len(cif_files)} images saved to {output_dir}")
    if errors:
        print(f"  {len(errors)} file(s) failed.")


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crystal_viz.py",
        description="Crystal structure visualizer (non-magnetic) — PyVista backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input (mutually exclusive)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--cif", metavar="PATH",
                     help="Single CIF file to visualize.")
    grp.add_argument("--dir", metavar="DIR",
                     help="Folder of CIF files (batch mode).")

    # Output
    p.add_argument("--output", metavar="PATH",
                   help="PNG output path (single-CIF mode). Auto-named if omitted.")
    p.add_argument("--output_dir", metavar="DIR", default="crystal_images",
                   help="Output folder for batch mode.")

    # Structure options
    p.add_argument("--supercell", nargs=3, type=int, default=[1, 1, 1],
                   metavar=("NA", "NB", "NC"), help="Supercell replication.")
    p.add_argument("--atom_scale", type=float, default=0.25,
                   help="Scale factor applied to all atomic radii.")
    p.add_argument("--no_unit_cell", action="store_true",
                   help="Hide unit-cell wireframe.")
    p.add_argument("--no_supercell", action="store_true",
                   help="Hide supercell wireframe.")
    p.add_argument("--no_loop", action="store_true",
                   help="Disable periodic-boundary atom images.")
    p.add_argument("--primitive", action="store_true",
                   help="Reduce to primitive cell before plotting.")

    # Camera / view
    p.add_argument("--azimuth",      type=float, default=30.0,
                   help="Camera azimuth angle (degrees).")
    p.add_argument("--elevation",    type=float, default=75.0,
                   help="Camera elevation angle (degrees).")
    p.add_argument("--zoom_padding", type=float, default=0.45,
                   help="Zoom padding (lower = more zoomed out).")

    # Rendering
    p.add_argument("--opacity",    type=float, default=1.0,
                   help="Atom sphere opacity (0–1).")
    p.add_argument("--lighting",   default="standard",
                   choices=["standard", "diffused", "top_left", "right"],
                   help="Lighting preset.")
    p.add_argument("--background", default="white",
                   help="Background color (e.g. white, black, lightgrey).")
    p.add_argument("--width",      type=int, default=1024,
                   help="Image width in pixels.")
    p.add_argument("--height",     type=int, default=768,
                   help="Image height in pixels.")
    p.add_argument("--interactive", action="store_true",
                   help="Open interactive window (requires X11 / display).")
    p.add_argument("--title",       default=None,
                   help="Title text (upper-left). Defaults to reduced formula.")
    p.add_argument("--jmol_csv",    default=None,
                   help="Path to jmol_param.csv (optional override).")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    jmol_csv = Path(args.jmol_csv) if args.jmol_csv else None
    kwargs = dict(
        supercell       = tuple(args.supercell),
        atom_scale      = args.atom_scale,
        show_unit_cell  = not args.no_unit_cell,
        show_supercell  = not args.no_supercell,
        loop            = not args.no_loop,
        azimuth         = args.azimuth,
        elevation       = args.elevation,
        zoom_padding    = args.zoom_padding,
        opacity         = args.opacity,
        lighting_preset = args.lighting,
        background_color= args.background,
        window_size     = (args.width, args.height),
        primitive       = args.primitive,
        interactive     = args.interactive,
        title           = args.title,
        jmol_csv        = jmol_csv,
    )

    if args.cif:
        visualize_crystal(args.cif, output_file=args.output, **kwargs)
    else:
        visualize_batch(args.dir, args.output_dir, **kwargs)


if __name__ == "__main__":
    main()
