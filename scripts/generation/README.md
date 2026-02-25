# Generation Scripts

This directory contains scripts for generating crystal structures from trained models.

## Scripts

- **`gen_mul.py`**: Batch generation script for multiple structure constraints
- **`gen_mul_stop.py`**: Generation with stop conditions
- **`gen_mul_elem.py`**: Element-specific generation
- **`gen_mul_natm.py`**: Generation with specified number of atoms
- **`gen_mul_stop_2d.py`**: 2D structure generation with stop conditions

## Usage

These scripts are wrapper scripts that call the core generation functionality in `../script/generation.py`.

Example:
```bash
python scripts/generation/gen_mul.py
```

## Configuration

Edit the script file to configure:
- Model path
- Dataset name
- Structure constraints (sc_list)
- Atom species (atom_list)
- Batch size and number of batches
- Output labels



