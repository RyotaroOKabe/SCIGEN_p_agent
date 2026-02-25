# Screening Scripts

This directory contains scripts for screening and evaluating generated crystal structures.

## Scripts

- **`screen_mul.py`**: Batch screening script for multiple structure types
- **`screen_mul_natm.py`**: Screening with specified number of atoms
- **`screen_mul_2d.py`**: 2D structure screening

## Usage

These scripts call the core evaluation functionality in `../script/eval_screen.py`.

Example:
```bash
python scripts/screening/screen_mul.py
```

## Configuration

Edit the script file to configure:
- Number of materials to screen
- Structure architectures (arch_list)
- Whether to generate CIF files
- Whether to screen for magnetic properties



