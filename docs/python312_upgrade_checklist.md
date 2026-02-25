# Python 3.12 Upgrade Checklist

Quick reference checklist for the Python 3.12 upgrade process.

## Pre-Migration
- [ ] Backup current codebase
- [ ] Review `python312_upgrade_plan.md`
- [ ] Ensure Python 3.12 is installed on system

## Environment Setup
- [ ] Update `setup/env.yml` with Python 3.12 and target package versions
- [ ] Create new conda environment: `conda env create -f setup/env.yml -n scigen_py312`
- [ ] Activate environment: `conda activate scigen_py312`
- [ ] Verify Python version: `python --version` (should show 3.12.x)
- [ ] Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
- [ ] Verify PyTorch Lightning: `python -c "import pytorch_lightning; print(pytorch_lightning.__version__)"`

## Critical Code Changes

### PyTorch Lightning API Updates
- [ ] **`scigen/run.py` line 80**: Change `gpus = 0` to `accelerator = "cpu"` and `devices = 1` (or remove)
- [ ] **`scigen/run.py` line 154**: Remove `progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,`
- [ ] **`scigen/run.py` line 155**: Change `resume_from_checkpoint=ckpt` to `ckpt_path=ckpt`
- [ ] **`scigen/run.py` line 124**: Consider updating WandB `start_method="fork"` to `"thread"` or remove

### Configuration Files
- [ ] **`conf/train/default.yaml` line 9**: 
  - Remove: `gpus: 1`
  - Add: `accelerator: gpu` and `devices: 1`
- [ ] **`conf/logging/default.yaml` line 3**: Remove `progress_bar_refresh_rate: 1`

### Documentation Updates (Optional)
- [ ] **`gnn_eval/README.md` line 20**: Update `python==3.9.20` to `python>=3.12` (or remove version pin)
- [ ] Update main `README.md` with Python 3.12 requirement
- [ ] Create or update `requirements.txt` with pinned versions

## Testing

### Basic Functionality
- [ ] Test imports: `python -c "import scigen; print('OK')"`
- [ ] Test Hydra config loading: `python scigen/run.py --help`
- [ ] Test trainer initialization (with `fast_dev_run: True`)

### Training
- [ ] Run training with minimal dataset
- [ ] Verify checkpoint saving works
- [ ] Verify checkpoint loading works
- [ ] Check for deprecation warnings (should be minimal or none)

### Generation Scripts
- [ ] Test `gen_mul.py` (or similar generation script)
- [ ] Test evaluation scripts
- [ ] Verify WandB logging (if used)

### Performance Check
- [ ] Compare training speed (should be similar or better)
- [ ] Check memory usage (should be similar)
- [ ] Verify model outputs are consistent

## Post-Migration
- [ ] Document any issues encountered
- [ ] Update all documentation to reflect Python 3.12
- [ ] Test with actual workloads/data
- [ ] Remove old Python 3.9 environment (only after full validation)

## Rollback Plan
- [ ] Keep original Python 3.9 environment available
- [ ] Document any breaking changes found
- [ ] Have plan to revert if critical issues found

---

**Reference**: See `python312_upgrade_plan.md` for detailed explanations and troubleshooting.
