# Python 3.12 Upgrade Action Plan

## Overview
This document outlines the steps required to upgrade the SCIGEN_p_agent codebase from Python 3.9.20 to Python 3.12, along with compatible versions of dependencies.

---

## Current vs Target Versions

### Current Versions
- **Python**: 3.9.20
- **PyTorch**: 2.0.1+cu118
- **torch-geometric**: 2.3.0
- **pytorch-lightning**: 1.3.8
- **pymatgen**: 2023.9.25
- **hydra-core**: 1.1.0
- **hydra-joblib-launcher**: 1.1.5
- **matminer**: 0.7.3
- **torchmetrics**: 0.7.3

### Recommended Target Versions (Python 3.12 Compatible)
- **Python**: 3.12.x
- **PyTorch**: 2.1.0+ (or latest 2.3.x/2.4.x with CUDA 12.x support)
- **torch-geometric**: 2.5.x or later
- **pytorch-lightning**: 2.1.x or later (major API changes)
- **pymatgen**: 2024.x.x (latest stable)
- **hydra-core**: 1.3.x or later
- **hydra-joblib-launcher**: 1.2.x or later
- **matminer**: 0.9.x or later
- **torchmetrics**: 1.2.x or later

---

## Critical Code Changes Required

### 1. PyTorch Lightning API Migration (Priority: HIGH)

#### 1.1 Deprecated `progress_bar_refresh_rate`
**Location**: `scigen/run.py:154`, `conf/logging/default.yaml:3`

**Current Code**:
```python
progress_bar_refresh_rate=cfg.logging.progress_bar_refresh_rate,
```

**Required Change**:
- **Option A** (PL 2.x): Remove this parameter entirely (default is enabled)
- **Option B**: Replace with `enable_progress_bar=True/False` if needed

**Action**:
1. Remove `progress_bar_refresh_rate` parameter from `scigen/run.py` line 154
2. Remove `progress_bar_refresh_rate: 1` from `conf/logging/default.yaml`
3. If progress bar control is needed, add `enable_progress_bar=True` to trainer config

#### 1.2 Deprecated `resume_from_checkpoint`
**Location**: `scigen/run.py:155`

**Current Code**:
```python
resume_from_checkpoint=ckpt,
```

**Required Change**:
Replace with `ckpt_path` parameter:
```python
ckpt_path=ckpt,
```

**Action**:
- Update `scigen/run.py` line 155 to use `ckpt_path=ckpt` instead of `resume_from_checkpoint=ckpt`

#### 1.3 Deprecated `gpus` Parameter
**Location**: `conf/train/default.yaml:9`, `scigen/run.py:80`

**Current Code**:
```yaml
pl_trainer:
  gpus: 1
```

**Required Change**:
```yaml
pl_trainer:
  accelerator: gpu
  devices: 1
```

**Action**:
1. Update `conf/train/default.yaml` line 9:
   - Replace `gpus: 1` with:
     ```yaml
     accelerator: gpu
     devices: 1
     ```
2. Update `scigen/run.py` line 80:
   - Replace `cfg.train.pl_trainer.gpus = 0` with:
     ```python
     cfg.train.pl_trainer.accelerator = "cpu"
     cfg.train.pl_trainer.devices = 1
     ```
   - Or remove entirely for fast_dev_run (uses CPU automatically)

#### 1.4 `trainer.test()` API Changes
**Location**: `scigen/run.py:165`

**Current Code**:
```python
trainer.test(datamodule=datamodule)
```

**Action**:
- This should still work, but verify in PL 2.x. May need to use `trainer.test(model=model, datamodule=datamodule)` if issues occur

#### 1.5 WandB Settings `start_method`
**Location**: `scigen/run.py:124`

**Current Code**:
```python
settings=wandb.Settings(start_method="fork"),
```

**Action**:
- On some systems (especially Windows/macOS), "fork" may not be available
- Change to `start_method="thread"` or `start_method="spawn"` for better compatibility
- Or remove and use default: `settings=wandb.Settings()`

---

### 2. Environment Configuration File Updates

#### 2.1 Update `setup/env.yml`
**Location**: `setup/env.yml`

**Required Changes**:
1. Update Python version: `python=3.12.*`
2. Update CUDA toolkit version (if needed): Use `cudatoolkit=11.8` or newer, or use PyTorch with CUDA built-in
3. Update package versions to match target versions above
4. For PyTorch 2.x with CUDA 12.x, may need to install from pip instead of conda

**Recommended env.yml structure**:
```yaml
channels:
- pytorch
- nvidia  # for CUDA support
- conda-forge
- defaults
dependencies:
- python=3.12.*
- pip
- pytorch>=2.1.0
- torchvision>=0.16.0
- pytorch-cuda=11.8  # or 12.1 for newer CUDA
- pip:
  - torch-geometric>=2.5.0
  - pytorch-lightning>=2.1.0
  - hydra-core>=1.3.0
  - hydra-joblib-launcher>=1.2.0
  - pymatgen>=2024.1.0
  - matminer>=0.9.0
  - torchmetrics>=1.2.0
  # ... other dependencies
```

---

### 3. Python 3.12 Specific Changes

#### 3.1 Typing Annotations (Optional but Recommended)
**Location**: Multiple files using `typing.List`, `typing.Dict`, etc.

**Current Code**:
```python
from typing import List, Dict
def build_callbacks(cfg: DictConfig) -> List[Callback]:
```

**Action**:
- Python 3.9+ supports lowercase built-in types in type hints
- Optional migration: Replace `List[Callback]` with `list[Callback]` and `Dict[str, Any]` with `dict[str, Any]`
- Files affected: `scigen/run.py`, `scigen/pl_modules/*.py`, `scigen/pl_data/datamodule.py`, etc.
- **Note**: This is optional - current code will still work, but modern Python style prefers lowercase types

#### 3.2 `distutils` Module Removal
**Location**: Check if any dependencies use `distutils`

**Action**:
- Python 3.12 removes `distutils` from stdlib
- Install `setuptools>=68.2.2` which includes distutils replacement
- Most packages should have migrated by now, but verify

---

### 4. Dependency-Specific Updates

#### 4.1 PyTorch Geometric
**Action**:
- Ensure compatible with PyTorch 2.x
- May need to reinstall torch-cluster, torch-scatter, torch-sparse as these are version-dependent
- Installation order matters: Install PyTorch first, then torch-geometric and extensions

#### 4.2 Hydra Core 1.3.x Changes
**Potential Issues**:
- Hydra 1.3.x may have minor API changes
- Check if `hydra.utils.log` usage is still valid (should be fine)
- Verify `hydra.utils.instantiate` behavior

**Action**:
- Test hydra configuration loading after upgrade
- Check for deprecation warnings

#### 4.3 pymatgen Updates
**Action**:
- Generally backward compatible
- May have new features that could be useful
- Check for any deprecated method warnings

#### 4.4 torchmetrics Updates
**Potential Issues**:
- torchmetrics 1.x has API changes from 0.7.3
- Check if any explicit torchmetrics usage exists in codebase

**Action**:
- Search for `torchmetrics` imports
- Update any explicit metric instantiation if needed

---

## Step-by-Step Migration Plan

### Phase 1: Preparation
1. ✅ **Create backup** of current codebase
2. ✅ **Review this document** and understand all changes
3. ✅ **Create new conda/venv environment** with Python 3.12

### Phase 2: Environment Setup
1. **Update `setup/env.yml`** with Python 3.12 and target package versions
2. **Create new environment**:
   ```bash
   conda env create -f setup/env.yml -n scigen_py312
   conda activate scigen_py312
   ```
3. **Verify installation**:
   ```bash
   python --version  # Should show 3.12.x
   python -c "import torch; print(torch.__version__)"
   python -c "import pytorch_lightning; print(pytorch_lightning.__version__)"
   ```

### Phase 3: Code Modifications
1. **Update PyTorch Lightning API calls**:
   - [ ] Remove `progress_bar_refresh_rate` from `scigen/run.py` and config
   - [ ] Change `resume_from_checkpoint` to `ckpt_path` in `scigen/run.py`
   - [ ] Update `gpus` to `accelerator`/`devices` in `conf/train/default.yaml` and `scigen/run.py`
   - [ ] Update WandB settings `start_method` if needed

2. **Update configuration files**:
   - [ ] Update `conf/logging/default.yaml` (remove progress_bar_refresh_rate)
   - [ ] Update `conf/train/default.yaml` (gpus → accelerator/devices)

3. **Optional modernizations**:
   - [ ] Update typing annotations to use lowercase types (list, dict, etc.)

### Phase 4: Testing
1. **Run basic imports**:
   ```bash
   python -c "import scigen; print('Import successful')"
   ```

2. **Test configuration loading**:
   ```bash
   python scigen/run.py --help
   ```

3. **Run unit tests** (if available):
   ```bash
   pytest
   ```

4. **Test training script** with small dataset:
   - Use `fast_dev_run: True` in config
   - Verify trainer initialization works
   - Check for deprecation warnings

5. **Test inference/generation scripts**:
   ```bash
   # Test with minimal examples
   python gen_mul.py  # (after updating config paths)
   ```

### Phase 5: Documentation and Cleanup
1. **Document any issues encountered** in this file
2. **Update README.md** with new Python version requirement
3. **Create requirements.txt** (if not exists) with pinned versions
4. **Update any setup instructions** in documentation

---

## Files That Need Modification

### Critical (Must Update)
1. **`scigen/run.py`** - PyTorch Lightning API changes
   - Line 80: `gpus` → `accelerator`/`devices`
   - Line 154: Remove `progress_bar_refresh_rate`
   - Line 155: `resume_from_checkpoint` → `ckpt_path`
   - Line 124: WandB settings `start_method` (optional)

2. **`conf/train/default.yaml`** - Trainer configuration
   - Line 9: `gpus: 1` → `accelerator: gpu` + `devices: 1`

3. **`conf/logging/default.yaml`** - Logging configuration
   - Line 3: Remove `progress_bar_refresh_rate: 1`

4. **`setup/env.yml`** - Environment specification
   - Update Python version and all package versions

### Optional (Recommended)
- All files using `typing.List`, `typing.Dict` → modernize to `list`, `dict`
- Add type hints where missing
- Update docstrings if API changed

5. **`gnn_eval/README.md`** - Documentation update
   - Line 20: Update `python==3.9.20` to `python>=3.12` or remove version pin if flexible

---

## Potential Issues and Solutions

### Issue 1: CUDA Version Mismatch
**Symptom**: PyTorch can't find CUDA or version mismatch errors

**Solution**:
- Ensure CUDA toolkit version matches PyTorch CUDA version
- For PyTorch 2.1+ with CUDA 11.8: Use `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- For CUDA 12.x: Use appropriate PyTorch build

### Issue 2: torch-geometric Installation Issues
**Symptom**: torch-geometric fails to install or import errors

**Solution**:
- Install PyTorch first, then torch-geometric
- Use pip installation: `pip install torch-geometric`
- May need to manually install extensions: `pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_version}.html`

### Issue 3: PyTorch Lightning 2.x API Warnings
**Symptom**: Deprecation warnings during training

**Solution**:
- Review PyTorch Lightning 2.x migration guide
- Update any deprecated API calls
- Check for new recommended patterns

### Issue 4: Hydra Configuration Loading Errors
**Symptom**: Hydra fails to load configs or instantiate objects

**Solution**:
- Verify Hydra 1.3.x is backward compatible with 1.1.0 configs
- Check for any deprecated Hydra config patterns
- Update config files if needed

### Issue 5: Memory Issues with Newer Versions
**Symptom**: OOM errors or slower training

**Solution**:
- Newer PyTorch versions may have different memory behavior
- Adjust batch sizes if needed
- Check for memory leaks with profiling tools

---

## Testing Checklist

- [ ] Environment creates successfully with Python 3.12
- [ ] All dependencies install without errors
- [ ] Imports work: `import torch`, `import pytorch_lightning`, `import scigen`
- [ ] Hydra config loading works
- [ ] Trainer initialization works
- [ ] Model instantiation works
- [ ] DataModule setup works
- [ ] Training loop runs (at least one step with fast_dev_run)
- [ ] Checkpoint saving/loading works
- [ ] WandB logging works (if enabled)
- [ ] Generation scripts work
- [ ] Evaluation scripts work
- [ ] No deprecation warnings (or only expected ones)
- [ ] Performance is acceptable (no major regressions)

---

## Rollback Plan

If issues are encountered:
1. Keep original Python 3.9 environment intact
2. Document specific issues in this file
3. Can create hybrid environment with Python 3.12 but older package versions if needed
4. Consider gradual migration (e.g., Python 3.10 first, then 3.12)

---

## Additional Resources

- [PyTorch Lightning 2.0 Migration Guide](https://lightning.ai/docs/pytorch/stable/common/migration.html)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Python 3.12 What's New](https://docs.python.org/3/whatsnew/3.12.html)
- [Hydra Changelog](https://github.com/facebookresearch/hydra/releases)

---

## Notes

- This migration involves major version upgrades (especially PyTorch Lightning 1.x → 2.x), so thorough testing is essential
- Some changes may require adjustments to training hyperparameters or workflows
- Keep the original environment available until migration is fully validated
- Consider setting up CI/CD with Python 3.12 to catch future compatibility issues

---

**Last Updated**: [Date]
**Status**: Planning Phase
