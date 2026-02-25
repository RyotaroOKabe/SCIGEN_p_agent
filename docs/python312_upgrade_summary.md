# Python 3.12 Upgrade - Summary

## Quick Overview

This document summarizes the key changes needed to upgrade SCIGEN_p_agent from Python 3.9.20 to Python 3.12.

## Critical Changes Required

### 1. PyTorch Lightning API Changes (3 changes)
**File**: `scigen/run.py`
- Line 80: `gpus = 0` → `accelerator = "cpu"` + `devices = 1`
- Line 154: **Remove** `progress_bar_refresh_rate` parameter
- Line 155: `resume_from_checkpoint` → `ckpt_path`

**Files**: `conf/train/default.yaml`, `conf/logging/default.yaml`
- Replace `gpus: 1` with `accelerator: gpu` + `devices: 1`
- Remove `progress_bar_refresh_rate: 1`

### 2. Environment Configuration
**File**: `setup/env.yml`
- Update Python: `3.8.*` → `3.12.*`
- Update all package versions to Python 3.12 compatible versions

### 3. Documentation Updates
**File**: `gnn_eval/README.md`
- Update Python version requirement

## Version Changes

| Package | Current | Target (Python 3.12) |
|---------|---------|---------------------|
| Python | 3.9.20 | 3.12.x |
| PyTorch | 2.0.1+cu118 | 2.1.0+ or 2.3.x+ |
| PyTorch Lightning | 1.3.8 | 2.1.x+ |
| torch-geometric | 2.3.0 | 2.5.x+ |
| hydra-core | 1.1.0 | 1.3.x+ |
| pymatgen | 2023.9.25 | 2024.x.x |
| torchmetrics | 0.7.3 | 1.2.x+ |
| matminer | 0.7.3 | 0.9.x+ |

## Migration Steps

1. **Update code files** (3 files: `scigen/run.py`, `conf/train/default.yaml`, `conf/logging/default.yaml`)
2. **Update environment file** (`setup/env.yml`)
3. **Create new environment** and install packages
4. **Test** with small dataset
5. **Update documentation**

## Detailed Documentation

- **Full Action Plan**: See `python312_upgrade_plan.md` for detailed explanations, troubleshooting, and all code changes
- **Quick Checklist**: See `python312_upgrade_checklist.md` for a step-by-step checklist

## Estimated Time

- Code changes: ~15-30 minutes
- Environment setup: ~30-60 minutes
- Testing: ~1-2 hours (depending on dataset size)

## Risk Level

**Medium**: Major PyTorch Lightning version upgrade (1.x → 2.x) requires API changes, but most changes are straightforward. Thorough testing recommended before production use.

---

**Last Updated**: 2024
**Status**: Ready for implementation
