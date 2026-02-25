#!/usr/bin/env bash
# run_repro.sh — SCIGEN_p_agent reproducibility entrypoint
#
# Usage:
#   bash run_repro.sh [--stage smoke|verify|generate|screen|all]
#
# Stages:
#   smoke    — fast_dev_run training on CPU (~1 min, no GPU needed)
#   verify   — print dataset sizes and checkpoint inventory
#   generate — run small generation batch using best alex2d checkpoint
#   screen   — screen the last generation output
#   all      — smoke + verify + generate + screen (default)
#
# Always run from the project root:
#   cd /pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
#   bash run_repro.sh

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO=/pscratch/sd/r/ryotaro/data/generative/SCIGEN_p_agent
CONDA_ENV=/pscratch/sd/r/ryotaro/anaconda3/envs/scigen_py312
CONDA_BASE=/pscratch/sd/r/ryotaro/anaconda3

# Best existing checkpoints (pre-trained; use directly without retraining)
CKPT_DIR_ALEX2D=/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun/2026-01-14/alex2d_py312_2gpu
CKPT_DIR_MP20=/pscratch/sd/r/ryotaro/data/generative/hydra/singlerun/2026-01-12/test_py312

# ── Argument parsing ─────────────────────────────────────────────────────────
STAGE="${1:---stage}"
if [[ "$STAGE" == "--stage" ]]; then
    STAGE="${2:-all}"
fi

echo "==========================================="
echo "  SCIGEN_p_agent Reproduction Script"
echo "  Stage: $STAGE"
echo "  Date:  $(date)"
echo "==========================================="

# ── Activate environment ──────────────────────────────────────────────────────
echo ""
echo "[setup] Activating conda env: $CONDA_ENV"
source "$CONDA_BASE/bin/activate" "$CONDA_ENV"
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline   # avoid requiring internet access for smoke tests

cd "$REPO"
echo "[setup] CWD: $(pwd)"
echo "[setup] Python: $(python --version)"
echo "[setup] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "[setup] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# ── Stage: verify ─────────────────────────────────────────────────────────────
run_verify() {
    echo ""
    echo "=== [verify] Dataset inventory ==="
    python - <<'PYEOF'
import torch, os
datasets = {
    "alex_2d": "data/alex_2d",
    "mp_20":   "data/mp_20",
}
for ds, base in datasets.items():
    for split in ["train", "val", "test"]:
        csv_path = f"{base}/{split}.csv"
        pt_path  = f"{base}/{split}_sym.pt"
        csv_ok = "✓" if os.path.exists(csv_path) else "✗"
        pt_ok  = "✓" if os.path.exists(pt_path)  else "✗"
        if os.path.exists(pt_path):
            data = torch.load(pt_path, weights_only=False)
            n = len(data)
        else:
            n = "MISSING"
        print(f"  {ds}/{split}: csv={csv_ok}  cache={pt_ok}  n={n}")
PYEOF

    echo ""
    echo "=== [verify] Checkpoint inventory ==="
    find /pscratch/sd/r/ryotaro/data/generative/hydra/singlerun \
        -name "*.ckpt" 2>/dev/null \
        | sort \
        | while read f; do
            sz=$(du -sh "$f" 2>/dev/null | cut -f1)
            echo "  $sz  $f"
          done

    echo ""
    echo "=== [verify] GNN eval models ==="
    ls -lh gnn_eval/models/*.torch 2>/dev/null || echo "  (none found)"
}

# ── Stage: smoke ──────────────────────────────────────────────────────────────
run_smoke() {
    echo ""
    echo "=== [smoke] 1-epoch / 2-batch training on CPU ==="
    # Notes:
    #   - fast_dev_run=True fails in PL 2.6.1 because trainer.test(ckpt_path="best")
    #     requires a saved checkpoint; use max_epochs=1 + limit_*_batches instead.
    #   - strategy=auto is required in PL 2.6.1 for single-device runs.
    #   - val_check_interval=1 ensures validation runs each epoch (default is 5).
    python scigen/run.py \
        data=mp_20 \
        model=diffusion_w_type \
        expname=smoke_repro \
        train.pl_trainer.max_epochs=1 \
        +train.pl_trainer.limit_train_batches=2 \
        +train.pl_trainer.limit_val_batches=2 \
        +train.pl_trainer.limit_test_batches=2 \
        train.pl_trainer.accelerator=cpu \
        train.pl_trainer.devices=1 \
        train.pl_trainer.strategy=auto \
        logging.val_check_interval=1 \
        logging.wandb.mode=offline
    echo "[smoke] PASSED"
}

# ── Stage: generate ───────────────────────────────────────────────────────────
run_generate() {
    echo ""
    echo "=== [generate] Small generation batch (kagome, 5 batches x 20) ==="
    if [[ ! -d "$CKPT_DIR_ALEX2D" ]]; then
        echo "ERROR: checkpoint dir not found: $CKPT_DIR_ALEX2D"
        echo "  Run training first: sbatch hpc/job_repro_train_alex2d.sh"
        exit 1
    fi

    python script/generation.py \
        --model_path  "$CKPT_DIR_ALEX2D" \
        --dataset     alex_2d \
        --label       repro_kag100_000 \
        --sc          kag \
        --batch_size  20 \
        --num_batches_to_samples 5 \
        --natm_range  1 12 \
        --known_species Mn Fe Co Ni Ru \
        --frac_z      0.5 \
        --t_mask      True

    echo "[generate] Output: $CKPT_DIR_ALEX2D/eval_gen_repro_kag100_000.pt"
}

# ── Stage: screen ─────────────────────────────────────────────────────────────
run_screen() {
    echo ""
    echo "=== [screen] Screening repro_kag100_000 ==="
    # eval_screen.py reads job_dir / hydra_dir from config_scigen.py
    # Override by setting env vars that config_scigen.py uses,
    # or by passing args directly.
    python script/eval_screen.py \
        --label repro_kag100_000 \
        --gen_cif True \
        --screen_mag False

    echo "[screen] Done. Check $CKPT_DIR_ALEX2D for filtered structures."
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$STAGE" in
    smoke)
        run_smoke
        ;;
    verify)
        run_verify
        ;;
    generate)
        run_generate
        ;;
    screen)
        run_screen
        ;;
    all)
        run_verify
        run_smoke
        run_generate
        run_screen
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Valid stages: smoke, verify, generate, screen, all"
        exit 1
        ;;
esac

echo ""
echo "==========================================="
echo "  Done at: $(date)"
echo "==========================================="
