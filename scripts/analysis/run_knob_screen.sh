#!/usr/bin/env bash
# Fast first-stage screen for a knob: train reduced-epoch points + cheap audits,
# print r_R/r_F per point so we can see if the knob moves the recovery radius.
# Usage: run_knob_screen.sh <config> <suite1> <suite2> ...
set -u
cd "$(dirname "$0")/../.."
CFG="$1"; shift
SEED=42
for suite in "$@"; do
  dir="checkpoints/unlearn_lora/${suite}_seed_${SEED}"
  if [ ! -f "${dir}/adapter_model.safetensors" ]; then
    echo "=== TRAIN ${suite} ==="
    conda run -n forgetgate python scripts/2_train_unlearning_lora.py \
      --config "$CFG" --suite "$suite" --seed "$SEED"
    if [ ! -f "${dir}/adapter_model.safetensors" ] && [ -d "${dir}/best_model" ]; then
      cp "${dir}/best_model/"adapter_model.safetensors "${dir}/best_model/"adapter_config.json "${dir}/" 2>/dev/null
    fi
  else
    echo "=== SKIP TRAIN ${suite} (present) ==="
  fi
  tag=$(echo "$suite" | sed -e 's/^unlearn_//' -e 's/_vit_cifar10_forget0$//' -e 's/_/-/g')
  out="results/analysis/recovery_radius_cifar10_vit_tiny_forget0_seed_${SEED}_test_with_retain_adam_margin_${tag}.json"
  if [ ! -f "$out" ]; then
    echo "=== AUDIT ${suite} (cheap) ==="
    conda run -n forgetgate python scripts/audits/17_recovery_radius_audit.py \
      --config "$CFG" --model-suites "$suite" --attack-loss adam_margin \
      --seed "$SEED" --max-forget-samples 48 --max-retain-samples 48
  else
    echo "=== SKIP AUDIT ${suite} (present) ==="
  fi
done
echo "=== KNOB SCREEN COMPLETE ==="
