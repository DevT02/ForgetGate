#!/usr/bin/env bash
# Controlled WD-sweep: train SalUn at 5 weight-decay values x 2 seeds, then audit
# each, to estimate a CAUSAL within-method recovery-radius elasticity.
set -u
cd "$(dirname "$0")/../.."
CFG=configs/suites/elasticity_wd_sweep.yaml
SUITES="unlearn_salun_wd0_vit_cifar10_forget0 unlearn_salun_wd1e3_vit_cifar10_forget0 unlearn_salun_wd1e2_vit_cifar10_forget0 unlearn_salun_wd1e1_vit_cifar10_forget0 unlearn_salun_wd5e1_vit_cifar10_forget0"
SEEDS="42 123"

for suite in $SUITES; do
  for seed in $SEEDS; do
    dir="checkpoints/unlearn_lora/${suite}_seed_${seed}"
    # Train only if the top-level adapter isn't already present.
    if [ ! -f "${dir}/adapter_model.safetensors" ]; then
      echo "=== TRAIN ${suite} seed ${seed} ==="
      conda run -n forgetgate python scripts/2_train_unlearning_lora.py \
        --config "$CFG" --suite "$suite" --seed "$seed"
      # Ensure the best_model adapter is promoted to the top-level dir the audit reads.
      if [ ! -f "${dir}/adapter_model.safetensors" ] && [ -d "${dir}/best_model" ]; then
        cp "${dir}/best_model/"adapter_model.safetensors "${dir}/best_model/"adapter_config.json "${dir}/" 2>/dev/null
      fi
    else
      echo "=== SKIP TRAIN ${suite} seed ${seed} (adapter present) ==="
    fi

    # Audit output tag: strip unlearn_ prefix and _vit_cifar10_forget0, swap _ for -.
    tag=$(echo "$suite" | sed -e 's/^unlearn_//' -e 's/_vit_cifar10_forget0$//' -e 's/_/-/g')
    out="results/analysis/recovery_radius_cifar10_vit_tiny_forget0_seed_${seed}_test_with_retain_adam_margin_${tag}.json"
    if [ ! -f "$out" ]; then
      echo "=== AUDIT ${suite} seed ${seed} ==="
      conda run -n forgetgate python scripts/audits/17_recovery_radius_audit.py \
        --config "$CFG" --model-suites "$suite" --attack-loss adam_margin \
        --seed "$seed" --max-forget-samples 128 --max-retain-samples 128
    else
      echo "=== SKIP AUDIT ${suite} seed ${seed} (output present) ==="
    fi
  done
done
echo "=== WD-SWEEP COMPLETE ==="
