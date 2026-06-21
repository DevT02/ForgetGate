#!/usr/bin/env bash
# STAGED controlled base-robustness elasticity sweep. Train SalUn LoRA on five
# adversarial-strength ViT-tiny CIFAR-10 bases x two seeds, then audit each.
# Run with: FG_NUM_WORKERS=0 bash scripts/analysis/run_base_robustness_sweep.sh
# Discipline (learned the hard way): one sequential job, do NOT interrupt; if you
# must stop it, kill the whole tree with: taskkill /F /IM python.exe /T  AND the
# bash running this script -- TaskStop alone orphans the loop and it respawns.
set -u
cd "$(dirname "$0")/../.."
CFG=configs/suites/base_robustness_sweep.yaml
SUITES="unlearn_salun_advclean_vit_cifar10_forget0 unlearn_salun_advsmoke_vit_cifar10_forget0 unlearn_salun_advmid_vit_cifar10_forget0 unlearn_salun_advhigh_vit_cifar10_forget0 unlearn_salun_advultra_vit_cifar10_forget0"
SEEDS="42 123"

for suite in $SUITES; do
  for seed in $SEEDS; do
    dir="checkpoints/unlearn_lora/${suite}_seed_${seed}"
    if [ ! -f "${dir}/adapter_model.safetensors" ]; then
      echo "=== TRAIN ${suite} seed ${seed} ==="
      conda run -n forgetgate python scripts/2_train_unlearning_lora.py \
        --config "$CFG" --suite "$suite" --seed "$seed"
      if [ ! -f "${dir}/adapter_model.safetensors" ] && [ -d "${dir}/best_model" ]; then
        cp "${dir}/best_model/"adapter_model.safetensors "${dir}/best_model/"adapter_config.json "${dir}/" 2>/dev/null
      fi
    else
      echo "=== SKIP TRAIN ${suite} seed ${seed} (present) ==="
    fi
    tag=$(echo "$suite" | sed -e 's/^unlearn_//' -e 's/_vit_cifar10_forget0$//' -e 's/_/-/g')
    out="results/analysis/recovery_radius_cifar10_vit_tiny_forget0_seed_${seed}_test_with_retain_adam_margin_${tag}.json"
    if [ ! -f "$out" ]; then
      echo "=== AUDIT ${suite} seed ${seed} ==="
      conda run -n forgetgate python scripts/audits/17_recovery_radius_audit.py \
        --config "$CFG" --model-suites "$suite" --attack-loss adam_margin \
        --seed "$seed" --max-forget-samples 128 --max-retain-samples 128
    else
      echo "=== SKIP AUDIT ${suite} seed ${seed} (present) ==="
    fi
  done
done
echo "=== BASE-ROBUSTNESS SWEEP COMPLETE ==="
