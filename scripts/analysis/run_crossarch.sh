#!/usr/bin/env bash
# Cross-architecture observational replication on ResNet18/CIFAR-100: train each
# LoRA unlearning method, audit, so we can test the fragility confound and
# forget-radius super-elasticity on a different architecture + dataset.
# Single sequential job -- do not interrupt (TaskStop orphans the loop).
set -u
cd "$(dirname "$0")/../.."
CFG=configs/suites/crossarch_resnet18.yaml
for suite in unlearn_salun_resnet18_ca_forget0 unlearn_scrub_resnet18_ca_forget0 unlearn_ceascent_resnet18_ca_forget0 unlearn_uniformkl_resnet18_ca_forget0 unlearn_featurescrub_resnet18_ca_forget0 unlearn_smoothedmargin_resnet18_ca_forget0; do
  dir="checkpoints/unlearn_lora/${suite}_seed_42"
  if [ ! -f "${dir}/adapter_model.safetensors" ]; then
    echo "=== TRAIN ${suite} ==="
    conda run -n forgetgate python scripts/2_train_unlearning_lora.py --config "$CFG" --suite "$suite" --seed 42
    if [ ! -f "${dir}/adapter_model.safetensors" ] && [ -d "${dir}/best_model" ]; then
      cp "${dir}/best_model/"adapter_model.safetensors "${dir}/best_model/"adapter_config.json "${dir}/" 2>/dev/null
    fi
  else
    echo "=== SKIP TRAIN ${suite} (present) ==="
  fi
  tag=$(echo "$suite" | sed -e 's/^unlearn_//' | tr '_' '-')
  out="results/analysis/recovery_radius_cifar100_resnet18_forget0_seed_42_test_with_retain_adam_margin_${tag}.json"
  if [ ! -f "$out" ]; then
    echo "=== AUDIT ${suite} ==="
    conda run -n forgetgate python scripts/audits/17_recovery_radius_audit.py --config "$CFG" --model-suites "$suite" --attack-loss adam_margin --seed 42 --max-forget-samples 48 --max-retain-samples 48
  else
    echo "=== SKIP AUDIT ${suite} (present) ==="
  fi
done
echo "=== CROSSARCH COMPLETE ==="
