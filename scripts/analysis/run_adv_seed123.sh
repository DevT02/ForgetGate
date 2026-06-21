#!/usr/bin/env bash
# Seed-123 replication of the PGD-base elasticity screen: train SalUn (8ep) on
# the five seed-123 adversarial bases, then audit at eps_max=0.25 (so robust
# bases are not censored). Single sequential job -- do not interrupt.
set -u
cd "$(dirname "$0")/../.."
CFG=configs/suites/base_robustness_screen.yaml
EPS=0.25
for suite in unlearn_salun_advclean_vit_cifar10_forget0 unlearn_salun_advsmoke_vit_cifar10_forget0 unlearn_salun_advmid_vit_cifar10_forget0 unlearn_salun_advhigh_vit_cifar10_forget0 unlearn_salun_advultra_vit_cifar10_forget0; do
  dir="checkpoints/unlearn_lora/${suite}_seed_123"
  if [ ! -f "${dir}/adapter_model.safetensors" ]; then
    echo "=== TRAIN ${suite} seed 123 ==="
    conda run -n forgetgate python scripts/2_train_unlearning_lora.py --config "$CFG" --suite "$suite" --seed 123
    if [ ! -f "${dir}/adapter_model.safetensors" ] && [ -d "${dir}/best_model" ]; then
      cp "${dir}/best_model/"adapter_model.safetensors "${dir}/best_model/"adapter_config.json "${dir}/" 2>/dev/null
    fi
  else
    echo "=== SKIP TRAIN ${suite} seed 123 ==="
  fi
  echo "=== AUDIT ${suite} seed 123 eps=${EPS} ==="
  conda run -n forgetgate python scripts/audits/17_recovery_radius_audit.py --config "$CFG" --model-suites "$suite" --attack-loss adam_margin --seed 123 --max-forget-samples 48 --max-retain-samples 48 --eps-max "$EPS"
done
echo "=== ADV SEED123 COMPLETE ==="
