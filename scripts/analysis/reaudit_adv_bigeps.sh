#!/usr/bin/env bash
# Re-audit the PGD-base adapters at a larger eps_max so the robust bases' radii
# are not right-censored at the default 8/255 budget. Adapters already trained;
# this overwrites the censored audits with measurable ones.
set -u
cd "$(dirname "$0")/../.."
CFG=configs/suites/base_robustness_screen.yaml
EPS=0.25
for suite in unlearn_salun_advclean_vit_cifar10_forget0 unlearn_salun_advsmoke_vit_cifar10_forget0 unlearn_salun_advmid_vit_cifar10_forget0 unlearn_salun_advhigh_vit_cifar10_forget0 unlearn_salun_advultra_vit_cifar10_forget0; do
  echo "=== REAUDIT ${suite} eps=${EPS} ==="
  conda run -n forgetgate python scripts/audits/17_recovery_radius_audit.py \
    --config "$CFG" --model-suites "$suite" --attack-loss adam_margin \
    --seed 42 --max-forget-samples 48 --max-retain-samples 48 --eps-max "$EPS"
done
echo "=== REAUDIT COMPLETE ==="
