#!/usr/bin/env bash
# Re-audit the seed-123 ResNet18/CIFAR-100 cross-arch adapters at a wider eps_max
# so the retain controls are not right-censored (default 8/255 censored several:
# retain success 0.62-0.96). An un-censored median recovery radius is budget-
# independent, so this makes the seed-123 r_R directly comparable to seed-42 and
# lets us tell whether the 2-seed pooled breakdown is a censoring artifact or real.
# Single sequential job -- do not interrupt.
set -u
cd "$(dirname "$0")/../.."
CFG=configs/suites/crossarch_resnet18.yaml
SEED=123
EPS=0.25
for suite in unlearn_salun_resnet18_ca_forget0 unlearn_scrub_resnet18_ca_forget0 unlearn_ceascent_resnet18_ca_forget0 unlearn_uniformkl_resnet18_ca_forget0 unlearn_featurescrub_resnet18_ca_forget0 unlearn_smoothedmargin_resnet18_ca_forget0; do
  echo "=== REAUDIT ${suite} (seed ${SEED}, eps ${EPS}) ==="
  conda run -n forgetgate python scripts/audits/17_recovery_radius_audit.py --config "$CFG" --model-suites "$suite" --attack-loss adam_margin --seed "$SEED" --max-forget-samples 48 --max-retain-samples 48 --eps-max "$EPS"
done
echo "=== REAUDIT SEED ${SEED} COMPLETE ==="
