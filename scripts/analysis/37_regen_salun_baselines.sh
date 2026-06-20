#!/usr/bin/env bash
# Regenerate SalUn baselines from the committed recipe (repro_salun.yaml) at
# seeds 42/123/456, then audit each at uniform steps=40. Writes a summary so
# Table 3's SalUn rows can be updated to reproducible numbers.
set -u
cd "$(dirname "$0")/../.." || exit 1
CFG=configs/suites/repro_salun.yaml
SUITE=unlearn_salun_repro_vit_cifar10_forget0
SUMMARY=results/analysis/reports/salun_regen_summary.txt
RUN="conda run -n forgetgate python"
: > "$SUMMARY"

for seed in 42 123 456; do
  ckpt="checkpoints/unlearn_lora/${SUITE}_seed_${seed}"
  if [ -f "$ckpt/adapter_model.safetensors" ]; then
    echo ">>> seed $seed: checkpoint exists, skipping training"
  else
    echo ">>> seed $seed: training SalUn repro"
    $RUN scripts/2_train_unlearning_lora.py --config "$CFG" --suite "$SUITE" --seed "$seed" >/dev/null 2>&1
  fi
  echo ">>> seed $seed: auditing (steps=40)"
  out=$($RUN scripts/audits/17_recovery_radius_audit.py --config "$CFG" --seed "$seed" \
        --model-suites "$SUITE" --split test --steps 40 --attack-loss adam_margin 2>/dev/null)
  path=$(printf '%s\n' "$out" | grep -oE 'results/analysis/[^ ]+\.json' | head -1)
  python - "$seed" "$path" >> "$SUMMARY" <<'PY'
import json,sys
seed,path=sys.argv[1],sys.argv[2]
d=json.load(open(path))
f=d['forget_recovery'][list(d['forget_recovery'])[0]]['summary']
r=d['retain_control'][list(d['retain_control'])[0]]['summary']
print("seed %s | forget: clean=%.3f median=%.5f succ=%.3f | retain median=%.5f | %s" % (
    seed, f.get('clean_success_rate',-1), f['median_radius'], f['success_rate'],
    r['median_radius'], path))
PY
done

echo "=== SalUn regen summary ==="
cat "$SUMMARY"
echo "DONE"
