#!/usr/bin/env bash
# Reconstruct BalDRO/RURK baselines (scaled-smoke recipe; NOT the originals) at
# seeds 42/123/456, then audit each at uniform steps=40. RECONSTRUCTIONS — these
# will not match the original Table 3 numbers and must be labeled as such.
set -u
cd "$(dirname "$0")/../.." || exit 1
CFG=configs/suites/recon_baldro_rurk.yaml
SUMMARY=results/analysis/reports/recon_baldro_rurk_summary.txt
RUN="conda run -n forgetgate python"
# RURK's loss runs a nested model forward; Windows DataLoader workers deadlock
# on that. Force single-process loading for all training here (BalDRO already
# trained; this only affects the RURK reruns).
export FG_NUM_WORKERS=0
: > "$SUMMARY"

for method in baldro rurk; do
  SUITE="unlearn_${method}_recon_vit_cifar10_forget0"
  for seed in 42 123 456; do
    ckpt="checkpoints/unlearn_lora/${SUITE}_seed_${seed}"
    if [ -f "$ckpt/adapter_model.safetensors" ]; then
      echo ">>> $method seed $seed: checkpoint exists, skipping training"
    else
      echo ">>> $method seed $seed: training (recon)"
      $RUN scripts/2_train_unlearning_lora.py --config "$CFG" --suite "$SUITE" --seed "$seed" >/dev/null 2>&1
    fi
    echo ">>> $method seed $seed: auditing (steps=40)"
    out=$($RUN scripts/audits/17_recovery_radius_audit.py --config "$CFG" --seed "$seed" \
          --model-suites "$SUITE" --split test --steps 40 --attack-loss adam_margin 2>/dev/null)
    path=$(printf '%s\n' "$out" | grep -oE 'results/analysis/[^ ]+\.json' | head -1)
    python - "$method" "$seed" "$path" >> "$SUMMARY" <<'PY'
import json,sys
method,seed,path=sys.argv[1],sys.argv[2],sys.argv[3]
try:
    d=json.load(open(path))
    f=d['forget_recovery'][list(d['forget_recovery'])[0]]['summary']
    r=d['retain_control'][list(d['retain_control'])[0]]['summary']
    print("%-7s seed %-3s | forget: clean=%.3f median=%.5f succ=%.3f | retain median=%.5f" % (
        method, seed, f.get('clean_success_rate',-1), f['median_radius'], f['success_rate'], r['median_radius']))
except Exception as e:
    print("%-7s seed %-3s | AUDIT FAILED: %r" % (method, seed, e))
PY
  done
done
echo "=== recon summary (RECONSTRUCTIONS, not original baselines) ==="
cat "$SUMMARY"
echo "DONE"
