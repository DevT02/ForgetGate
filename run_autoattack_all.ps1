Write-Host "============================================================"
Write-Host "ForgetGate: AutoAttack eval (seeds 456, 123)"
Write-Host "============================================================"

$env:CUDA_LAUNCH_BLOCKING = "0"

function Run-AutoAttack($seed) {
  Write-Host ""
  Write-Host "== Seed $seed =="
  python scripts/4_adv_evaluate.py --config configs/experiment_suites.yaml --suite eval_autoattack_vit_cifar10_forget0 --seed $seed 2>&1 | Tee-Object -FilePath "results/logs/autoattack_seed${seed}.log"
  if ($LASTEXITCODE -ne 0) { throw "AutoAttack failed for seed $seed" }
}

# Run whichever seeds you want
Run-AutoAttack 456
Run-AutoAttack 123

Write-Host "============================================================"
Write-Host "Done."
Write-Host "============================================================"
