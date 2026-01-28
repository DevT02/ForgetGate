Write-Host "============================================================"
Write-Host "ForgetGate: Clean baseline eval (seed 456)"
Write-Host "============================================================"
$cmd = "python scripts/4_adv_evaluate.py --config configs/experiment_suites.yaml --suite eval_paper_baselines_vit_cifar10_forget0 --seed 456"
Write-Host "[RUN] $cmd"
& $cmd
if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
Write-Host "============================================================"
Write-Host "Done."
Write-Host "============================================================"
