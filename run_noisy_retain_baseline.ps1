param(
  [int[]]$Seeds = @(42,123,456)
)

$ErrorActionPreference = "Stop"

function Run-IfMissing {
  param(
    [string]$OutPath,
    [string]$Cmd
  )
  if (Test-Path $OutPath) {
    Write-Host "[SKIP] $OutPath exists"
  } else {
    Write-Host "[RUN] $Cmd"
    & powershell -NoProfile -Command $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Cmd" }
  }
}

Write-Host "============================================================"
Write-Host "ForgetGate: Noisy retain-only baseline (resume-safe)"
Write-Host "============================================================"

foreach ($s in $Seeds) {
  $hist = "results/logs/unlearn_noisy_retain_vit_cifar10_forget0_seed_${s}_history.json"
  $cmd = "python scripts/2_train_unlearning_lora.py --config configs/experiment_suites.yaml --suite unlearn_noisy_retain_vit_cifar10_forget0 --seed $s"
  Run-IfMissing $hist $cmd
}

Write-Host "Done."
