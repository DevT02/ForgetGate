# Resume-safe AutoAttack eval for seed 123
param(
  [int]$Seed = 123
)

$ErrorActionPreference = 'Stop'

$log = "results/logs/eval_autoattack_vit_cifar10_forget0_seed_${Seed}_evaluation.json"
if (Test-Path $log) {
  Write-Host "[SKIP] eval_autoattack_vit_cifar10_forget0 seed $Seed"
} else {
  $cmd = "python scripts/4_adv_evaluate.py --config configs/experiment_suites.yaml --suite eval_autoattack_vit_cifar10_forget0 --seed $Seed"
  Write-Host "[RUN] $cmd"
  & cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
}
