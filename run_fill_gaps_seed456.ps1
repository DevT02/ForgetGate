param(
  [int]$SeedLowShot = 456,
  [int[]]$DepSeeds = @(123,456)
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

# Low-shot controls (prompt length 5, k=1/5) for seed 456
Run-IfMissing "results/logs/vpt_oracle_vit_cifar10_forget0_10shot_prompt5_kshot1_seed_$SeedLowShot.jsonl" `
  "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_oracle_vit_cifar10_forget0_10shot_prompt5_kshot1 --seed $SeedLowShot"

Run-IfMissing "results/logs/vpt_oracle_vit_cifar10_forget0_10shot_prompt5_kshot5_seed_$SeedLowShot.jsonl" `
  "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_oracle_vit_cifar10_forget0_10shot_prompt5_kshot5 --seed $SeedLowShot"

Run-IfMissing "results/logs/vpt_resurrect_kl_forget0_10shot_prompt5_kshot1_seed_$SeedLowShot.jsonl" `
  "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_kl_forget0_10shot_prompt5_kshot1 --seed $SeedLowShot"

Run-IfMissing "results/logs/vpt_resurrect_kl_forget0_10shot_prompt5_kshot5_seed_$SeedLowShot.jsonl" `
  "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite vpt_resurrect_kl_forget0_10shot_prompt5_kshot5 --seed $SeedLowShot"

# Dependency-aware eval for seeds 123 and 456
foreach ($s in $DepSeeds) {
  Run-IfMissing "results/logs/eval_dependency_vit_cifar10_forget0_seed_${s}_evaluation.json" `
    "python scripts/4_adv_evaluate.py --config configs/experiment_suites.yaml --suite eval_dependency_vit_cifar10_forget0 --seed $s"
}

Write-Host "Done."
