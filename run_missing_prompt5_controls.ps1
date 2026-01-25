# Resume-safe runner for missing prompt5 low-shot + random-label controls
param(
  [int]$SeedA = 42,
  [int]$SeedB = 123
)

$ErrorActionPreference = 'Stop'

function Has-Log {
  param([string]$Suite, [int]$Seed)
  $path = "results/logs/${Suite}_seed_${Seed}.jsonl"
  return Test-Path $path
}

function Run-If-Missing {
  param([string]$Cmd, [string]$Suite, [int]$Seed)
  if (Has-Log -Suite $Suite -Seed $Seed) {
    Write-Host "[SKIP] $Suite seed $Seed"
  } else {
    Write-Host "[RUN] $Cmd"
    & cmd /c $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Cmd" }
  }
}

Write-Host "============================================================"
Write-Host "ForgetGate: Missing prompt5 controls (resume-safe)"
Write-Host "Seeds: $SeedA, $SeedB"
Write-Host "============================================================"

# Low-shot prompt5 (k=1,5) Oracle + KL
$lowShots = @(1,5)
foreach ($k in $lowShots) {
  foreach ($seed in @($SeedA,$SeedB)) {
    $suiteOracle = "vpt_oracle_vit_cifar10_forget0_${k}shot"
    $suiteKL = "vpt_resurrect_kl_forget0_${k}shot"
    Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteOracle --seed $seed --prompt-length 5" "${suiteOracle}_prompt5" $seed
    Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteKL --seed $seed --prompt-length 5" "${suiteKL}_prompt5" $seed
  }
}

# Random-label control prompt5 k=10 (seed 42 only missing)
$seed = $SeedA
$suiteOracle = "vpt_oracle_vit_cifar10_forget0_10shot"
$suiteKL = "vpt_resurrect_kl_forget0_10shot"
Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteOracle --seed $seed --prompt-length 5 --label-mode random" "${suiteOracle}_prompt5_randomlabels" $seed
Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteKL --seed $seed --prompt-length 5 --label-mode random" "${suiteKL}_prompt5_randomlabels" $seed

Write-Host "============================================================"
Write-Host "Done."
Write-Host "============================================================"
