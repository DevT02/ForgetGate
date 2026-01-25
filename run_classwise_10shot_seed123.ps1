# Resume-safe runner for class-wise 10-shot (seed 123 by default)
param(
  [int]$Seed = 123
)

$ErrorActionPreference = 'Stop'

function Has-Log {
  param([string]$Suite, [int]$Seed)
  $path = "results/logs/${Suite}_seed_${Seed}.jsonl"
  return Test-Path $path
}

function Has-Oracle {
  param([string]$SuiteName, [int]$Seed)
  $paths = @(
    "checkpoints/oracle/${SuiteName}_seed_${Seed}_final.pt",
    "checkpoints/oracle/${SuiteName}_seed_${Seed}_final.pth",
    "checkpoints/oracle/${SuiteName}_seed_${Seed}_best.pt",
    "checkpoints/oracle/${SuiteName}_seed_${Seed}_best.pth"
  )
  foreach ($p in $paths) { if (Test-Path $p) { return $true } }
  return $false
}

function Get-Unlearned-Stamp {
  param([string]$SuiteName, [int]$Seed)
  $base = "checkpoints/unlearn_lora/${SuiteName}_seed_${Seed}"
  $paths = @(
    (Join-Path -Path $base -ChildPath "adapter_model.safetensors"),
    (Join-Path -Path $base -ChildPath "final_model/adapter_model.safetensors"),
    (Join-Path -Path $base -ChildPath "best_model/adapter_model.safetensors")
  )
  foreach ($p in $paths) {
    if (Test-Path $p) { return (Get-Item $p).LastWriteTime }
  }
  return $null
}

function Get-Log-Stamp {
  param([string]$Suite, [int]$Seed)
  $path = "results/logs/${Suite}_seed_${Seed}.jsonl"
  if (Test-Path $path) { return (Get-Item $path).LastWriteTime }
  return $null
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

function Run-If-Stale {
  param([string]$Cmd, [string]$Suite, [int]$Seed, [datetime]$InputStamp)
  $logStamp = Get-Log-Stamp -Suite $Suite -Seed $Seed
  if (-not $logStamp) {
    Write-Host "[RUN] $Cmd"
    & cmd /c $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Cmd" }
    return
  }
  if ($InputStamp -and ($InputStamp -gt $logStamp)) {
    Write-Host "[RERUN] $Suite seed $Seed (log older than unlearned checkpoint)"
    & cmd /c $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Cmd" }
  } else {
    Write-Host "[SKIP] $Suite seed $Seed"
  }
}

Write-Host "============================================================"
Write-Host "ForgetGate: Class-wise 10-shot (resume-safe)"
Write-Host "Seed: $Seed"
Write-Host "============================================================"

$classes = @(1,2,5,9)
foreach ($cls in $classes) {
  $oracleSuite = "oracle_vit_cifar10_forget${cls}"
  if (-not (Has-Oracle -SuiteName $oracleSuite -Seed $Seed)) {
    $cmd = "python scripts/1b_train_retrained_oracle.py --config configs/experiment_suites.yaml --suite $oracleSuite --seed $Seed"
    Write-Host "[RUN] $cmd"
    & cmd /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
  } else {
    Write-Host "[SKIP] $oracleSuite seed $Seed (checkpoint exists)"
  }

  $unlearnSuite = "unlearn_kl_vit_cifar10_forget${cls}"
  if (-not (Get-Unlearned-Stamp -SuiteName $unlearnSuite -Seed $Seed)) {
    $cmd = "python scripts/2_train_unlearning_lora.py --config configs/experiment_suites.yaml --suite $unlearnSuite --seed $Seed"
    Write-Host "[RUN] $cmd"
    & cmd /c $cmd
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
  } else {
    Write-Host "[SKIP] $unlearnSuite seed $Seed (checkpoint exists)"
  }

  $suiteOracle = "vpt_oracle_vit_cifar10_forget${cls}_10shot"
  $suiteKL = "vpt_resurrect_kl_forget${cls}_10shot"
  Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteOracle --seed $Seed" $suiteOracle $Seed

  $stamp = Get-Unlearned-Stamp -SuiteName $unlearnSuite -Seed $Seed
  Run-If-Stale "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteKL --seed $Seed" $suiteKL $Seed $stamp
}

Write-Host "============================================================"
Write-Host "Done."
Write-Host "============================================================"
