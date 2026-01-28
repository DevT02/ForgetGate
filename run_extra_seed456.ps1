# Resume-safe runner for extra seed 456: k-shot + class-wise + controls
param(
  [int]$Seed = 456
)

$ErrorActionPreference = 'Stop'

function Has-Log {
  param([string]$Suite, [int]$Seed)
  $path = "results/logs/${Suite}_seed_${Seed}.jsonl"
  return Test-Path $path
}

function Has-Base {
  param([int]$Seed)
  $paths = @(
    "checkpoints/base/base_vit_cifar10_seed_${Seed}_final.pt",
    "checkpoints/base/base_vit_cifar10_seed_${Seed}_final.pth",
    "checkpoints/base/base_vit_cifar10_seed_${Seed}_best.pt",
    "checkpoints/base/base_vit_cifar10_seed_${Seed}_best.pth"
  )
  foreach ($p in $paths) { if (Test-Path $p) { return $true } }
  return $false
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

function Has-Unlearned {
  param([string]$SuiteName, [int]$Seed)
  $base = "checkpoints/unlearn_lora/${SuiteName}_seed_${Seed}"
  $paths = @(
    (Join-Path -Path $base -ChildPath "adapter_model.safetensors"),
    (Join-Path -Path $base -ChildPath "final_model/adapter_model.safetensors"),
    (Join-Path -Path $base -ChildPath "best_model/adapter_model.safetensors")
  )
  foreach ($p in $paths) { if (Test-Path $p) { return $true } }
  return $false
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
Write-Host "ForgetGate: Extra seed $Seed (k-shot + class-wise)"
Write-Host "============================================================"

# Ensure base checkpoint (force single-worker to avoid Windows shared mem errors)
if (-not (Has-Base -Seed $Seed)) {
  $old = $env:FG_NUM_WORKERS
  $env:FG_NUM_WORKERS = '0'
  $cmd = "python scripts/1_train_base.py --config configs/experiment_suites.yaml --suite base_vit_cifar10 --seed $Seed"
  Write-Host "[RUN] $cmd (FG_NUM_WORKERS=0)"
  & cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
  if ($null -ne $old) { $env:FG_NUM_WORKERS = $old } else { Remove-Item Env:FG_NUM_WORKERS -ErrorAction SilentlyContinue }
} else {
  Write-Host "[SKIP] base_vit_cifar10 seed $Seed (checkpoint exists)"
}

# Ensure oracle + unlearned checkpoints for forget0
$oracle0 = "oracle_vit_cifar10_forget0"
if (-not (Has-Oracle -SuiteName $oracle0 -Seed $Seed)) {
  $cmd = "python scripts/1b_train_retrained_oracle.py --config configs/experiment_suites.yaml --suite $oracle0 --seed $Seed"
  Write-Host "[RUN] $cmd"
  & cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
} else {
  Write-Host "[SKIP] $oracle0 seed $Seed (checkpoint exists)"
}

$unlearn0 = "unlearn_kl_vit_cifar10_forget0"
if (-not (Has-Unlearned -SuiteName $unlearn0 -Seed $Seed)) {
  $cmd = "python scripts/2_train_unlearning_lora.py --config configs/experiment_suites.yaml --suite $unlearn0 --seed $Seed"
  Write-Host "[RUN] $cmd"
  & cmd /c $cmd
  if ($LASTEXITCODE -ne 0) { throw "Command failed: $cmd" }
} else {
  Write-Host "[SKIP] $unlearn0 seed $Seed (checkpoint exists)"
}

# K-shot default prompt length (10 tokens)
$kshots = @(10,25,50,100)
foreach ($k in $kshots) {
  $suiteOracle = "vpt_oracle_vit_cifar10_forget0_${k}shot"
  $suiteKL = "vpt_resurrect_kl_forget0_${k}shot"
  Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteOracle --seed $Seed" $suiteOracle $Seed
  Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteKL --seed $Seed" $suiteKL $Seed
}

# Low-shot controls prompt length 5
$lowShots = @(1,5)
foreach ($k in $lowShots) {
  $suiteOracle = "vpt_oracle_vit_cifar10_forget0_${k}shot"
  $suiteKL = "vpt_resurrect_kl_forget0_${k}shot"
  Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteOracle --seed $Seed --prompt-length 5" "${suiteOracle}_prompt5" $Seed
  Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteKL --seed $Seed --prompt-length 5" "${suiteKL}_prompt5" $Seed
}

# Label controls (prompt length 5, k=10)
$suiteOracle = "vpt_oracle_vit_cifar10_forget0_10shot"
$suiteKL = "vpt_resurrect_kl_forget0_10shot"
Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteOracle --seed $Seed --prompt-length 5 --label-mode shuffle" "${suiteOracle}_prompt5_shufflelabels" $Seed
Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteKL --seed $Seed --prompt-length 5 --label-mode shuffle" "${suiteKL}_prompt5_shufflelabels" $Seed
Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteOracle --seed $Seed --prompt-length 5 --label-mode random" "${suiteOracle}_prompt5_randomlabels" $Seed
Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteKL --seed $Seed --prompt-length 5 --label-mode random" "${suiteKL}_prompt5_randomlabels" $Seed

# Class-wise (k=10, default prompt length)
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
  if (-not (Has-Unlearned -SuiteName $unlearnSuite -Seed $Seed)) {
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
  Run-If-Missing "python scripts/3_train_vpt_resurrector.py --config configs/experiment_suites.yaml --suite $suiteKL --seed $Seed" $suiteKL $Seed
}

Write-Host "============================================================"
Write-Host "Done."
Write-Host "============================================================"
