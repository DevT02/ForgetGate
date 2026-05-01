param(
    [string]$EnvName = "forgetgate",
    [string]$Config = "configs/experiment_suites.yaml",
    [switch]$UseActivePython,
    [int[]]$Seeds = @(42, 123),
    [int]$TransferSeed = 42,
    [string[]]$TargetSuites = @(
        "unlearn_kl_vit_cifar10_forget0",
        "unlearn_salun_vit_cifar10_forget0",
        "unlearn_scrub_distill_vit_cifar10_forget0",
        "unlearn_orbit_vit_cifar10_forget0",
        "oracle_vit_cifar10_forget0"
    ),
    [switch]$SkipTrain,
    [switch]$SkipTransfer,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Get-PythonRunner {
    if ($UseActivePython -or $env:CONDA_DEFAULT_ENV -eq $EnvName) {
        return @{
            Exe = "python"
            Args = @()
            Label = "python"
        }
    }

    return @{
        Exe = "conda"
        Args = @("run", "--no-capture-output", "-n", $EnvName, "python")
        Label = "conda run -n $EnvName python"
    }
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$CommandArgs
    )

    $runner = Get-PythonRunner
    $fullArgs = @($runner.Args + $CommandArgs)
    Write-Host ""
    Write-Host "[RUN ] $Name"
    Write-Host "       $($runner.Label) $($CommandArgs -join ' ')"
    if ($DryRun) {
        return
    }
    & $runner.Exe @fullArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Name' failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipTrain) {
    foreach ($seed in $Seeds) {
        Invoke-PythonStep -Name "train universal patch seed=$seed" -CommandArgs @(
            "scripts/13_train_universal_patch.py",
            "--config", $Config,
            "--suite", "patch_resurrect_kl_forget0_10shot",
            "--seed", "$seed",
            "--num-workers", "0"
        )
    }
}

if (-not $SkipTransfer) {
    $patchDir = "checkpoints/universal_patch/patch_resurrect_kl_forget0_10shot_seed_$TransferSeed"
    $transferArgs = @(
        "scripts/14_transfer_universal_patch.py",
        "--config", $Config,
        "--patch-dir", $patchDir,
        "--patch-seed", "$TransferSeed",
        "--target-suites"
    ) + $TargetSuites + @(
        "--eval-batch-size", "256",
        "--num-workers", "0"
    )
    Invoke-PythonStep -Name "transfer universal patch" -CommandArgs $transferArgs
}

Write-Host ""
Write-Host "Universal patch attack pipeline complete."
Write-Host "Key outputs:"
Write-Host "  checkpoints/universal_patch/patch_resurrect_kl_forget0_10shot_seed_<seed>"
Write-Host "  results/analysis/patch_resurrect_kl_forget0_10shot_seed_<seed>_summary.json"
Write-Host "  results/analysis/patch_resurrect_kl_forget0_10shot_seed_$TransferSeed`_transfer_eval.json"
