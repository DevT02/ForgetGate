param(
    [string]$EnvName = "forgetgate",
    [string]$Config = "configs/experiment_suites.yaml",
    [switch]$UseActivePython,
    [int[]]$Seeds = @(42, 123),
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [switch]$SkipAttack,
    [float]$FeatureGuideWeight = 1.0,
    [float]$ContrastWeight = 1.0,
    [float]$ContrastMargin = 0.0,
    [int]$NumWorkers = 0,
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

$trainSuite = "unlearn_orbit_vit_cifar10_forget0"
$evalSuite = "eval_orbit_current_vit_cifar10_forget0"
$attackSuite = "vpt_resurrect_orbit_forget0_10shot"
$attackGuide = "oracle_contrastive_probe"

if (-not $SkipTrain) {
    foreach ($seed in $Seeds) {
        Invoke-PythonStep `
            -Name "train ORBIT seed=$seed" `
            -CommandArgs @(
                "scripts/2_train_unlearning_lora.py",
                "--config", $Config,
                "--suite", $trainSuite,
                "--seed", "$seed"
            )
    }
}

if (-not $SkipEval) {
    foreach ($seed in $Seeds) {
        Invoke-PythonStep `
            -Name "clean eval seed=$seed" `
            -CommandArgs @(
                "scripts/4_adv_evaluate.py",
                "--config", $Config,
                "--suite", $evalSuite,
                "--seed", "$seed"
            )
    }
}

if (-not $SkipAttack) {
    foreach ($seed in $Seeds) {
        Invoke-PythonStep `
            -Name "attack ORBIT seed=$seed guide=$attackGuide" `
            -CommandArgs @(
                "scripts/3_train_vpt_resurrector.py",
                "--config", $Config,
                "--suite", $attackSuite,
                "--seed", "$seed",
                "--feature-guide", $attackGuide,
                "--feature-guide-weight", "$FeatureGuideWeight",
                "--feature-contrast-weight", "$ContrastWeight",
                "--feature-contrast-margin", "$ContrastMargin",
                "--num-workers", "$NumWorkers"
            )
    }
}

Write-Host ""
Write-Host "ORBIT pipeline complete."
Write-Host "Key outputs:"
Write-Host "  checkpoints/unlearn_lora/unlearn_orbit_vit_cifar10_forget0_seed_<seed>"
Write-Host "  results/logs/eval_orbit_current_vit_cifar10_forget0_seed_<seed>_evaluation.json"
Write-Host "  results/analysis/vpt_resurrect_orbit_forget0_10shot_fgoraclecontrastiveprobe_seed_<seed>_summary.json"
