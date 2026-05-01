param(
    [string]$EnvName = "forgetgate",
    [string]$Config = "configs/experiment_suites.yaml",
    [switch]$UseActivePython,
    [int[]]$Seeds = @(42, 123),
    [switch]$IncludeScrub,
    [switch]$RunSweep,
    [string]$SweepLambdas = "0.0,0.1,0.5,1.0,2.0,5.0",
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

$guide = "oracle_contrastive_probe"
$baseSuites = @("vpt_resurrect_kl_forget0_10shot")
if ($IncludeScrub) {
    $baseSuites += "vpt_resurrect_scrub_forget0_10shot"
}

foreach ($suite in $baseSuites) {
    foreach ($seed in $Seeds) {
        Invoke-PythonStep `
            -Name "$suite seed=$seed guide=$guide" `
            -CommandArgs @(
                "scripts/3_train_vpt_resurrector.py",
                "--config", $Config,
                "--suite", $suite,
                "--seed", "$seed",
                "--feature-guide", $guide,
                "--feature-guide-weight", "$FeatureGuideWeight",
                "--feature-contrast-weight", "$ContrastWeight",
                "--feature-contrast-margin", "$ContrastMargin",
                "--num-workers", "$NumWorkers"
            )
    }
}

if ($RunSweep) {
    Invoke-PythonStep `
        -Name "tradeoff sweep seed=42 guide=$guide" `
        -CommandArgs @(
            "scripts/11_vpt_tradeoff_sweep.py",
            "--config", $Config,
            "--suite", "vpt_resurrect_kl_forget0_10shot",
            "--seed", "42",
            "--lambdas", $SweepLambdas,
            "--feature-guide", $guide,
            "--feature-guide-weight", "$FeatureGuideWeight",
            "--feature-contrast-weight", "$ContrastWeight",
            "--feature-contrast-margin", "$ContrastMargin"
        )
}

Write-Host ""
Write-Host "Oracle-contrastive probe run complete."
Write-Host "Check summaries under results/analysis/*oraclecontrastiveprobe*_summary.json"
