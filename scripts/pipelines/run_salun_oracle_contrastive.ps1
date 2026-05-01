param(
    [string]$EnvName = "forgetgate",
    [string]$Config = "configs/experiment_suites.yaml",
    [switch]$UseActivePython,
    [int[]]$Seeds = @(42, 123),
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

$suite = "vpt_resurrect_salun_forget0_10shot"
$guide = "oracle_contrastive_probe"

foreach ($seed in $Seeds) {
    Invoke-PythonStep `
        -Name "attack SalUn seed=$seed guide=$guide" `
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

Write-Host ""
Write-Host "SalUn oracle-contrastive run complete."
Write-Host "Check summaries under results/analysis/vpt_resurrect_salun_forget0_10shot_fgoraclecontrastiveprobe_seed_<seed>_summary.json"
