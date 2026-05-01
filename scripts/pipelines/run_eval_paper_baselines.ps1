param(
    [string]$EnvName = "forgetgate",
    [string]$Config = "configs/experiment_suites.yaml",
    [string]$Suite = "eval_paper_baselines_vit_cifar10_forget0",
    [switch]$UseActivePython,
    [int[]]$Seeds = @(42, 123, 456),
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

foreach ($seed in $Seeds) {
    Invoke-PythonStep `
        -Name "paper baselines seed=$seed" `
        -CommandArgs @(
            "scripts/4_adv_evaluate.py",
            "--config", $Config,
            "--suite", $Suite,
            "--seed", "$seed"
        )
}

Write-Host ""
Write-Host "Paper baseline eval run complete."
Write-Host "Expected outputs:"
foreach ($seed in $Seeds) {
    Write-Host "  results/logs/$($Suite)_seed_$seed`_evaluation.json"
}
