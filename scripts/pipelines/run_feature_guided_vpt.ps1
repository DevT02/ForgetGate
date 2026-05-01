param(
    [string]$EnvName = "forgetgate",
    [string]$Config = "configs/experiment_suites.yaml",
    [switch]$UseActivePython,
    [int[]]$Seeds = @(42, 123, 456),
    [string[]]$FeatureGuides = @("linear_probe", "prototype_margin"),
    [switch]$SkipOracle,
    [switch]$SkipKL,
    [switch]$SkipSweeps,
    [switch]$SkipAudit,
    [string]$SweepLambdas = "0.0,0.1,0.5,1.0,2.0,5.0",
    [int[]]$SweepSeeds = @(42),
    [int]$NumWorkers = 0
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
    & $runner.Exe @fullArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step '$Name' failed with exit code $LASTEXITCODE"
    }
}

$oracleSuite = "vpt_oracle_vit_cifar10_forget0_10shot"
$klSuite = "vpt_resurrect_kl_forget0_10shot"

foreach ($seed in $Seeds) {
    foreach ($featureGuide in $FeatureGuides) {
        if (-not $SkipOracle) {
            Invoke-PythonStep `
                -Name "oracle seed=$seed guide=$featureGuide" `
                -CommandArgs @(
                    "scripts/3_train_vpt_resurrector.py",
                    "--config", $Config,
                    "--suite", $oracleSuite,
                    "--seed", "$seed",
                    "--feature-guide", $featureGuide,
                    "--feature-guide-weight", "1.0",
                    "--num-workers", "$NumWorkers"
                )
        }

        if (-not $SkipKL) {
            Invoke-PythonStep `
                -Name "kl seed=$seed guide=$featureGuide" `
                -CommandArgs @(
                    "scripts/3_train_vpt_resurrector.py",
                    "--config", $Config,
                    "--suite", $klSuite,
                    "--seed", "$seed",
                    "--feature-guide", $featureGuide,
                    "--feature-guide-weight", "1.0",
                    "--num-workers", "$NumWorkers"
                )
        }
    }
}

if (-not $SkipSweeps) {
    foreach ($sweepSeed in $SweepSeeds) {
        foreach ($featureGuide in $FeatureGuides) {
            Invoke-PythonStep `
                -Name "tradeoff sweep seed=$sweepSeed guide=$featureGuide" `
                -CommandArgs @(
                    "scripts/11_vpt_tradeoff_sweep.py",
                    "--config", $Config,
                    "--suite", $klSuite,
                    "--seed", "$sweepSeed",
                    "--lambdas", $SweepLambdas,
                    "--feature-guide", $featureGuide,
                    "--feature-guide-weight", "1.0"
                )
        }
    }
}

if (-not $SkipAudit) {
    $auditArgs = @(
        "scripts/audit_feature_guided_vpt.py",
        "--log-dir", "results/logs",
        "--analysis-dir", "results/analysis",
        "--checkpoint-dir", "checkpoints/vpt_resurrector",
        "--seeds"
    ) + ($Seeds | ForEach-Object { "$_" }) + @(
        "--sweep-seeds"
    ) + ($SweepSeeds | ForEach-Object { "$_" })

    Invoke-PythonStep `
        -Name "feature-guided audit" `
        -CommandArgs $auditArgs
}

Write-Host ""
Write-Host "Feature-guided VPT run complete."
Write-Host "Audit report: results/analysis/feature_guided_vpt_audit.md"
