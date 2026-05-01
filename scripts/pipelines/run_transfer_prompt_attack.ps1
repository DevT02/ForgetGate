param(
    [string]$EnvName = "forgetgate",
    [string]$Config = "configs/experiment_suites.yaml",
    [switch]$UseActivePython,
    [string]$PromptDir,
    [int]$PromptSeed = 42,
    [string[]]$TargetSuites = @(
        "unlearn_kl_vit_cifar10_forget0",
        "unlearn_salun_vit_cifar10_forget0",
        "unlearn_scrub_distill_vit_cifar10_forget0",
        "unlearn_orbit_vit_cifar10_forget0",
        "oracle_vit_cifar10_forget0"
    ),
    [int]$EvalBatchSize = 256,
    [int]$NumWorkers = 0,
    [string]$Output = "",
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

if (-not $PromptDir) {
    $PromptDir = "checkpoints/vpt_resurrector/vpt_resurrect_kl_forget0_10shot_fgoraclecontrastiveprobe_seed_$PromptSeed"
}

$commandArgs = @(
    "scripts/transfer_prompt_attack.py",
    "--config", $Config,
    "--prompt-dir", $PromptDir,
    "--prompt-seed", "$PromptSeed",
    "--target-suites"
) + $TargetSuites + @(
    "--eval-batch-size", "$EvalBatchSize",
    "--num-workers", "$NumWorkers"
)

if ($Output) {
    $commandArgs += @("--output", $Output)
}

Invoke-PythonStep -Name "transfer prompt attack" -CommandArgs $commandArgs

Write-Host ""
Write-Host "Transfer prompt evaluation complete."
