param(
    [string]$EnvName = "forgetgate",
    [string]$Config = "configs/experiment_suites.yaml",
    [switch]$UseActivePython,
    [switch]$IncludeSalUn,
    [string]$StatePath = "results/analysis/rigor_resume_state.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

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

function Ensure-ParentDir {
    param([string]$Path)

    $parent = Split-Path -Parent $Path
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
}

function Load-State {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return @{ steps = @{} }
    }

    try {
        $raw = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        $state = @{ steps = @{} }
        if ($null -ne $raw -and $raw.PSObject.Properties.Name -contains "steps") {
            foreach ($stepProp in $raw.steps.PSObject.Properties) {
                $entry = @{}
                foreach ($field in $stepProp.Value.PSObject.Properties) {
                    $entry[$field.Name] = $field.Value
                }
                $state.steps[$stepProp.Name] = $entry
            }
        }
        return $state
    } catch {
        Write-Warning "State file '$Path' is invalid. Starting a fresh state."
        return @{ steps = @{} }
    }
}

function Save-State {
    param(
        [hashtable]$State,
        [string]$Path
    )

    Ensure-ParentDir -Path $Path
    $State.updated_at = (Get-Date).ToString("s")
    $State | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $Path -Encoding utf8
}

function Test-JsonHasModel {
    param(
        [string]$Path,
        [string]$ModelName
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    try {
        $json = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        return $json.PSObject.Properties.Name -contains $ModelName
    } catch {
        return $false
    }
}

function Test-JsonlHasKey {
    param(
        [string]$Path,
        [string]$Key
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    try {
        $lines = Get-Content -LiteralPath $Path
        if (-not $lines -or $lines.Count -eq 0) {
            return $false
        }
        $last = $lines[-1] | ConvertFrom-Json
        return $last.PSObject.Properties.Name -contains $Key
    } catch {
        return $false
    }
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$CommandArgs,
        [scriptblock]$IsComplete,
        [hashtable]$State,
        [scriptblock]$PostRunCheck = $null,
        [switch]$AlwaysRun
    )

    if (-not $AlwaysRun -and (& $IsComplete)) {
        Write-Host "[SKIP] $Name"
        $State.steps[$Name] = @{
            status = "skipped"
            checked_at = (Get-Date).ToString("s")
        }
        Save-State -State $State -Path $StatePath
        return
    }

    $runner = Get-PythonRunner
    $fullArgs = @($runner.Args + $CommandArgs)
    $cmdText = "$($runner.Label) $($CommandArgs -join ' ')"
    Write-Host "[RUN ] $Name"
    Write-Host "       $cmdText"

    $State.steps[$Name] = @{
        status = "running"
        command = $cmdText
        started_at = (Get-Date).ToString("s")
    }
    Save-State -State $State -Path $StatePath

    & $runner.Exe @fullArgs
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        $State.steps[$Name] = @{
            status = "failed"
            command = $cmdText
            failed_at = (Get-Date).ToString("s")
            exit_code = $exitCode
        }
        Save-State -State $State -Path $StatePath
        throw "Step '$Name' failed with exit code $exitCode"
    }

    $verified = $true
    if ($null -ne $PostRunCheck) {
        $verified = (& $PostRunCheck)
    } elseif ($null -ne $IsComplete) {
        $verified = (& $IsComplete)
    }

    if (-not $verified) {
        $State.steps[$Name] = @{
            status = "incomplete"
            command = $cmdText
            finished_at = (Get-Date).ToString("s")
        }
        Save-State -State $State -Path $StatePath
        throw "Step '$Name' finished but the expected output check did not pass"
    }

    $State.steps[$Name] = @{
        status = "completed"
        command = $cmdText
        completed_at = (Get-Date).ToString("s")
    }
    Save-State -State $State -Path $StatePath
}

$State = Load-State -Path $StatePath

$steps = @(
    @{
        Name = "eval_autoattack_seed_456"
        Args = @("scripts/4_adv_evaluate.py", "--config", $Config, "--suite", "eval_autoattack_vit_cifar10_forget0", "--seed", "456")
        Check = { Test-JsonHasModel "results/logs/eval_autoattack_vit_cifar10_forget0_seed_456_evaluation.json" "vpt_resurrect_vit_cifar10_forget0" }
    },
    @{
        Name = "eval_dependency_seed_456"
        Args = @("scripts/4_adv_evaluate.py", "--config", $Config, "--suite", "eval_dependency_vit_cifar10_forget0", "--seed", "456")
        Check = { Test-JsonHasModel "results/logs/eval_dependency_vit_cifar10_forget0_seed_456_evaluation.json" "vpt_resurrect_vit_cifar10_forget0" }
    },
    @{
        Name = "eval_dependency_mix_seed_123"
        Args = @("scripts/4_adv_evaluate.py", "--config", $Config, "--suite", "eval_dependency_mix_vit_cifar10_forget0", "--seed", "123")
        Check = { Test-JsonHasModel "results/logs/eval_dependency_mix_vit_cifar10_forget0_seed_123_evaluation.json" "vpt_resurrect_vit_cifar10_forget0" }
    },
    @{
        Name = "eval_dependency_mix_seed_456"
        Args = @("scripts/4_adv_evaluate.py", "--config", $Config, "--suite", "eval_dependency_mix_vit_cifar10_forget0", "--seed", "456")
        Check = { Test-JsonHasModel "results/logs/eval_dependency_mix_vit_cifar10_forget0_seed_456_evaluation.json" "vpt_resurrect_vit_cifar10_forget0" }
    },
    @{
        Name = "vpt_scrub_10shot_seed_123"
        Args = @("scripts/3_train_vpt_resurrector.py", "--config", $Config, "--suite", "vpt_resurrect_scrub_forget0_10shot", "--seed", "123", "--num-workers", "0")
        Check = { Test-JsonlHasKey "results/logs/vpt_resurrect_scrub_forget0_10shot_seed_123.jsonl" "resurrection_acc" }
    },
    @{
        Name = "vpt_scrub_10shot_seed_456"
        Args = @("scripts/3_train_vpt_resurrector.py", "--config", $Config, "--suite", "vpt_resurrect_scrub_forget0_10shot", "--seed", "456", "--num-workers", "0")
        Check = { Test-JsonlHasKey "results/logs/vpt_resurrect_scrub_forget0_10shot_seed_456.jsonl" "resurrection_acc" }
    },
    @{
        Name = "vpt_scrub_25shot_seed_123"
        Args = @("scripts/3_train_vpt_resurrector.py", "--config", $Config, "--suite", "vpt_resurrect_scrub_forget0_10shot", "--seed", "123", "--k-shot", "25", "--num-workers", "0")
        Check = { Test-JsonlHasKey "results/logs/vpt_resurrect_scrub_forget0_10shot_kshot25_seed_123.jsonl" "resurrection_acc" }
    },
    @{
        Name = "vpt_scrub_25shot_seed_456"
        Args = @("scripts/3_train_vpt_resurrector.py", "--config", $Config, "--suite", "vpt_resurrect_scrub_forget0_10shot", "--seed", "456", "--k-shot", "25", "--num-workers", "0")
        Check = { Test-JsonlHasKey "results/logs/vpt_resurrect_scrub_forget0_10shot_kshot25_seed_456.jsonl" "resurrection_acc" }
    },
    @{
        Name = "vpt_scrub_50shot_seed_456"
        Args = @("scripts/3_train_vpt_resurrector.py", "--config", $Config, "--suite", "vpt_resurrect_scrub_forget0_50shot", "--seed", "456", "--num-workers", "0")
        Check = { Test-JsonlHasKey "results/logs/vpt_resurrect_scrub_forget0_50shot_seed_456.jsonl" "resurrection_acc" }
    },
    @{
        Name = "vpt_scrub_100shot_seed_456"
        Args = @("scripts/3_train_vpt_resurrector.py", "--config", $Config, "--suite", "vpt_resurrect_scrub_forget0_100shot", "--seed", "456", "--num-workers", "0")
        Check = { Test-JsonlHasKey "results/logs/vpt_resurrect_scrub_forget0_100shot_seed_456.jsonl" "resurrection_acc" }
    }
)

if ($IncludeSalUn) {
    $steps += @(
        @{
            Name = "vpt_salun_50shot_seed_456"
            Args = @("scripts/3_train_vpt_resurrector.py", "--config", $Config, "--suite", "vpt_resurrect_salun_forget0_50shot", "--seed", "456", "--num-workers", "0")
            Check = { Test-JsonlHasKey "results/logs/vpt_resurrect_salun_forget0_50shot_seed_456.jsonl" "resurrection_acc" }
        },
        @{
            Name = "vpt_salun_100shot_seed_456"
            Args = @("scripts/3_train_vpt_resurrector.py", "--config", $Config, "--suite", "vpt_resurrect_salun_forget0_100shot", "--seed", "456", "--num-workers", "0")
            Check = { Test-JsonlHasKey "results/logs/vpt_resurrect_salun_forget0_100shot_seed_456.jsonl" "resurrection_acc" }
        }
    )
}

foreach ($step in $steps) {
    Invoke-Step -Name $step.Name -CommandArgs $step.Args -IsComplete $step.Check -State $State
}

$summaryArgs = @("scripts/summarize_kshot_results.py", "--seeds", "42", "123", "456")
$manifestArgs = @("scripts/audit_results_manifest.py", "--seeds", "42", "123", "456", "--prompt-seeds", "42", "123")
$testArgs = @("-m", "pytest", "tests/test_basic.py", "tests/test_high_value_artifacts.py", "tests/test_integrity.py", "-q", "-p", "no:cacheprovider")

Invoke-Step -Name "summarize_kshot_results" -CommandArgs $summaryArgs -IsComplete {
    Test-Path -LiteralPath "results/analysis/kshot_summary.md"
} -PostRunCheck {
    Test-Path -LiteralPath "results/analysis/kshot_summary.md"
} -AlwaysRun -State $State

Invoke-Step -Name "audit_results_manifest" -CommandArgs $manifestArgs -IsComplete {
    Test-Path -LiteralPath "results/analysis/manifest_report.md"
} -PostRunCheck {
    Test-Path -LiteralPath "results/analysis/manifest_report.md"
} -AlwaysRun -State $State

Invoke-Step -Name "pytest_artifacts" -CommandArgs $testArgs -IsComplete { $false } -PostRunCheck { $true } -AlwaysRun -State $State

Write-Host ""
Write-Host "Rigor run finished."
Write-Host "State saved to $StatePath"
if ($IncludeSalUn) {
    Write-Host "SalUn VPT steps were included."
}
